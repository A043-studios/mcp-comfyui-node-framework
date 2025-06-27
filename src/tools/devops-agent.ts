import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { promises as fs } from 'fs';
import { join, dirname } from 'path';
import { v4 as uuidv4 } from 'uuid';
import * as tar from 'tar';

import { ExecutionError, ValidationError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('DevOpsAgent');

// Input validation schemas
const PackagingInputSchema = z.object({
  nodeCode: z.string().min(1, 'Node code is required'),
  nodeName: z.string().min(1, 'Node name is required'),
  version: z.string().default('1.0.0'),
  dependencies: z.array(z.string()).default([]),
  options: z.object({
    includeDockerfile: z.boolean().default(true),
    includeCI: z.boolean().default(true),
    packageManager: z.enum(['npm', 'pip', 'conda']).default('pip'),
    ciProvider: z.enum(['github', 'gitlab', 'jenkins']).default('github'),
    includeTests: z.boolean().default(true),
    generateManifest: z.boolean().default(true),
  }).optional(),
});

const DeploymentInputSchema = z.object({
  packagePath: z.string().min(1, 'Package path is required'),
  target: z.enum(['local', 'docker', 'cloud', 'registry']).default('local'),
  options: z.object({
    registry: z.string().optional(),
    environment: z.enum(['development', 'staging', 'production']).default('development'),
    autoScale: z.boolean().default(false),
    monitoring: z.boolean().default(true),
    backup: z.boolean().default(true),
  }).optional(),
});

type PackagingInput = z.infer<typeof PackagingInputSchema>;
type DeploymentInput = z.infer<typeof DeploymentInputSchema>;

export class DevOpsAgent {
  private config = getConfig();
  private templates = new Map<string, string>();

  constructor() {
    this.loadDevOpsTemplates();
  }

  /**
   * Package ComfyUI node for distribution
   */
  async packageNode(input: PackagingInput): Promise<{
    packageFiles: Record<string, string>;
    manifest: Record<string, any>;
    buildArtifacts: Record<string, string>;
    metadata: {
      packageId: string;
      version: string;
      size: number;
      createdAt: string;
      checksum: string;
    };
  }> {
    const validatedInput = PackagingInputSchema.parse(input);
    const { nodeCode, nodeName, version, dependencies, options = {} } = validatedInput;

    logger.info(`Packaging node: ${nodeName} v${version}`);

    try {
      // Generate package structure
      const packageFiles: Record<string, string> = {};
      
      // Main node file
      packageFiles[`${nodeName.toLowerCase()}.py`] = nodeCode;
      
      // Package metadata
      packageFiles['__init__.py'] = this.generateInitFile(nodeName);
      packageFiles['setup.py'] = this.generateSetupPy(nodeName, version, dependencies);
      packageFiles['requirements.txt'] = dependencies.join('\n');
      packageFiles['pyproject.toml'] = this.generatePyprojectToml(nodeName, version, dependencies);
      
      // Documentation
      packageFiles['README.md'] = this.generatePackageReadme(nodeName, version);
      packageFiles['LICENSE'] = this.generateLicense();
      
      // Configuration
      if (options?.generateManifest) {
        packageFiles['manifest.json'] = this.generateManifest(nodeName, version, dependencies);
      }

      // Docker support
      if (options?.includeDockerfile) {
        packageFiles['Dockerfile'] = this.generateDockerfile(nodeName, dependencies);
        packageFiles['docker-compose.yml'] = this.generateDockerCompose(nodeName);
        packageFiles['.dockerignore'] = this.generateDockerIgnore();
      }

      // CI/CD configuration
      if (options?.includeCI) {
        const ciFiles = this.generateCIConfig(nodeName, options?.ciProvider || 'github');
        Object.assign(packageFiles, ciFiles);
      }

      // Test configuration
      if (options?.includeTests) {
        packageFiles['pytest.ini'] = this.generatePytestConfig();
        packageFiles['tox.ini'] = this.generateToxConfig();
      }
      
      // Build artifacts
      const buildArtifacts = await this.generateBuildArtifacts(packageFiles, nodeName, version);
      
      // Generate manifest
      const manifest = this.createPackageManifest(nodeName, version, dependencies, packageFiles);
      
      // Calculate metadata
      const packageSize = Object.values(packageFiles).reduce((total, content) => total + content.length, 0);
      const checksum = this.calculateChecksum(packageFiles);

      const result = {
        packageFiles,
        manifest,
        buildArtifacts,
        metadata: {
          packageId: uuidv4(),
          version,
          size: packageSize,
          createdAt: new Date().toISOString(),
          checksum,
        },
      };

      logger.info(`Successfully packaged ${nodeName} v${version} (${packageSize} bytes)`);
      return result;

    } catch (error) {
      logger.error(`Failed to package node: ${error}`);
      throw new ExecutionError(`Packaging failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Deploy packaged node to target environment
   */
  async deployNode(input: DeploymentInput): Promise<{
    deploymentId: string;
    status: 'success' | 'failed' | 'pending';
    endpoint?: string;
    logs: string[];
    monitoring: {
      healthCheck: string;
      metrics: string[];
      alerts: string[];
    };
  }> {
    const validatedInput = DeploymentInputSchema.parse(input);
    const { packagePath, target, options = {} } = validatedInput;

    logger.info(`Deploying package to ${target} environment`);

    try {
      const deploymentId = uuidv4();
      const logs: string[] = [];
      
      logs.push(`Starting deployment ${deploymentId}`);
      logs.push(`Target: ${target}`);
      logs.push(`Environment: ${options?.environment || 'development'}`);
      
      // Simulate deployment process
      let endpoint: string | undefined;
      let status: 'success' | 'failed' | 'pending' = 'pending';
      
      switch (target) {
        case 'local':
          endpoint = await this.deployLocal(packagePath, logs);
          status = 'success';
          break;
        case 'docker':
          endpoint = await this.deployDocker(packagePath, logs);
          status = 'success';
          break;
        case 'cloud':
          endpoint = await this.deployCloud(packagePath, options, logs);
          status = 'success';
          break;
        case 'registry':
          endpoint = await this.deployRegistry(packagePath, options, logs);
          status = 'success';
          break;
      }
      
      // Setup monitoring
      const monitoring = this.setupMonitoring(deploymentId, endpoint, options);
      
      logs.push(`Deployment completed successfully`);
      logs.push(`Endpoint: ${endpoint || 'N/A'}`);

      return {
        deploymentId,
        status,
        endpoint,
        logs,
        monitoring,
      };

    } catch (error) {
      logger.error(`Deployment failed: ${error}`);
      throw new ExecutionError(`Deployment failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Setup CI/CD pipeline for automated deployment
   */
  async setupCIPipeline(nodeName: string, repository: string, provider: 'github' | 'gitlab' | 'jenkins' = 'github'): Promise<{
    pipelineConfig: Record<string, string>;
    webhooks: string[];
    secrets: string[];
    documentation: string;
  }> {
    logger.info(`Setting up CI/CD pipeline for ${nodeName} on ${provider}`);

    try {
      const pipelineConfig = this.generateCIConfig(nodeName, provider);
      const webhooks = this.generateWebhookConfig(repository, provider);
      const secrets = this.getRequiredSecrets(provider);
      const documentation = this.generateCIDocumentation(nodeName, provider);

      return {
        pipelineConfig,
        webhooks,
        secrets,
        documentation,
      };

    } catch (error) {
      logger.error(`CI/CD setup failed: ${error}`);
      throw new ExecutionError(`CI/CD setup failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private async loadDevOpsTemplates(): Promise<void> {
    this.templates.set('dockerfile', `
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Expose port (if needed)
EXPOSE 8000

# Run the application
CMD ["python", "-m", "{{NODE_NAME}}"]
`);

    this.templates.set('github_workflow', `
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python \${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: \${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      run: |
        pytest tests/ --cov={{NODE_NAME}} --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Build package
      run: |
        python -m pip install --upgrade pip build
        python -m build
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying {{NODE_NAME}} to production"
        # Add deployment steps here
`);

    this.templates.set('setup_py', `
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{{NODE_NAME}}",
    version="{{VERSION}}",
    author="ComfyUI Framework",
    author_email="contact@example.com",
    description="{{DESCRIPTION}}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/{{NODE_NAME}}",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "{{NODE_NAME}}={{NODE_NAME}}.cli:main",
        ],
    },
)
`);
  }

  private generateInitFile(nodeName: string): string {
    return `"""
${nodeName} - ComfyUI Node Package
"""

__version__ = "1.0.0"
__author__ = "ComfyUI Framework"

from .${nodeName.toLowerCase()} import ${nodeName}

__all__ = ["${nodeName}"]
`;
  }

  private generateSetupPy(nodeName: string, version: string, dependencies: string[]): string {
    const template = this.templates.get('setup_py') || '';
    return template
      .replace(/{{NODE_NAME}}/g, nodeName.toLowerCase())
      .replace(/{{VERSION}}/g, version)
      .replace(/{{DESCRIPTION}}/g, `${nodeName} ComfyUI node package`);
  }

  private generatePyprojectToml(nodeName: string, version: string, dependencies: string[]): string {
    return `[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "${nodeName.toLowerCase()}"
version = "${version}"
description = "${nodeName} ComfyUI node package"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "ComfyUI Framework", email = "contact@example.com"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
requires-python = ">=3.8"
dependencies = [
${dependencies.map(dep => `    "${dep}",`).join('\n')}
]

[project.urls]
Homepage = "https://github.com/example/${nodeName.toLowerCase()}"
Repository = "https://github.com/example/${nodeName.toLowerCase()}.git"
Issues = "https://github.com/example/${nodeName.toLowerCase()}/issues"

[tool.setuptools]
packages = ["${nodeName.toLowerCase()}"]
`;
  }

  private generatePackageReadme(nodeName: string, version: string): string {
    return `# ${nodeName}

ComfyUI node package for ${nodeName}.

## Installation

\`\`\`bash
pip install ${nodeName.toLowerCase()}
\`\`\`

## Usage

\`\`\`python
from ${nodeName.toLowerCase()} import ${nodeName}

# Use the node in your ComfyUI workflow
node = ${nodeName}()
\`\`\`

## Version

${version}

## License

MIT License
`;
  }

  private generateLicense(): string {
    return `MIT License

Copyright (c) ${new Date().getFullYear()} ComfyUI Framework

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`;
  }

  private generateManifest(nodeName: string, version: string, dependencies: string[]): string {
    return JSON.stringify({
      name: nodeName,
      version,
      description: `${nodeName} ComfyUI node`,
      type: "comfyui-node",
      dependencies,
      author: "ComfyUI Framework",
      license: "MIT",
      repository: `https://github.com/example/${nodeName.toLowerCase()}`,
      keywords: ["comfyui", "node", "ai", "image-processing"],
      engines: {
        python: ">=3.8",
        comfyui: ">=1.0.0"
      },
      files: [
        `${nodeName.toLowerCase()}.py`,
        "__init__.py",
        "README.md",
        "LICENSE"
      ]
    }, null, 2);
  }

  private generateDockerfile(nodeName: string, dependencies: string[]): string {
    const template = this.templates.get('dockerfile') || '';
    return template.replace(/{{NODE_NAME}}/g, nodeName.toLowerCase());
  }

  private generateDockerCompose(nodeName: string): string {
    return `version: '3.8'

services:
  ${nodeName.toLowerCase()}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
`;
  }

  private generateDockerIgnore(): string {
    return `__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

.DS_Store
.vscode
.idea
*.swp
*.swo

node_modules
npm-debug.log*
yarn-debug.log*
yarn-error.log*

dist
build
*.egg-info
`;
  }

  private generateCIConfig(nodeName: string, provider: string): Record<string, string> {
    const configs: Record<string, string> = {};

    switch (provider) {
      case 'github':
        configs['.github/workflows/ci.yml'] = this.templates.get('github_workflow')?.replace(/{{NODE_NAME}}/g, nodeName.toLowerCase()) || '';
        break;
      case 'gitlab':
        configs['.gitlab-ci.yml'] = this.generateGitLabCI(nodeName);
        break;
      case 'jenkins':
        configs['Jenkinsfile'] = this.generateJenkinsfile(nodeName);
        break;
    }

    return configs;
  }

  private generateGitLabCI(nodeName: string): string {
    return `stages:
  - test
  - build
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip
    - venv/

test:
  stage: test
  image: python:3.9
  script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
    - pip install -e .
    - pytest tests/ --cov=${nodeName.toLowerCase()}

build:
  stage: build
  image: python:3.9
  script:
    - python -m pip install --upgrade pip build
    - python -m build
  artifacts:
    paths:
      - dist/

deploy:
  stage: deploy
  script:
    - echo "Deploying ${nodeName}"
  only:
    - main
`;
  }

  private generateJenkinsfile(nodeName: string): string {
    return `pipeline {
    agent any
    
    stages {
        stage('Test') {
            steps {
                sh 'python -m venv venv'
                sh 'source venv/bin/activate && pip install -r requirements.txt'
                sh 'source venv/bin/activate && pip install -e .'
                sh 'source venv/bin/activate && pytest tests/ --cov=${nodeName.toLowerCase()}'
            }
        }
        
        stage('Build') {
            steps {
                sh 'source venv/bin/activate && python -m build'
                archiveArtifacts artifacts: 'dist/*', fingerprint: true
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo 'Deploying ${nodeName}'
                // Add deployment steps
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
`;
  }

  private generatePytestConfig(): string {
    return `[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
`;
  }

  private generateToxConfig(): string {
    return `[tox]
envlist = py38,py39,py310,lint,type

[testenv]
deps = 
    pytest
    pytest-cov
commands = pytest {posargs}

[testenv:lint]
deps = 
    flake8
    black
    isort
commands = 
    flake8 src tests
    black --check src tests
    isort --check-only src tests

[testenv:type]
deps = 
    mypy
commands = mypy src
`;
  }

  private async generateBuildArtifacts(packageFiles: Record<string, string>, nodeName: string, version: string): Promise<Record<string, string>> {
    // Simulate building artifacts
    const artifacts: Record<string, string> = {};
    
    // Create wheel file content (simulated)
    artifacts[`${nodeName.toLowerCase()}-${version}-py3-none-any.whl`] = 'Binary wheel content';
    
    // Create source distribution (simulated)
    artifacts[`${nodeName.toLowerCase()}-${version}.tar.gz`] = 'Source distribution content';
    
    return artifacts;
  }

  private createPackageManifest(nodeName: string, version: string, dependencies: string[], packageFiles: Record<string, string>): Record<string, any> {
    return {
      name: nodeName,
      version,
      type: 'comfyui-node',
      dependencies,
      files: Object.keys(packageFiles),
      size: Object.values(packageFiles).reduce((total, content) => total + content.length, 0),
      checksum: this.calculateChecksum(packageFiles),
      createdAt: new Date().toISOString(),
    };
  }

  private calculateChecksum(files: Record<string, string>): string {
    // Simple checksum calculation (in real implementation would use proper hashing)
    const content = Object.values(files).join('');
    return Buffer.from(content).toString('base64').slice(0, 16);
  }

  private async deployLocal(packagePath: string, logs: string[]): Promise<string> {
    logs.push('Deploying to local environment');
    logs.push('Installing package locally');
    logs.push('Package installed successfully');
    return 'file://localhost/comfyui/custom_nodes';
  }

  private async deployDocker(packagePath: string, logs: string[]): Promise<string> {
    logs.push('Building Docker image');
    logs.push('Starting Docker container');
    logs.push('Container started successfully');
    return 'http://localhost:8000';
  }

  private async deployCloud(packagePath: string, options: any, logs: string[]): Promise<string> {
    logs.push('Uploading to cloud provider');
    logs.push('Configuring cloud resources');
    logs.push('Deployment completed');
    return 'https://cloud-provider.com/deployments/node-instance';
  }

  private async deployRegistry(packagePath: string, options: any, logs: string[]): Promise<string> {
    logs.push('Publishing to package registry');
    logs.push('Package published successfully');
    return options.registry || 'https://pypi.org/project/package';
  }

  private setupMonitoring(deploymentId: string, endpoint: string | undefined, options: any): any {
    return {
      healthCheck: endpoint ? `${endpoint}/health` : 'N/A',
      metrics: [
        'cpu_usage',
        'memory_usage',
        'request_count',
        'response_time',
      ],
      alerts: [
        'high_cpu_usage',
        'memory_leak',
        'service_down',
      ],
    };
  }

  private generateWebhookConfig(repository: string, provider: string): string[] {
    const webhooks: string[] = [];
    
    switch (provider) {
      case 'github':
        webhooks.push(`${repository}/settings/hooks`);
        break;
      case 'gitlab':
        webhooks.push(`${repository}/-/settings/integrations`);
        break;
      case 'jenkins':
        webhooks.push('Jenkins webhook configuration');
        break;
    }
    
    return webhooks;
  }

  private getRequiredSecrets(provider: string): string[] {
    const secrets = ['DEPLOYMENT_TOKEN', 'REGISTRY_PASSWORD'];
    
    switch (provider) {
      case 'github':
        secrets.push('GITHUB_TOKEN');
        break;
      case 'gitlab':
        secrets.push('GITLAB_TOKEN');
        break;
      case 'jenkins':
        secrets.push('JENKINS_API_TOKEN');
        break;
    }
    
    return secrets;
  }

  private generateCIDocumentation(nodeName: string, provider: string): string {
    return `# CI/CD Documentation for ${nodeName}

## Overview

This document describes the CI/CD pipeline setup for ${nodeName} using ${provider}.

## Pipeline Stages

1. **Test**: Run unit tests and code quality checks
2. **Build**: Create distribution packages
3. **Deploy**: Deploy to target environment

## Configuration

The pipeline is configured in the following files:
- CI configuration: See repository CI files
- Deployment scripts: See deployment directory
- Environment variables: Configure in ${provider} settings

## Secrets

The following secrets need to be configured:
${this.getRequiredSecrets(provider).map(secret => `- ${secret}`).join('\n')}

## Monitoring

- Health checks: Automated endpoint monitoring
- Metrics: Performance and usage metrics
- Alerts: Automated alerting for issues

## Troubleshooting

Common issues and solutions:
- Build failures: Check dependency versions
- Test failures: Review test logs
- Deployment issues: Verify environment configuration
`;
  }
}

// MCP Tool definitions
export const devopsAgentPackageTool: Tool = {
  name: 'devops_agent_package',
  description: 'Package ComfyUI node for distribution with Docker, CI/CD, and deployment configurations',
  inputSchema: {
    type: 'object',
    properties: {
      nodeCode: {
        type: 'string',
        description: 'The ComfyUI node code to package',
      },
      nodeName: {
        type: 'string',
        description: 'Name of the ComfyUI node',
      },
      version: {
        type: 'string',
        default: '1.0.0',
        description: 'Package version',
      },
      dependencies: {
        type: 'array',
        items: { type: 'string' },
        default: [],
        description: 'List of package dependencies',
      },
      options: {
        type: 'object',
        properties: {
          includeDockerfile: { type: 'boolean', default: true },
          includeCI: { type: 'boolean', default: true },
          packageManager: { type: 'string', enum: ['npm', 'pip', 'conda'], default: 'pip' },
          ciProvider: { type: 'string', enum: ['github', 'gitlab', 'jenkins'], default: 'github' },
          includeTests: { type: 'boolean', default: true },
          generateManifest: { type: 'boolean', default: true },
        },
      },
    },
    required: ['nodeCode', 'nodeName'],
  },
};

export const devopsAgentDeployTool: Tool = {
  name: 'devops_agent_deploy',
  description: 'Deploy packaged ComfyUI node to target environment',
  inputSchema: {
    type: 'object',
    properties: {
      packagePath: {
        type: 'string',
        description: 'Path to the package to deploy',
      },
      target: {
        type: 'string',
        enum: ['local', 'docker', 'cloud', 'registry'],
        default: 'local',
        description: 'Deployment target',
      },
      options: {
        type: 'object',
        properties: {
          registry: { type: 'string' },
          environment: { type: 'string', enum: ['development', 'staging', 'production'], default: 'development' },
          autoScale: { type: 'boolean', default: false },
          monitoring: { type: 'boolean', default: true },
          backup: { type: 'boolean', default: true },
        },
      },
    },
    required: ['packagePath'],
  },
};

export const devopsAgentCIPipelineTool: Tool = {
  name: 'devops_agent_ci_pipeline',
  description: 'Setup CI/CD pipeline for automated deployment',
  inputSchema: {
    type: 'object',
    properties: {
      nodeName: {
        type: 'string',
        description: 'Name of the ComfyUI node',
      },
      repository: {
        type: 'string',
        description: 'Repository URL',
      },
      provider: {
        type: 'string',
        enum: ['github', 'gitlab', 'jenkins'],
        default: 'github',
        description: 'CI/CD provider',
      },
    },
    required: ['nodeName', 'repository'],
  },
};
