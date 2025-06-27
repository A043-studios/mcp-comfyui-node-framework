/**
 * ComfyUI Framework MCP Server
 * 
 * Professional MCP server for AI-powered ComfyUI node generation from research papers and repositories.
 * 
 * @author A043 Studios
 * @version 1.0.0
 */

export { ComfyUIFrameworkServer } from './server';

// Export all agent classes for programmatic use
export { ResearchAgent } from './tools/research-agent';
export { CodingAgent } from './tools/coding-agent';
export { TestingAgent } from './tools/testing-agent';
export { DocumentationAgent } from './tools/documentation-agent';
export { DevOpsAgent } from './tools/devops-agent';
export { AdvancedWebScraper } from './tools/web-scraper';
export { CodeAnalyzer } from './tools/code-analyzer';
export { WorkflowOrchestrator } from './tools/workflow-orchestrator';

// Export configuration and utilities
export { getConfig, updateConfig, validateConfig } from './config';
export { createLogger } from './utils/logger';

// Export types
export * from './types';

// Export resources and prompts
export { resources, handleResourceRequest } from './resources';
export { prompts, handlePromptRequest } from './prompts';

// Version information
export const VERSION = '1.0.0';
export const NAME = 'ComfyUI Framework MCP Server';
export const DESCRIPTION = 'Professional MCP server for AI-powered ComfyUI node generation';

/**
 * Default configuration for the framework
 */
export const DEFAULT_CONFIG = {
  server: {
    name: NAME,
    version: VERSION,
    description: DESCRIPTION,
  },
  framework: {
    defaultOutputDir: './output',
    defaultQuality: 'production' as const,
    supportedAgents: ['research', 'coding', 'testing', 'documentation', 'devops'],
    defaultAgents: ['research', 'coding', 'testing', 'documentation'],
    maxConcurrentExecutions: 5,
    executionTimeout: 3600000, // 1 hour
  },
  agents: {
    research: {
      timeout: 300000, // 5 minutes
      maxContentLength: 1000000, // 1MB
      supportedSources: ['arxiv', 'github', 'url', 'file'],
      cacheResults: true,
      cacheTtl: 86400000, // 24 hours
    },
    coding: {
      timeout: 600000, // 10 minutes
      templatePath: './templates/comfyui',
      outputFormat: 'python' as const,
      includeTests: true,
      codeStyle: 'standard' as const,
    },
    testing: {
      timeout: 300000, // 5 minutes
      framework: 'pytest' as const,
      coverage: true,
      minCoverage: 80,
      includeIntegrationTests: true,
    },
    documentation: {
      timeout: 180000, // 3 minutes
      format: 'markdown' as const,
      includeExamples: true,
      includeApiDocs: true,
      generateReadme: true,
    },
    devops: {
      timeout: 120000, // 2 minutes
      includeDockerfile: true,
      includeCi: true,
      ciProvider: 'github' as const,
      packageManager: 'pip' as const,
    },
  },
  scraping: {
    rateLimitDelay: 1000,
    timeout: 30000,
    userAgent: 'ComfyUI-Framework-MCP/1.0',
    maxRetries: 3,
    supportedMethods: ['auto', 'arxiv', 'github', 'puppeteer', 'cheerio', 'jsdom'],
    puppeteer: {
      headless: true,
      timeout: 30000,
      viewport: {
        width: 1920,
        height: 1080,
      },
    },
  },
  llm: {
    provider: 'openrouter' as const,
    model: 'anthropic/claude-3.5-sonnet',
    maxTokens: 4000,
    temperature: 0.1,
  },
  logging: {
    level: 'info' as const,
    format: 'detailed' as const,
    maxFiles: 5,
    maxSize: '10m',
  },
  storage: {
    type: 'filesystem' as const,
    path: './data',
    cleanup: {
      enabled: true,
      maxAge: 604800000, // 7 days
      maxSize: '1gb',
    },
  },
  security: {
    enableCors: true,
    allowedOrigins: ['*'],
    rateLimit: {
      enabled: true,
      windowMs: 900000, // 15 minutes
      maxRequests: 100,
    },
    sanitizeInputs: true,
    validateOutputs: true,
  },
};

/**
 * Workflow templates for common use cases
 */
export const WORKFLOW_TEMPLATES = {
  'quick-prototype': {
    name: 'Quick Prototype',
    description: 'Fast prototyping workflow for development',
    agents: ['research', 'coding'],
    estimatedTime: '30 minutes',
    complexity: 'low',
    configuration: {
      parallel: false,
      timeout: 1800000, // 30 minutes
      qualityLevel: 'development',
    },
  },
  'production-ready': {
    name: 'Production Ready',
    description: 'Complete workflow for production-ready nodes',
    agents: ['research', 'coding', 'testing', 'documentation', 'devops'],
    estimatedTime: '2 hours',
    complexity: 'high',
    configuration: {
      parallel: false,
      timeout: 7200000, // 2 hours
      qualityLevel: 'production',
    },
  },
  'research-only': {
    name: 'Research Analysis',
    description: 'Research and analysis only',
    agents: ['research'],
    estimatedTime: '15 minutes',
    complexity: 'low',
    configuration: {
      parallel: false,
      timeout: 900000, // 15 minutes
      qualityLevel: 'development',
    },
  },
  'code-review': {
    name: 'Code Review & Optimization',
    description: 'Review and optimize existing code',
    agents: ['coding', 'testing', 'documentation'],
    estimatedTime: '45 minutes',
    complexity: 'medium',
    configuration: {
      parallel: true,
      timeout: 2700000, // 45 minutes
      qualityLevel: 'production',
    },
  },
};

/**
 * Supported input source types
 */
export const INPUT_TYPES = {
  ARXIV: 'arxiv',
  GITHUB: 'github',
  URL: 'url',
  FILE: 'file',
} as const;

/**
 * Quality levels for generation
 */
export const QUALITY_LEVELS = {
  DRAFT: 'draft',
  DEVELOPMENT: 'development',
  PRODUCTION: 'production',
} as const;

/**
 * Supported agent types
 */
export const AGENT_TYPES = {
  RESEARCH: 'research',
  CODING: 'coding',
  TESTING: 'testing',
  DOCUMENTATION: 'documentation',
  DEVOPS: 'devops',
} as const;

/**
 * Framework capabilities
 */
export const CAPABILITIES = {
  // Research capabilities
  ARXIV_EXTRACTION: 'Extract content from arXiv papers',
  GITHUB_ANALYSIS: 'Analyze GitHub repositories',
  WEB_SCRAPING: 'Advanced web content scraping',
  PDF_PARSING: 'Parse PDF documents',
  CONTENT_ANALYSIS: 'Analyze and summarize content',
  
  // Code generation capabilities
  COMFYUI_NODES: 'Generate ComfyUI nodes',
  CODE_OPTIMIZATION: 'Optimize existing code',
  PATTERN_RECOGNITION: 'Recognize code patterns',
  DEPENDENCY_ANALYSIS: 'Analyze dependencies',
  CODE_VALIDATION: 'Validate code structure',
  
  // Testing capabilities
  UNIT_TESTS: 'Generate unit tests',
  INTEGRATION_TESTS: 'Create integration tests',
  PERFORMANCE_TESTS: 'Performance testing',
  COVERAGE_ANALYSIS: 'Test coverage analysis',
  
  // Documentation capabilities
  API_DOCS: 'Generate API documentation',
  USER_GUIDES: 'Create user guides',
  TUTORIALS: 'Generate tutorials',
  EXAMPLES: 'Create code examples',
  
  // DevOps capabilities
  PACKAGING: 'Package for distribution',
  CONTAINERIZATION: 'Docker containerization',
  CI_CD: 'CI/CD pipeline setup',
  DEPLOYMENT: 'Automated deployment',
  MONITORING: 'Setup monitoring',
  
  // Workflow capabilities
  ORCHESTRATION: 'Multi-agent orchestration',
  PARALLEL_EXECUTION: 'Parallel agent execution',
  ERROR_RECOVERY: 'Error handling and recovery',
  PROGRESS_TRACKING: 'Execution progress tracking',
};

/**
 * Framework metadata
 */
export const METADATA = {
  name: NAME,
  version: VERSION,
  description: DESCRIPTION,
  author: 'A043 Studios',
  license: 'MIT',
  repository: 'https://github.com/A043-studios/mcp-comfyui-framework',
  homepage: 'https://github.com/A043-studios/mcp-comfyui-framework#readme',
  bugs: 'https://github.com/A043-studios/mcp-comfyui-framework/issues',
  keywords: [
    'mcp',
    'model-context-protocol',
    'comfyui',
    'ai-automation',
    'code-generation',
    'research-to-code',
    'node-development',
    'typescript',
    'professional',
  ],
  engines: {
    node: '>=18.0.0',
    npm: '>=8.0.0',
  },
  capabilities: Object.values(CAPABILITIES),
  supportedInputTypes: Object.values(INPUT_TYPES),
  supportedQualityLevels: Object.values(QUALITY_LEVELS),
  supportedAgents: Object.values(AGENT_TYPES),
  workflowTemplates: Object.keys(WORKFLOW_TEMPLATES),
};
