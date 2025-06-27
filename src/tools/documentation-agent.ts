import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { promises as fs } from 'fs';
import { join, dirname } from 'path';
import { v4 as uuidv4 } from 'uuid';

import { ExecutionError, ValidationError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('DocumentationAgent');

// Input validation schemas
const DocumentationInputSchema = z.object({
  nodeCode: z.string().min(1, 'Node code is required'),
  nodeName: z.string().min(1, 'Node name is required'),
  researchContent: z.object({
    title: z.string(),
    content: z.string(),
    keyFindings: z.array(z.string()).optional(),
    technologies: z.array(z.string()).optional(),
  }).optional(),
  documentationTypes: z.array(z.enum(['readme', 'api', 'tutorial', 'examples', 'changelog'])).default(['readme', 'api']),
  options: z.object({
    format: z.enum(['markdown', 'rst', 'html']).default('markdown'),
    includeExamples: z.boolean().default(true),
    includeImages: z.boolean().default(false),
    generateToc: z.boolean().default(true),
    style: z.enum(['technical', 'user-friendly', 'comprehensive']).default('user-friendly'),
  }).optional(),
});

type DocumentationInput = z.infer<typeof DocumentationInputSchema>;

export class DocumentationAgent {
  private config = getConfig();
  private templates = new Map<string, string>();

  constructor() {
    this.loadDocumentationTemplates();
  }

  /**
   * Generate comprehensive documentation for ComfyUI nodes
   */
  async generateDocumentation(input: DocumentationInput): Promise<{
    documents: Record<string, string>;
    assets: Record<string, string>;
    metadata: {
      documentationId: string;
      generatedAt: string;
      nodeVersion: string;
      totalPages: number;
      wordCount: number;
    };
  }> {
    const validatedInput = DocumentationInputSchema.parse(input);
    const { nodeCode, nodeName, researchContent, documentationTypes, options = {} } = validatedInput;

    logger.info(`Generating documentation for node: ${nodeName}`);

    try {
      // Analyze node structure for documentation
      const nodeAnalysis = await this.analyzeNodeForDocumentation(nodeCode, nodeName);

      // Generate different types of documentation
      const documents: Record<string, string> = {};

      for (const docType of documentationTypes) {
        switch (docType) {
          case 'readme':
            documents['README.md'] = await this.generateReadme(nodeAnalysis, researchContent, options);
            break;
          case 'api':
            documents['API.md'] = await this.generateApiDocumentation(nodeAnalysis, options);
            break;
          case 'tutorial':
            documents['TUTORIAL.md'] = await this.generateTutorial(nodeAnalysis, options);
            break;
          case 'examples':
            documents['EXAMPLES.md'] = await this.generateExamples(nodeAnalysis, options);
            break;
          case 'changelog':
            documents['CHANGELOG.md'] = await this.generateChangelog(nodeAnalysis);
            break;
        }
      }

      // Generate assets (diagrams, images, etc.)
      const assets = options?.includeImages ? await this.generateAssets(nodeAnalysis) : {};

      // Calculate metadata
      const totalPages = Object.keys(documents).length;
      const wordCount = Object.values(documents).reduce((total, content) =>
        total + content.split(/\s+/).length, 0);

      const result = {
        documents,
        assets,
        metadata: {
          documentationId: uuidv4(),
          generatedAt: new Date().toISOString(),
          nodeVersion: '1.0.0',
          totalPages,
          wordCount,
        },
      };

      logger.info(`Generated ${totalPages} documentation pages for ${nodeName}`);
      return result;

    } catch (error) {
      logger.error(`Failed to generate documentation: ${error}`);
      throw new ExecutionError(`Documentation generation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Generate API reference documentation
   */
  async generateApiReference(nodeCode: string, nodeName: string): Promise<{
    apiDoc: string;
    schema: Record<string, any>;
    examples: Record<string, string>;
  }> {
    logger.info(`Generating API reference for: ${nodeName}`);

    try {
      const nodeAnalysis = await this.analyzeNodeForDocumentation(nodeCode, nodeName);

      const apiDoc = await this.generateApiDocumentation(nodeAnalysis, { format: 'markdown' });
      const schema = this.generateJsonSchema(nodeAnalysis);
      const examples = await this.generateApiExamples(nodeAnalysis);

      return {
        apiDoc,
        schema,
        examples,
      };

    } catch (error) {
      logger.error(`API reference generation failed: ${error}`);
      throw new ExecutionError(`API reference generation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Generate user tutorials and guides
   */
  async generateUserGuide(nodeCode: string, nodeName: string, userLevel: 'beginner' | 'intermediate' | 'advanced' = 'beginner'): Promise<{
    guide: string;
    quickStart: string;
    troubleshooting: string;
    faq: string;
  }> {
    logger.info(`Generating user guide for: ${nodeName} (${userLevel} level)`);

    try {
      const nodeAnalysis = await this.analyzeNodeForDocumentation(nodeCode, nodeName);

      const guide = await this.generateTutorial(nodeAnalysis, { style: userLevel === 'beginner' ? 'user-friendly' : 'technical' });
      const quickStart = await this.generateQuickStart(nodeAnalysis, userLevel);
      const troubleshooting = await this.generateTroubleshooting(nodeAnalysis);
      const faq = await this.generateFaq(nodeAnalysis);

      return {
        guide,
        quickStart,
        troubleshooting,
        faq,
      };

    } catch (error) {
      logger.error(`User guide generation failed: ${error}`);
      throw new ExecutionError(`User guide generation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private async loadDocumentationTemplates(): Promise<void> {
    this.templates.set('readme', `# {{NODE_NAME}}

{{DESCRIPTION}}

{{RESEARCH_CONTEXT}}

## Features

{{FEATURES}}

## Installation

\`\`\`bash
# Copy to ComfyUI custom nodes directory
cp -r {{NODE_NAME}} /path/to/ComfyUI/custom_nodes/

# Install dependencies
pip install -r requirements.txt

# Restart ComfyUI
\`\`\`

## Usage

{{USAGE_EXAMPLES}}

## API Reference

{{API_REFERENCE}}

## Examples

{{EXAMPLES}}

## Troubleshooting

{{TROUBLESHOOTING}}

## Contributing

{{CONTRIBUTING}}

## License

{{LICENSE}}
`);

    this.templates.set('api', `# {{NODE_NAME}} API Reference

## Overview

{{OVERVIEW}}

## Input Types

{{INPUT_TYPES}}

## Return Types

{{RETURN_TYPES}}

## Methods

{{METHODS}}

## Parameters

{{PARAMETERS}}

## Error Handling

{{ERROR_HANDLING}}

## Examples

{{API_EXAMPLES}}
`);

    this.templates.set('tutorial', `# {{NODE_NAME}} Tutorial

## Introduction

{{INTRODUCTION}}

## Prerequisites

{{PREREQUISITES}}

## Step-by-Step Guide

{{STEP_BY_STEP}}

## Advanced Usage

{{ADVANCED_USAGE}}

## Best Practices

{{BEST_PRACTICES}}

## Common Patterns

{{COMMON_PATTERNS}}

## Next Steps

{{NEXT_STEPS}}
`);
  }

  private async analyzeNodeForDocumentation(nodeCode: string, nodeName: string): Promise<{
    name: string;
    description: string;
    category: string;
    inputTypes: Record<string, any>;
    returnTypes: string[];
    functionName: string;
    methods: Array<{ name: string; signature: string; description: string }>;
    dependencies: string[];
    complexity: 'simple' | 'moderate' | 'complex';
    features: string[];
    useCases: string[];
  }> {
    // Parse node structure
    const categoryMatch = nodeCode.match(/CATEGORY\s*=\s*["']([^"']+)["']/);
    const descriptionMatch = nodeCode.match(/"""([^"]+)"""/);
    const inputTypesMatch = nodeCode.match(/INPUT_TYPES.*?return\s*{([^}]+)}/s);
    const returnTypesMatch = nodeCode.match(/RETURN_TYPES\s*=\s*\[([^\]]+)\]/);
    const functionMatch = nodeCode.match(/FUNCTION\s*=\s*["']([^"']+)["']/);

    // Extract methods with docstrings
    const methodMatches = nodeCode.match(/def\s+(\w+)\s*\([^)]*\):\s*"""([^"]+)"""/g) || [];
    const methods = methodMatches.map(match => {
      const nameMatch = match.match(/def\s+(\w+)/);
      const docMatch = match.match(/"""([^"]+)"""/);
      const signatureMatch = match.match(/def\s+(\w+\s*\([^)]*\))/);

      return {
        name: nameMatch?.[1] || 'unknown',
        signature: signatureMatch?.[1] || '',
        description: docMatch?.[1]?.trim() || 'No description available',
      };
    });

    // Extract dependencies
    const importMatches = nodeCode.match(/^(?:from|import)\s+([^\s]+)/gm) || [];
    const dependencies = importMatches.map(match => match.replace(/^(?:from|import)\s+/, '').split('.')[0]);

    // Determine complexity and features
    const lines = nodeCode.split('\n').length;
    const complexity = lines > 200 ? 'complex' : lines > 100 ? 'moderate' : 'simple';

    const features = this.extractFeatures(nodeCode);
    const useCases = this.generateUseCases(nodeName, features);

    return {
      name: nodeName,
      description: descriptionMatch?.[1]?.trim() || `${nodeName} ComfyUI node`,
      category: categoryMatch?.[1] || 'custom',
      inputTypes: this.parseInputTypes(inputTypesMatch?.[1] || ''),
      returnTypes: this.parseReturnTypes(returnTypesMatch?.[1] || ''),
      functionName: functionMatch?.[1] || 'process',
      methods,
      dependencies,
      complexity,
      features,
      useCases,
    };
  }

  private extractFeatures(nodeCode: string): string[] {
    const features: string[] = [];

    if (nodeCode.includes('torch')) features.push('PyTorch Integration');
    if (nodeCode.includes('cv2') || nodeCode.includes('opencv')) features.push('OpenCV Support');
    if (nodeCode.includes('numpy')) features.push('NumPy Arrays');
    if (nodeCode.includes('PIL') || nodeCode.includes('Image')) features.push('PIL Image Processing');
    if (nodeCode.includes('batch')) features.push('Batch Processing');
    if (nodeCode.includes('cuda') || nodeCode.includes('gpu')) features.push('GPU Acceleration');
    if (nodeCode.includes('async') || nodeCode.includes('await')) features.push('Async Processing');

    return features;
  }

  private generateUseCases(nodeName: string, features: string[]): string[] {
    const useCases: string[] = [];

    if (features.includes('PyTorch Integration')) {
      useCases.push('Deep learning model inference');
      useCases.push('Neural network processing');
    }

    if (features.includes('OpenCV Support')) {
      useCases.push('Computer vision tasks');
      useCases.push('Image preprocessing');
    }

    if (features.includes('Batch Processing')) {
      useCases.push('High-throughput processing');
      useCases.push('Automated workflows');
    }

    // Add generic use cases
    useCases.push('Creative image generation');
    useCases.push('Research and experimentation');

    return useCases;
  }

  private parseInputTypes(inputTypesStr: string): Record<string, any> {
    const inputTypes: Record<string, any> = {};

    // Simple parser - in real implementation would be more robust
    if (inputTypesStr.includes('IMAGE')) {
      inputTypes.image = { type: 'IMAGE', required: true, description: 'Input image tensor' };
    }
    if (inputTypesStr.includes('INT')) {
      inputTypes.steps = { type: 'INT', required: false, description: 'Number of processing steps' };
    }
    if (inputTypesStr.includes('FLOAT')) {
      inputTypes.weight = { type: 'FLOAT', required: false, description: 'Processing weight factor' };
    }
    if (inputTypesStr.includes('STRING')) {
      inputTypes.text = { type: 'STRING', required: false, description: 'Text input parameter' };
    }

    return inputTypes;
  }

  private parseReturnTypes(returnTypesStr: string): string[] {
    return returnTypesStr.split(',').map(type => type.trim().replace(/["']/g, ''));
  }

  private async generateReadme(nodeAnalysis: any, researchContent: any, options: any): Promise<string> {
    const template = this.templates.get('readme') || '';

    const researchContext = researchContent ? `
## Research Background

This node is based on research from: **${researchContent.title}**

${researchContent.keyFindings ? researchContent.keyFindings.slice(0, 3).map((finding: string) => `- ${finding}`).join('\n') : ''}
` : '';

    const features = nodeAnalysis.features.map((feature: string) => `- ${feature}`).join('\n');

    const usageExamples = `
### Basic Usage

1. Add the ${nodeAnalysis.name} node to your ComfyUI workflow
2. Connect an image input to the node
3. Configure the parameters as needed
4. Run the workflow to see results

### Parameters

${Object.entries(nodeAnalysis.inputTypes).map(([key, value]: [string, any]) =>
  `- **${key}** (${value.type}): ${value.description || 'Parameter description'}`
).join('\n')}
`;

    const apiReference = `
### Input Types
${Object.entries(nodeAnalysis.inputTypes).map(([key, value]: [string, any]) =>
  `- \`${key}\` (${value.type}): ${value.required ? 'Required' : 'Optional'}`
).join('\n')}

### Return Types
${nodeAnalysis.returnTypes.map((type: string, index: number) =>
  `- Output ${index + 1}: ${type}`
).join('\n')}
`;

    const examples = `
### Example Workflow

\`\`\`python
# Load the node
node = ${nodeAnalysis.name}()

# Process an image
result = node.${nodeAnalysis.functionName}(input_image)
\`\`\`

### Use Cases

${nodeAnalysis.useCases.map((useCase: string) => `- ${useCase}`).join('\n')}
`;

    return template
      .replace(/{{NODE_NAME}}/g, nodeAnalysis.name)
      .replace(/{{DESCRIPTION}}/g, nodeAnalysis.description)
      .replace(/{{RESEARCH_CONTEXT}}/g, researchContext)
      .replace(/{{FEATURES}}/g, features)
      .replace(/{{USAGE_EXAMPLES}}/g, usageExamples)
      .replace(/{{API_REFERENCE}}/g, apiReference)
      .replace(/{{EXAMPLES}}/g, examples)
      .replace(/{{TROUBLESHOOTING}}/g, this.generateTroubleshootingSection(nodeAnalysis))
      .replace(/{{CONTRIBUTING}}/g, 'Please see CONTRIBUTING.md for guidelines.')
      .replace(/{{LICENSE}}/g, 'MIT License - see LICENSE file for details.');
  }

  private async generateApiDocumentation(nodeAnalysis: any, options: any): Promise<string> {
    const template = this.templates.get('api') || '';

    const overview = `
The ${nodeAnalysis.name} node provides ${nodeAnalysis.description.toLowerCase()}.

**Category:** ${nodeAnalysis.category}
**Complexity:** ${nodeAnalysis.complexity}
`;

    const inputTypes = Object.entries(nodeAnalysis.inputTypes).map(([key, value]: [string, any]) => `
### ${key}

- **Type:** \`${value.type}\`
- **Required:** ${value.required ? 'Yes' : 'No'}
- **Description:** ${value.description || 'No description available'}
${value.default !== undefined ? `- **Default:** \`${value.default}\`` : ''}
${value.min !== undefined ? `- **Minimum:** \`${value.min}\`` : ''}
${value.max !== undefined ? `- **Maximum:** \`${value.max}\`` : ''}
`).join('\n');

    const returnTypes = nodeAnalysis.returnTypes.map((type: string, index: number) => `
### Output ${index + 1}

- **Type:** \`${type}\`
- **Description:** Processed ${type.toLowerCase()} output
`).join('\n');

    const methods = nodeAnalysis.methods.map((method: any) => `
### ${method.name}

\`\`\`python
${method.signature}
\`\`\`

${method.description}
`).join('\n');

    return template
      .replace(/{{NODE_NAME}}/g, nodeAnalysis.name)
      .replace(/{{OVERVIEW}}/g, overview)
      .replace(/{{INPUT_TYPES}}/g, inputTypes)
      .replace(/{{RETURN_TYPES}}/g, returnTypes)
      .replace(/{{METHODS}}/g, methods)
      .replace(/{{PARAMETERS}}/g, 'See Input Types section above.')
      .replace(/{{ERROR_HANDLING}}/g, 'The node includes comprehensive error handling for invalid inputs.')
      .replace(/{{API_EXAMPLES}}/g, await this.generateApiExampleSection(nodeAnalysis));
  }

  private async generateTutorial(nodeAnalysis: any, options: any): Promise<string> {
    const template = this.templates.get('tutorial') || '';

    const introduction = `
Welcome to the ${nodeAnalysis.name} tutorial! This guide will walk you through using this powerful ComfyUI node for ${nodeAnalysis.description.toLowerCase()}.

**What you'll learn:**
- How to install and set up the node
- Basic usage patterns
- Advanced configuration options
- Best practices and tips
`;

    const prerequisites = `
Before starting this tutorial, make sure you have:

- ComfyUI installed and running
- Basic familiarity with ComfyUI workflows
- Required dependencies: ${nodeAnalysis.dependencies.join(', ')}
`;

    const stepByStep = `
## Step 1: Installation

1. Download the ${nodeAnalysis.name} node
2. Copy to your ComfyUI custom_nodes directory
3. Install dependencies: \`pip install -r requirements.txt\`
4. Restart ComfyUI

## Step 2: Basic Setup

1. Open ComfyUI in your browser
2. Add the ${nodeAnalysis.name} node to your workflow
3. Connect the required inputs

## Step 3: Configuration

Configure the node parameters:

${Object.entries(nodeAnalysis.inputTypes).map(([key, value]: [string, any]) =>
  `- **${key}**: ${value.description || 'Configure as needed'}`
).join('\n')}

## Step 4: Run and Test

1. Connect your input image
2. Set the parameters
3. Queue the workflow
4. Review the results
`;

    return template
      .replace(/{{NODE_NAME}}/g, nodeAnalysis.name)
      .replace(/{{INTRODUCTION}}/g, introduction)
      .replace(/{{PREREQUISITES}}/g, prerequisites)
      .replace(/{{STEP_BY_STEP}}/g, stepByStep)
      .replace(/{{ADVANCED_USAGE}}/g, 'Advanced configuration options and optimization tips.')
      .replace(/{{BEST_PRACTICES}}/g, 'Best practices for optimal performance and results.')
      .replace(/{{COMMON_PATTERNS}}/g, 'Common usage patterns and workflow examples.')
      .replace(/{{NEXT_STEPS}}/g, 'Explore advanced features and integration with other nodes.');
  }

  private async generateExamples(nodeAnalysis: any, options: any): Promise<string> {
    return `# ${nodeAnalysis.name} Examples

## Basic Example

\`\`\`python
# Simple usage example
node = ${nodeAnalysis.name}()
result = node.${nodeAnalysis.functionName}(input_image)
\`\`\`

## Advanced Examples

### Batch Processing

\`\`\`python
# Process multiple images
images = [image1, image2, image3]
results = []
for img in images:
    result = node.${nodeAnalysis.functionName}(img)
    results.append(result)
\`\`\`

### Parameter Optimization

\`\`\`python
# Optimize parameters for best results
best_params = {
${Object.keys(nodeAnalysis.inputTypes).map(key => `    "${key}": optimal_value`).join(',\n')}
}
result = node.${nodeAnalysis.functionName}(input_image, **best_params)
\`\`\`

## Use Case Examples

${nodeAnalysis.useCases.map((useCase: string, index: number) => `
### ${index + 1}. ${useCase}

Description of how to use the node for ${useCase.toLowerCase()}.

\`\`\`python
# Example code for ${useCase.toLowerCase()}
# Implementation details here
\`\`\`
`).join('\n')}
`;
  }

  private async generateChangelog(nodeAnalysis: any): Promise<string> {
    return `# Changelog

All notable changes to ${nodeAnalysis.name} will be documented in this file.

## [1.0.0] - ${new Date().toISOString().split('T')[0]}

### Added
- Initial release of ${nodeAnalysis.name}
- ${nodeAnalysis.features.map((feature: string) => `Support for ${feature}`).join('\n- ')}
- Comprehensive documentation and examples
- Full ComfyUI integration

### Features
${nodeAnalysis.features.map((feature: string) => `- ${feature}`).join('\n')}

### Dependencies
${nodeAnalysis.dependencies.map((dep: string) => `- ${dep}`).join('\n')}
`;
  }

  private async generateAssets(nodeAnalysis: any): Promise<Record<string, string>> {
    // In a real implementation, this would generate actual diagrams and images
    return {
      'workflow-diagram.svg': '<!-- SVG workflow diagram -->',
      'architecture-diagram.png': '<!-- Architecture diagram -->',
      'example-output.jpg': '<!-- Example output image -->',
    };
  }

  private generateJsonSchema(nodeAnalysis: any): Record<string, any> {
    return {
      type: 'object',
      properties: {
        name: { type: 'string', const: nodeAnalysis.name },
        category: { type: 'string', const: nodeAnalysis.category },
        inputs: {
          type: 'object',
          properties: Object.fromEntries(
            Object.entries(nodeAnalysis.inputTypes).map(([key, value]: [string, any]) => [
              key,
              {
                type: value.type.toLowerCase(),
                required: value.required,
                description: value.description,
              },
            ])
          ),
        },
        outputs: {
          type: 'array',
          items: {
            type: 'string',
            enum: nodeAnalysis.returnTypes,
          },
        },
      },
    };
  }

  private async generateApiExamples(nodeAnalysis: any): Promise<Record<string, string>> {
    return {
      basic: `node.${nodeAnalysis.functionName}(input_image)`,
      advanced: `node.${nodeAnalysis.functionName}(input_image, **parameters)`,
      batch: `[node.${nodeAnalysis.functionName}(img) for img in images]`,
    };
  }

  private async generateApiExampleSection(nodeAnalysis: any): Promise<string> {
    return `
### Basic Usage

\`\`\`python
result = node.${nodeAnalysis.functionName}(input_image)
\`\`\`

### With Parameters

\`\`\`python
result = node.${nodeAnalysis.functionName}(
    input_image,
${Object.keys(nodeAnalysis.inputTypes).map(key => `    ${key}=value`).join(',\n')}
)
\`\`\`
`;
  }

  private async generateQuickStart(nodeAnalysis: any, userLevel: string): Promise<string> {
    return `# Quick Start Guide

## Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Basic Usage
1. Add ${nodeAnalysis.name} to your workflow
2. Connect inputs
3. Run workflow

${userLevel === 'beginner' ? '## Beginner Tips\n- Start with default parameters\n- Check the examples folder' : ''}
`;
  }

  private async generateTroubleshooting(nodeAnalysis: any): Promise<string> {
    return this.generateTroubleshootingSection(nodeAnalysis);
  }

  private generateTroubleshootingSection(nodeAnalysis: any): string {
    return `
## Common Issues

### Installation Problems
- Ensure all dependencies are installed: \`pip install -r requirements.txt\`
- Check Python version compatibility
- Verify ComfyUI is running properly

### Runtime Errors
- Check input image format and dimensions
- Verify parameter values are within valid ranges
- Ensure sufficient memory for processing

### Performance Issues
- ${nodeAnalysis.features.includes('GPU Acceleration') ? 'Enable GPU acceleration if available' : 'Consider using smaller input images'}
- ${nodeAnalysis.features.includes('Batch Processing') ? 'Use batch processing for multiple images' : 'Process images individually for better memory management'}

### Getting Help
- Check the documentation and examples
- Review the API reference
- Submit issues on the project repository
`;
  }

  private async generateFaq(nodeAnalysis: any): Promise<string> {
    return `# Frequently Asked Questions

## Q: What image formats are supported?
A: The node supports standard image formats compatible with ComfyUI.

## Q: Can I use this node with other ComfyUI nodes?
A: Yes, ${nodeAnalysis.name} is fully compatible with the ComfyUI ecosystem.

## Q: What are the system requirements?
A: ${nodeAnalysis.dependencies.includes('torch') ? 'PyTorch-compatible system with sufficient GPU memory.' : 'Standard Python environment with required dependencies.'}

## Q: How do I optimize performance?
A: ${nodeAnalysis.features.includes('GPU Acceleration') ? 'Use GPU acceleration and appropriate batch sizes.' : 'Optimize input image sizes and parameter settings.'}
`;
  }
}

// MCP Tool definitions
export const documentationAgentGenerateTool: Tool = {
  name: 'documentation_agent_generate',
  description: 'Generate comprehensive documentation for ComfyUI nodes',
  inputSchema: {
    type: 'object',
    properties: {
      nodeCode: {
        type: 'string',
        description: 'The ComfyUI node code to document',
      },
      nodeName: {
        type: 'string',
        description: 'Name of the ComfyUI node',
      },
      researchContent: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          content: { type: 'string' },
          keyFindings: { type: 'array', items: { type: 'string' } },
          technologies: { type: 'array', items: { type: 'string' } },
        },
      },
      documentationTypes: {
        type: 'array',
        items: {
          type: 'string',
          enum: ['readme', 'api', 'tutorial', 'examples', 'changelog'],
        },
        default: ['readme', 'api'],
      },
      options: {
        type: 'object',
        properties: {
          format: { type: 'string', enum: ['markdown', 'rst', 'html'], default: 'markdown' },
          includeExamples: { type: 'boolean', default: true },
          includeImages: { type: 'boolean', default: false },
          generateToc: { type: 'boolean', default: true },
          style: { type: 'string', enum: ['technical', 'user-friendly', 'comprehensive'], default: 'user-friendly' },
        },
      },
    },
    required: ['nodeCode', 'nodeName'],
  },
};

export const documentationAgentApiTool: Tool = {
  name: 'documentation_agent_api',
  description: 'Generate API reference documentation',
  inputSchema: {
    type: 'object',
    properties: {
      nodeCode: {
        type: 'string',
        description: 'The ComfyUI node code to document',
      },
      nodeName: {
        type: 'string',
        description: 'Name of the ComfyUI node',
      },
    },
    required: ['nodeCode', 'nodeName'],
  },
};

export const documentationAgentGuideTool: Tool = {
  name: 'documentation_agent_guide',
  description: 'Generate user tutorials and guides',
  inputSchema: {
    type: 'object',
    properties: {
      nodeCode: {
        type: 'string',
        description: 'The ComfyUI node code to document',
      },
      nodeName: {
        type: 'string',
        description: 'Name of the ComfyUI node',
      },
      userLevel: {
        type: 'string',
        enum: ['beginner', 'intermediate', 'advanced'],
        default: 'beginner',
        description: 'Target user experience level',
      },
    },
    required: ['nodeCode', 'nodeName'],
  },
};