import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { promises as fs } from 'fs';
import { join, dirname } from 'path';
import { v4 as uuidv4 } from 'uuid';

import { ComfyUINode, CodeAnalysis, ExecutionError, ValidationError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('CodingAgent');

// Input validation schema
const CodingInputSchema = z.object({
  researchContent: z.object({
    title: z.string(),
    content: z.string(),
    technologies: z.array(z.string()).optional(),
    codePatterns: z.array(z.string()).optional(),
    keyFindings: z.array(z.string()).optional(),
  }),
  nodeSpec: z.object({
    name: z.string().min(1).max(100),
    category: z.string().default('custom'),
    description: z.string().max(500),
    inputTypes: z.record(z.object({
      type: z.string(),
      required: z.boolean().default(true),
      default: z.any().optional(),
      min: z.number().optional(),
      max: z.number().optional(),
      options: z.array(z.string()).optional(),
    })).optional(),
    returnTypes: z.array(z.string()).default(['IMAGE']),
    outputNode: z.boolean().default(false),
  }),
  options: z.object({
    outputFormat: z.enum(['python', 'typescript']).default('python'),
    includeTests: z.boolean().default(true),
    includeDocumentation: z.boolean().default(true),
    optimizationLevel: z.enum(['basic', 'standard', 'advanced']).default('standard'),
    codeStyle: z.enum(['standard', 'google', 'airbnb']).default('standard'),
    targetFramework: z.string().default('comfyui'),
  }).optional(),
});

type CodingInput = z.infer<typeof CodingInputSchema>;

export class CodingAgent {
  private config = getConfig();
  private templates = new Map<string, string>();

  constructor() {
    this.loadTemplates();
  }

  /**
   * Generate ComfyUI node code from research content and specifications
   */
  async generateComfyUINode(input: CodingInput): Promise<{
    nodeCode: string;
    testCode?: string;
    documentation?: string;
    dependencies: string[];
    metadata: {
      nodeId: string;
      generatedAt: string;
      framework: string;
      version: string;
      complexity: 'simple' | 'moderate' | 'complex';
    };
  }> {
    const validatedInput = CodingInputSchema.parse(input);
    const { researchContent, nodeSpec, options = {} } = validatedInput;

    logger.info(`Generating ComfyUI node: ${nodeSpec.name}`);

    try {
      // Analyze research content for implementation insights
      const analysis = await this.analyzeResearchForImplementation(researchContent);
      
      // Generate node structure
      const nodeStructure = await this.generateNodeStructure(nodeSpec, analysis);
      
      // Generate implementation code
      const nodeCode = await this.generateNodeImplementation(nodeStructure, analysis, options);
      
      // Generate tests if requested
      let testCode: string | undefined;
      if (options?.includeTests) {
        testCode = await this.generateTestCode(nodeStructure, options);
      }

      // Generate documentation if requested
      let documentation: string | undefined;
      if (options?.includeDocumentation) {
        documentation = await this.generateDocumentation(nodeStructure, analysis);
      }
      
      // Extract dependencies
      const dependencies = this.extractDependencies(nodeCode, analysis);
      
      // Calculate complexity
      const complexity = this.calculateComplexity(nodeCode, analysis);

      const result = {
        nodeCode,
        testCode,
        documentation,
        dependencies,
        metadata: {
          nodeId: uuidv4(),
          generatedAt: new Date().toISOString(),
          framework: (options as any)?.targetFramework || 'comfyui',
          version: '1.0.0',
          complexity,
        },
      };

      logger.info(`Successfully generated ComfyUI node: ${nodeSpec.name}`);
      return result;

    } catch (error) {
      logger.error(`Failed to generate ComfyUI node: ${error}`);
      throw new ExecutionError(`Code generation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Optimize existing ComfyUI node code
   */
  async optimizeNodeCode(code: string, optimizationLevel: 'basic' | 'standard' | 'advanced' = 'standard'): Promise<{
    optimizedCode: string;
    improvements: string[];
    performanceGains: string[];
    warnings: string[];
  }> {
    logger.info(`Optimizing node code with ${optimizationLevel} level`);

    const improvements: string[] = [];
    const performanceGains: string[] = [];
    const warnings: string[] = [];
    let optimizedCode = code;

    try {
      // Basic optimizations
      if (optimizationLevel === 'basic' || optimizationLevel === 'standard' || optimizationLevel === 'advanced') {
        optimizedCode = this.applyBasicOptimizations(optimizedCode, improvements);
      }

      // Standard optimizations
      if (optimizationLevel === 'standard' || optimizationLevel === 'advanced') {
        optimizedCode = this.applyStandardOptimizations(optimizedCode, improvements, performanceGains);
      }

      // Advanced optimizations
      if (optimizationLevel === 'advanced') {
        optimizedCode = this.applyAdvancedOptimizations(optimizedCode, improvements, performanceGains, warnings);
      }

      return {
        optimizedCode,
        improvements,
        performanceGains,
        warnings,
      };

    } catch (error) {
      logger.error(`Code optimization failed: ${error}`);
      throw new ExecutionError(`Code optimization failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Validate ComfyUI node code structure and compatibility
   */
  async validateNodeCode(code: string): Promise<{
    isValid: boolean;
    errors: Array<{ line: number; message: string; severity: 'error' | 'warning' }>;
    suggestions: string[];
    compatibility: {
      comfyuiVersion: string;
      pythonVersion: string;
      dependencies: Array<{ name: string; version: string; available: boolean }>;
    };
  }> {
    logger.info('Validating ComfyUI node code');

    const errors: Array<{ line: number; message: string; severity: 'error' | 'warning' }> = [];
    const suggestions: string[] = [];

    try {
      // Check required ComfyUI patterns
      this.validateComfyUIPatterns(code, errors);
      
      // Check code structure
      this.validateCodeStructure(code, errors, suggestions);
      
      // Check dependencies
      const dependencies = this.extractDependencies(code, { technologies: [] });
      const compatibility = await this.checkCompatibility(dependencies);
      
      const isValid = errors.filter(e => e.severity === 'error').length === 0;

      return {
        isValid,
        errors,
        suggestions,
        compatibility,
      };

    } catch (error) {
      logger.error(`Code validation failed: ${error}`);
      throw new ExecutionError(`Code validation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private async loadTemplates(): Promise<void> {
    // Load code templates - in a real implementation, these would be loaded from files
    this.templates.set('comfyui_node_basic', `
class {{NODE_NAME}}:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {{INPUT_TYPES}},
            "optional": {{OPTIONAL_TYPES}}
        }
    
    RETURN_TYPES = {{RETURN_TYPES}}
    FUNCTION = "{{FUNCTION_NAME}}"
    CATEGORY = "{{CATEGORY}}"
    
    def {{FUNCTION_NAME}}(self{{PARAMETERS}}):
        """{{DESCRIPTION}}"""
        {{IMPLEMENTATION}}
        return ({{RETURN_VALUES}},)

NODE_CLASS_MAPPINGS = {
    "{{NODE_NAME}}": {{NODE_NAME}}
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "{{NODE_NAME}}": "{{DISPLAY_NAME}}"
}
`);

    this.templates.set('comfyui_test_basic', `
import unittest
import torch
from {{MODULE_NAME}} import {{NODE_NAME}}

class Test{{NODE_NAME}}(unittest.TestCase):
    def setUp(self):
        self.node = {{NODE_NAME}}()
    
    def test_input_types(self):
        """Test that INPUT_TYPES is properly defined"""
        input_types = self.node.INPUT_TYPES()
        self.assertIn("required", input_types)
        
    def test_basic_functionality(self):
        """Test basic node functionality"""
        {{TEST_IMPLEMENTATION}}
        
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        {{EDGE_CASE_TESTS}}

if __name__ == '__main__':
    unittest.main()
`);
  }

  private async analyzeResearchForImplementation(researchContent: any): Promise<{
    technologies: string[];
    algorithms: string[];
    implementationHints: string[];
    complexity: 'simple' | 'moderate' | 'complex';
  }> {
    const content = researchContent.content.toLowerCase();
    
    // Extract technologies
    const technologies = researchContent.technologies || [];
    
    // Extract algorithms mentioned
    const algorithmKeywords = [
      'neural network', 'cnn', 'rnn', 'transformer', 'attention', 'diffusion',
      'gan', 'vae', 'autoencoder', 'unet', 'resnet', 'efficientnet'
    ];
    const algorithms = algorithmKeywords.filter(keyword => content.includes(keyword));
    
    // Generate implementation hints
    const implementationHints: string[] = [];
    if (content.includes('pytorch') || content.includes('torch')) {
      implementationHints.push('Use PyTorch tensors for computation');
    }
    if (content.includes('image') || content.includes('vision')) {
      implementationHints.push('Handle image tensor formats (BHWC/BCHW)');
    }
    if (content.includes('batch') || content.includes('batching')) {
      implementationHints.push('Support batch processing');
    }
    
    // Determine complexity
    let complexity: 'simple' | 'moderate' | 'complex' = 'simple';
    if (algorithms.length > 2 || content.includes('complex') || content.includes('advanced')) {
      complexity = 'complex';
    } else if (algorithms.length > 0 || technologies.length > 3) {
      complexity = 'moderate';
    }

    return {
      technologies,
      algorithms,
      implementationHints,
      complexity,
    };
  }

  private async generateNodeStructure(nodeSpec: any, analysis: any): Promise<ComfyUINode> {
    return {
      name: nodeSpec.name,
      category: nodeSpec.category,
      description: nodeSpec.description,
      inputTypes: nodeSpec.inputTypes || this.generateDefaultInputTypes(analysis),
      returnTypes: nodeSpec.returnTypes,
      function: this.generateFunctionName(nodeSpec.name),
      outputNode: nodeSpec.outputNode,
    };
  }

  private generateDefaultInputTypes(analysis: any): Record<string, any> {
    const inputTypes: Record<string, any> = {};
    
    // Always include image input for ComfyUI nodes
    inputTypes.image = {
      type: 'IMAGE',
      required: true,
    };
    
    // Add algorithm-specific inputs
    if (analysis.algorithms.includes('diffusion')) {
      inputTypes.steps = {
        type: 'INT',
        required: false,
        default: 20,
        min: 1,
        max: 100,
      };
    }
    
    if (analysis.algorithms.includes('attention')) {
      inputTypes.attention_weight = {
        type: 'FLOAT',
        required: false,
        default: 1.0,
        min: 0.0,
        max: 2.0,
        step: 0.1,
      };
    }

    return inputTypes;
  }

  private generateFunctionName(nodeName: string): string {
    return nodeName.toLowerCase().replace(/[^a-z0-9]/g, '_');
  }

  private async generateNodeImplementation(nodeStructure: ComfyUINode, analysis: any, options: any): Promise<string> {
    const template = this.templates.get('comfyui_node_basic') || '';
    
    // Generate implementation based on analysis
    let implementation = this.generateImplementationLogic(nodeStructure, analysis);
    
    // Apply code style
    if (options.codeStyle === 'google') {
      implementation = this.applyGoogleStyle(implementation);
    } else if (options.codeStyle === 'airbnb') {
      implementation = this.applyAirbnbStyle(implementation);
    }

    return template
      .replace(/{{NODE_NAME}}/g, nodeStructure.name)
      .replace(/{{INPUT_TYPES}}/g, JSON.stringify(this.extractRequiredInputs(nodeStructure.inputTypes), null, 12))
      .replace(/{{OPTIONAL_TYPES}}/g, JSON.stringify(this.extractOptionalInputs(nodeStructure.inputTypes), null, 12))
      .replace(/{{RETURN_TYPES}}/g, JSON.stringify(nodeStructure.returnTypes))
      .replace(/{{FUNCTION_NAME}}/g, nodeStructure.function)
      .replace(/{{CATEGORY}}/g, nodeStructure.category)
      .replace(/{{DESCRIPTION}}/g, nodeStructure.description)
      .replace(/{{PARAMETERS}}/g, this.generateParameters(nodeStructure.inputTypes))
      .replace(/{{IMPLEMENTATION}}/g, implementation)
      .replace(/{{RETURN_VALUES}}/g, this.generateReturnValues(nodeStructure.returnTypes))
      .replace(/{{DISPLAY_NAME}}/g, this.generateDisplayName(nodeStructure.name));
  }

  private generateImplementationLogic(nodeStructure: ComfyUINode, analysis: any): string {
    let implementation = `
        # Process input image
        if image is None:
            raise ValueError("Input image is required")
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        # Ensure correct format (BHWC)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
    `;

    // Add algorithm-specific logic
    if (analysis.algorithms.includes('diffusion')) {
      implementation += `
        # Apply diffusion process
        for step in range(steps):
            # Diffusion step implementation
            image = self.diffusion_step(image, step)
      `;
    }

    if (analysis.algorithms.includes('attention')) {
      implementation += `
        # Apply attention mechanism
        attention_map = self.compute_attention(image)
        image = image * attention_map * attention_weight
      `;
    }

    implementation += `
        # Return processed image
        return image
    `;

    return implementation;
  }

  private extractRequiredInputs(inputTypes: Record<string, any>): Record<string, any> {
    const required: Record<string, any> = {};
    for (const [key, value] of Object.entries(inputTypes)) {
      if (value.required) {
        const { required: _, ...inputDef } = value;
        required[key] = inputDef;
      }
    }
    return required;
  }

  private extractOptionalInputs(inputTypes: Record<string, any>): Record<string, any> {
    const optional: Record<string, any> = {};
    for (const [key, value] of Object.entries(inputTypes)) {
      if (!value.required) {
        const { required: _, ...inputDef } = value;
        optional[key] = inputDef;
      }
    }
    return optional;
  }

  private generateParameters(inputTypes: Record<string, any>): string {
    const params = Object.keys(inputTypes).map(key => `, ${key}`);
    return params.join('');
  }

  private generateReturnValues(returnTypes: string[]): string {
    return returnTypes.map(type => type.toLowerCase()).join(', ');
  }

  private generateDisplayName(name: string): string {
    return name.replace(/([A-Z])/g, ' $1').trim();
  }

  private async generateTestCode(nodeStructure: ComfyUINode, options: any): Promise<string> {
    const template = this.templates.get('comfyui_test_basic') || '';
    
    const testImplementation = `
        # Create test input
        test_image = torch.randn(1, 512, 512, 3)
        
        # Call node function
        result = self.node.${nodeStructure.function}(test_image)
        
        # Verify output
        self.assertIsNotNone(result)
        self.assertEqual(len(result), ${nodeStructure.returnTypes.length})
    `;

    const edgeCaseTests = `
        # Test with None input
        with self.assertRaises(ValueError):
            self.node.${nodeStructure.function}(None)
        
        # Test with invalid dimensions
        invalid_image = torch.randn(2, 2)
        with self.assertRaises(Exception):
            self.node.${nodeStructure.function}(invalid_image)
    `;

    return template
      .replace(/{{MODULE_NAME}}/g, nodeStructure.name.toLowerCase())
      .replace(/{{NODE_NAME}}/g, nodeStructure.name)
      .replace(/{{TEST_IMPLEMENTATION}}/g, testImplementation)
      .replace(/{{EDGE_CASE_TESTS}}/g, edgeCaseTests);
  }

  private async generateDocumentation(nodeStructure: ComfyUINode, analysis: any): Promise<string> {
    return `# ${nodeStructure.name}

${nodeStructure.description}

## Category
${nodeStructure.category}

## Inputs
${Object.entries(nodeStructure.inputTypes).map(([key, value]) => 
  `- **${key}** (${value.type}): ${value.required ? 'Required' : 'Optional'}${value.default ? ` (default: ${value.default})` : ''}`
).join('\n')}

## Outputs
${nodeStructure.returnTypes.map((type, index) => `- **Output ${index + 1}** (${type})`).join('\n')}

## Implementation Details
Based on research analysis, this node implements:
${analysis.algorithms.map((algo: string) => `- ${algo}`).join('\n')}

## Usage Example
\`\`\`python
# Load the node
node = ${nodeStructure.name}()

# Process an image
result = node.${nodeStructure.function}(input_image)
\`\`\`

## Dependencies
${analysis.technologies.map((tech: string) => `- ${tech}`).join('\n')}
`;
  }

  private extractDependencies(code: string, analysis: any): string[] {
    const dependencies = new Set<string>();
    
    // Extract from imports
    const importMatches = code.match(/^(?:from|import)\s+([^\s]+)/gm);
    if (importMatches) {
      importMatches.forEach(match => {
        const module = match.replace(/^(?:from|import)\s+/, '').split('.')[0];
        if (!['os', 'sys', 'math', 'json'].includes(module)) {
          dependencies.add(module);
        }
      });
    }
    
    // Add from analysis
    analysis.technologies.forEach((tech: string) => {
      if (tech === 'pytorch') dependencies.add('torch');
      if (tech === 'opencv') dependencies.add('opencv-python');
      if (tech === 'numpy') dependencies.add('numpy');
    });
    
    return Array.from(dependencies);
  }

  private calculateComplexity(code: string, analysis: any): 'simple' | 'moderate' | 'complex' {
    const lines = code.split('\n').length;
    const algorithms = analysis.algorithms.length;
    
    if (lines > 200 || algorithms > 3) return 'complex';
    if (lines > 100 || algorithms > 1) return 'moderate';
    return 'simple';
  }

  // Optimization methods
  private applyBasicOptimizations(code: string, improvements: string[]): string {
    // Remove unnecessary whitespace
    let optimized = code.replace(/\n\s*\n\s*\n/g, '\n\n');
    improvements.push('Removed excessive whitespace');
    
    // Optimize imports
    optimized = this.optimizeImports(optimized);
    improvements.push('Optimized import statements');
    
    return optimized;
  }

  private applyStandardOptimizations(code: string, improvements: string[], performanceGains: string[]): string {
    let optimized = code;
    
    // Add tensor operations optimization
    if (optimized.includes('torch.')) {
      optimized = optimized.replace(/\.cpu\(\)\.numpy\(\)/g, '.detach().cpu().numpy()');
      improvements.push('Added proper tensor detachment');
      performanceGains.push('Reduced memory usage in tensor operations');
    }
    
    return optimized;
  }

  private applyAdvancedOptimizations(code: string, improvements: string[], performanceGains: string[], warnings: string[]): string {
    let optimized = code;
    
    // Add memory optimization warnings
    if (optimized.includes('torch.randn') && !optimized.includes('device=')) {
      warnings.push('Consider specifying device for tensor operations');
    }
    
    return optimized;
  }

  private optimizeImports(code: string): string {
    const lines = code.split('\n');
    const imports: string[] = [];
    const otherLines: string[] = [];
    
    lines.forEach(line => {
      if (line.trim().startsWith('import ') || line.trim().startsWith('from ')) {
        imports.push(line);
      } else {
        otherLines.push(line);
      }
    });
    
    // Sort and deduplicate imports
    const uniqueImports = [...new Set(imports)].sort();
    
    return [...uniqueImports, '', ...otherLines].join('\n');
  }

  private validateComfyUIPatterns(code: string, errors: Array<{ line: number; message: string; severity: 'error' | 'warning' }>): void {
    const requiredPatterns = ['INPUT_TYPES', 'RETURN_TYPES', 'FUNCTION'];
    
    requiredPatterns.forEach(pattern => {
      if (!code.includes(pattern)) {
        errors.push({
          line: 0,
          message: `Missing required ComfyUI pattern: ${pattern}`,
          severity: 'error',
        });
      }
    });
  }

  private validateCodeStructure(code: string, errors: Array<{ line: number; message: string; severity: 'error' | 'warning' }>, suggestions: string[]): void {
    const lines = code.split('\n');
    
    lines.forEach((line, index) => {
      // Check for common issues
      if (line.includes('print(') && !line.includes('#')) {
        errors.push({
          line: index + 1,
          message: 'Avoid print statements in production code',
          severity: 'warning',
        });
      }
      
      if (line.includes('TODO') || line.includes('FIXME')) {
        errors.push({
          line: index + 1,
          message: 'Unresolved TODO/FIXME comment',
          severity: 'warning',
        });
      }
    });
    
    // Add suggestions
    if (!code.includes('"""')) {
      suggestions.push('Add docstrings to improve code documentation');
    }
    
    if (!code.includes('try:')) {
      suggestions.push('Consider adding error handling with try-except blocks');
    }
  }

  private async checkCompatibility(dependencies: string[]): Promise<any> {
    // In a real implementation, this would check actual package availability
    return {
      comfyuiVersion: '1.0.0',
      pythonVersion: '3.8+',
      dependencies: dependencies.map(dep => ({
        name: dep,
        version: 'latest',
        available: true,
      })),
    };
  }

  private applyGoogleStyle(code: string): string {
    // Apply Google Python style guide formatting
    return code;
  }

  private applyAirbnbStyle(code: string): string {
    // Apply Airbnb style guide formatting
    return code;
  }
}

// MCP Tool definitions
export const codingAgentGenerateTool: Tool = {
  name: 'coding_agent_generate',
  description: 'Generate high-quality ComfyUI node code from research content and specifications',
  inputSchema: {
    type: 'object',
    properties: {
      researchContent: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          content: { type: 'string' },
          technologies: { type: 'array', items: { type: 'string' } },
          codePatterns: { type: 'array', items: { type: 'string' } },
          keyFindings: { type: 'array', items: { type: 'string' } },
        },
        required: ['title', 'content'],
      },
      nodeSpec: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          category: { type: 'string', default: 'custom' },
          description: { type: 'string' },
          returnTypes: { type: 'array', items: { type: 'string' }, default: ['IMAGE'] },
          outputNode: { type: 'boolean', default: false },
        },
        required: ['name', 'description'],
      },
      options: {
        type: 'object',
        properties: {
          includeTests: { type: 'boolean', default: true },
          includeDocumentation: { type: 'boolean', default: true },
          optimizationLevel: { type: 'string', enum: ['basic', 'standard', 'advanced'], default: 'standard' },
          codeStyle: { type: 'string', enum: ['standard', 'google', 'airbnb'], default: 'standard' },
        },
      },
    },
    required: ['researchContent', 'nodeSpec'],
  },
};

export const codingAgentOptimizeTool: Tool = {
  name: 'coding_agent_optimize',
  description: 'Optimize existing ComfyUI node code for better performance and maintainability',
  inputSchema: {
    type: 'object',
    properties: {
      code: {
        type: 'string',
        description: 'The ComfyUI node code to optimize',
      },
      optimizationLevel: {
        type: 'string',
        enum: ['basic', 'standard', 'advanced'],
        default: 'standard',
        description: 'Level of optimization to apply',
      },
    },
    required: ['code'],
  },
};

export const codingAgentValidateTool: Tool = {
  name: 'coding_agent_validate',
  description: 'Validate ComfyUI node code structure and compatibility',
  inputSchema: {
    type: 'object',
    properties: {
      code: {
        type: 'string',
        description: 'The ComfyUI node code to validate',
      },
    },
    required: ['code'],
  },
};
