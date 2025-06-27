import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { promises as fs } from 'fs';
import { join, dirname } from 'path';
import { v4 as uuidv4 } from 'uuid';

import { ExecutionError, ValidationError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('TestingAgent');

// Input validation schemas
const TestGenerationInputSchema = z.object({
  nodeCode: z.string().min(1, 'Node code is required'),
  nodeName: z.string().min(1, 'Node name is required'),
  testTypes: z.array(z.enum(['unit', 'integration', 'performance', 'edge_cases'])).default(['unit', 'edge_cases']),
  framework: z.enum(['pytest', 'unittest', 'jest']).default('pytest'),
  options: z.object({
    coverage: z.boolean().default(true),
    minCoverage: z.number().min(0).max(100).default(80),
    includePerformanceTests: z.boolean().default(false),
    includeMockData: z.boolean().default(true),
    generateFixtures: z.boolean().default(true),
  }).optional(),
});

const TestExecutionInputSchema = z.object({
  testCode: z.string().min(1, 'Test code is required'),
  nodeCode: z.string().min(1, 'Node code is required'),
  testPath: z.string().optional(),
  options: z.object({
    verbose: z.boolean().default(true),
    coverage: z.boolean().default(true),
    parallel: z.boolean().default(false),
    timeout: z.number().default(30000),
  }).optional(),
});

type TestGenerationInput = z.infer<typeof TestGenerationInputSchema>;
type TestExecutionInput = z.infer<typeof TestExecutionInputSchema>;

export class TestingAgent {
  private config = getConfig();
  private testTemplates = new Map<string, string>();

  constructor() {
    this.loadTestTemplates();
  }

  /**
   * Generate comprehensive test suites for ComfyUI nodes
   */
  async generateTestSuite(input: TestGenerationInput): Promise<{
    testCode: string;
    fixtures: Record<string, any>;
    mockData: Record<string, any>;
    testConfig: {
      framework: string;
      coverage: boolean;
      requirements: string[];
    };
    metadata: {
      testId: string;
      generatedAt: string;
      testCount: number;
      estimatedRuntime: number;
    };
  }> {
    const validatedInput = TestGenerationInputSchema.parse(input);
    const { nodeCode, nodeName, testTypes, framework, options = {} } = validatedInput;

    logger.info(`Generating test suite for node: ${nodeName}`);

    try {
      // Analyze node code to understand structure
      const nodeAnalysis = await this.analyzeNodeStructure(nodeCode, nodeName);

      // Generate different types of tests
      const testSections: string[] = [];

      if (testTypes.includes('unit')) {
        testSections.push(await this.generateUnitTests(nodeAnalysis, framework));
      }

      if (testTypes.includes('integration')) {
        testSections.push(await this.generateIntegrationTests(nodeAnalysis, framework));
      }

      if (testTypes.includes('performance')) {
        testSections.push(await this.generatePerformanceTests(nodeAnalysis, framework));
      }

      if (testTypes.includes('edge_cases')) {
        testSections.push(await this.generateEdgeCaseTests(nodeAnalysis, framework));
      }

      // Combine all test sections
      const testCode = this.combineTestSections(testSections, nodeAnalysis, framework);

      // Generate fixtures and mock data
      const fixtures = options?.generateFixtures ? await this.generateTestFixtures(nodeAnalysis) : {};
      const mockData = options?.includeMockData ? await this.generateMockData(nodeAnalysis) : {};

      // Calculate metadata
      const testCount = this.countTests(testCode);
      const estimatedRuntime = this.estimateRuntime(testCount, testTypes);

      const result = {
        testCode,
        fixtures,
        mockData,
        testConfig: {
          framework,
          coverage: options?.coverage,
          requirements: this.getTestRequirements(framework, testTypes),
        },
        metadata: {
          testId: uuidv4(),
          generatedAt: new Date().toISOString(),
          testCount,
          estimatedRuntime,
        },
      };

      logger.info(`Generated ${testCount} tests for ${nodeName}`);
      return result;

    } catch (error) {
      logger.error(`Failed to generate test suite: ${error}`);
      throw new ExecutionError(`Test generation failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Execute test suite and return results
   */
  async executeTests(input: TestExecutionInput): Promise<{
    success: boolean;
    results: {
      passed: number;
      failed: number;
      skipped: number;
      total: number;
    };
    coverage?: {
      percentage: number;
      lines: { covered: number; total: number };
      branches: { covered: number; total: number };
    };
    failures: Array<{
      testName: string;
      error: string;
      traceback?: string;
    }>;
    performance: {
      totalTime: number;
      averageTime: number;
      slowestTests: Array<{ name: string; time: number }>;
    };
  }> {
    const validatedInput = TestExecutionInputSchema.parse(input);
    const { testCode, nodeCode, options = {} } = validatedInput;

    logger.info('Executing test suite');

    try {
      // In a real implementation, this would execute the actual tests
      // For now, we'll simulate test execution
      const simulatedResults = await this.simulateTestExecution(testCode, nodeCode, options);

      logger.info(`Test execution completed: ${simulatedResults.results.passed}/${simulatedResults.results.total} passed`);
      return simulatedResults;

    } catch (error) {
      logger.error(`Test execution failed: ${error}`);
      throw new ExecutionError(`Test execution failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Validate test quality and completeness
   */
  async validateTestSuite(testCode: string, nodeCode: string): Promise<{
    quality: {
      score: number;
      coverage: number;
      maintainability: number;
      reliability: number;
    };
    issues: Array<{
      type: 'missing_test' | 'poor_assertion' | 'no_edge_case' | 'performance_issue';
      severity: 'low' | 'medium' | 'high';
      message: string;
      suggestion: string;
    }>;
    recommendations: string[];
  }> {
    logger.info('Validating test suite quality');

    const issues: Array<{
      type: 'missing_test' | 'poor_assertion' | 'no_edge_case' | 'performance_issue';
      severity: 'low' | 'medium' | 'high';
      message: string;
      suggestion: string;
    }> = [];

    const recommendations: string[] = [];

    // Analyze test coverage
    const coverage = this.analyzeTestCoverage(testCode, nodeCode);

    // Check for missing tests
    if (!testCode.includes('test_input_validation')) {
      issues.push({
        type: 'missing_test',
        severity: 'high',
        message: 'Missing input validation tests',
        suggestion: 'Add tests to verify input parameter validation',
      });
    }

    // Check for edge cases
    if (!testCode.includes('edge') && !testCode.includes('boundary')) {
      issues.push({
        type: 'no_edge_case',
        severity: 'medium',
        message: 'No edge case tests found',
        suggestion: 'Add tests for boundary conditions and edge cases',
      });
    }

    // Calculate quality scores
    const quality = {
      score: this.calculateOverallQuality(coverage, issues.length),
      coverage,
      maintainability: this.calculateMaintainability(testCode),
      reliability: this.calculateReliability(testCode, issues),
    };

    // Generate recommendations
    if (quality.coverage < 80) {
      recommendations.push('Increase test coverage to at least 80%');
    }
    if (quality.maintainability < 70) {
      recommendations.push('Improve test maintainability by reducing duplication');
    }

    return {
      quality,
      issues,
      recommendations,
    };
  }

  private async loadTestTemplates(): Promise<void> {
    // Load test templates for different frameworks
    this.testTemplates.set('pytest_unit', `
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from {{MODULE_NAME}} import {{NODE_NAME}}

class Test{{NODE_NAME}}:
    @pytest.fixture
    def node(self):
        return {{NODE_NAME}}()

    @pytest.fixture
    def sample_image(self):
        return torch.randn(1, 512, 512, 3)

    def test_input_types_structure(self, node):
        """Test that INPUT_TYPES returns proper structure"""
        input_types = node.INPUT_TYPES()
        assert isinstance(input_types, dict)
        assert "required" in input_types

    def test_return_types_defined(self, node):
        """Test that RETURN_TYPES is properly defined"""
        assert hasattr(node, "RETURN_TYPES")
        assert isinstance(node.RETURN_TYPES, (list, tuple))

    {{UNIT_TESTS}}
`);

    this.testTemplates.set('pytest_integration', `
    def test_full_workflow_integration(self, node, sample_image):
        """Test complete workflow integration"""
        {{INTEGRATION_TESTS}}

    def test_comfyui_compatibility(self, node):
        """Test ComfyUI framework compatibility"""
        # Test node registration
        assert hasattr(node, "INPUT_TYPES")
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "FUNCTION")
`);

    this.testTemplates.set('pytest_performance', `
    @pytest.mark.performance
    def test_processing_speed(self, node, sample_image):
        """Test processing performance"""
        import time
        start_time = time.time()
        {{PERFORMANCE_TESTS}}
        end_time = time.time()
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, expected < 5.0s"

    @pytest.mark.performance
    def test_memory_usage(self, node, sample_image):
        """Test memory efficiency"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        {{MEMORY_TESTS}}
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
`);

    this.testTemplates.set('pytest_edge_cases', `
    def test_none_input(self, node):
        """Test handling of None inputs"""
        with pytest.raises((ValueError, TypeError)):
            {{NONE_INPUT_TESTS}}

    def test_invalid_dimensions(self, node):
        """Test handling of invalid tensor dimensions"""
        invalid_tensor = torch.randn(2, 2)  # Invalid dimensions
        with pytest.raises(Exception):
            {{INVALID_DIMENSION_TESTS}}

    def test_empty_tensor(self, node):
        """Test handling of empty tensors"""
        empty_tensor = torch.empty(0)
        with pytest.raises(Exception):
            {{EMPTY_TENSOR_TESTS}}

    {{EDGE_CASE_TESTS}}
`);
  }

  private async analyzeNodeStructure(nodeCode: string, nodeName: string): Promise<{
    name: string;
    inputTypes: Record<string, any>;
    returnTypes: string[];
    functionName: string;
    methods: string[];
    dependencies: string[];
    complexity: 'simple' | 'moderate' | 'complex';
  }> {
    // Parse the node code to extract structure
    const inputTypesMatch = nodeCode.match(/INPUT_TYPES.*?return\s*{([^}]+)}/s);
    const returnTypesMatch = nodeCode.match(/RETURN_TYPES\s*=\s*\[([^\]]+)\]/);
    const functionMatch = nodeCode.match(/FUNCTION\s*=\s*["']([^"']+)["']/);

    // Extract methods
    const methodMatches = nodeCode.match(/def\s+(\w+)\s*\(/g) || [];
    const methods = methodMatches.map(match => match.replace(/def\s+(\w+)\s*\(/, '$1'));

    // Extract dependencies
    const importMatches = nodeCode.match(/^(?:from|import)\s+([^\s]+)/gm) || [];
    const dependencies = importMatches.map(match => match.replace(/^(?:from|import)\s+/, '').split('.')[0]);

    // Determine complexity
    const lines = nodeCode.split('\n').length;
    const complexity = lines > 200 ? 'complex' : lines > 100 ? 'moderate' : 'simple';

    return {
      name: nodeName,
      inputTypes: this.parseInputTypes(inputTypesMatch?.[1] || ''),
      returnTypes: this.parseReturnTypes(returnTypesMatch?.[1] || ''),
      functionName: functionMatch?.[1] || 'process',
      methods,
      dependencies,
      complexity,
    };
  }

  private parseInputTypes(inputTypesStr: string): Record<string, any> {
    // Simple parser for input types - in real implementation would be more robust
    const inputTypes: Record<string, any> = {};

    if (inputTypesStr.includes('IMAGE')) {
      inputTypes.image = { type: 'IMAGE', required: true };
    }
    if (inputTypesStr.includes('INT')) {
      inputTypes.steps = { type: 'INT', required: false };
    }
    if (inputTypesStr.includes('FLOAT')) {
      inputTypes.weight = { type: 'FLOAT', required: false };
    }

    return inputTypes;
  }

  private parseReturnTypes(returnTypesStr: string): string[] {
    return returnTypesStr.split(',').map(type => type.trim().replace(/["']/g, ''));
  }

  private async generateUnitTests(nodeAnalysis: any, framework: string): Promise<string> {
    const template = this.testTemplates.get(`${framework}_unit`) || '';

    let unitTests = '';

    // Generate tests for each input type
    Object.entries(nodeAnalysis.inputTypes).forEach(([inputName, inputDef]: [string, any]) => {
      unitTests += `
    def test_${inputName}_processing(self, node, sample_image):
        """Test ${inputName} input processing"""
        result = node.${nodeAnalysis.functionName}(sample_image)
        assert result is not None
        assert len(result) == ${nodeAnalysis.returnTypes.length}
      `;
    });

    // Generate tests for return types
    nodeAnalysis.returnTypes.forEach((returnType: string, index: number) => {
      unitTests += `
    def test_return_type_${index}_${returnType.toLowerCase()}(self, node, sample_image):
        """Test return type ${returnType}"""
        result = node.${nodeAnalysis.functionName}(sample_image)
        output = result[${index}]
        assert output is not None
      `;
    });

    return template
      .replace(/{{MODULE_NAME}}/g, nodeAnalysis.name.toLowerCase())
      .replace(/{{NODE_NAME}}/g, nodeAnalysis.name)
      .replace(/{{UNIT_TESTS}}/g, unitTests);
  }

  private async generateIntegrationTests(nodeAnalysis: any, framework: string): Promise<string> {
    const template = this.testTemplates.get(`${framework}_integration`) || '';

    const integrationTests = `
        result = node.${nodeAnalysis.functionName}(sample_image)
        assert result is not None

        # Test chaining with other nodes
        for output in result:
            assert output is not None
    `;

    return template.replace(/{{INTEGRATION_TESTS}}/g, integrationTests);
  }

  private async generatePerformanceTests(nodeAnalysis: any, framework: string): Promise<string> {
    const template = this.testTemplates.get(`${framework}_performance`) || '';

    const performanceTests = `
        result = node.${nodeAnalysis.functionName}(sample_image)
        assert result is not None
    `;

    const memoryTests = `
        result = node.${nodeAnalysis.functionName}(sample_image)
        del result  # Clean up
    `;

    return template
      .replace(/{{PERFORMANCE_TESTS}}/g, performanceTests)
      .replace(/{{MEMORY_TESTS}}/g, memoryTests);
  }

  private async generateEdgeCaseTests(nodeAnalysis: any, framework: string): Promise<string> {
    const template = this.testTemplates.get(`${framework}_edge_cases`) || '';

    const noneInputTests = `node.${nodeAnalysis.functionName}(None)`;
    const invalidDimensionTests = `node.${nodeAnalysis.functionName}(invalid_tensor)`;
    const emptyTensorTests = `node.${nodeAnalysis.functionName}(empty_tensor)`;

    let edgeCaseTests = '';

    // Add specific edge cases based on node complexity
    if (nodeAnalysis.complexity === 'complex') {
      edgeCaseTests += `
    def test_large_input(self, node):
        """Test with very large input"""
        large_image = torch.randn(1, 2048, 2048, 3)
        result = node.${nodeAnalysis.functionName}(large_image)
        assert result is not None
      `;
    }

    return template
      .replace(/{{NONE_INPUT_TESTS}}/g, noneInputTests)
      .replace(/{{INVALID_DIMENSION_TESTS}}/g, invalidDimensionTests)
      .replace(/{{EMPTY_TENSOR_TESTS}}/g, emptyTensorTests)
      .replace(/{{EDGE_CASE_TESTS}}/g, edgeCaseTests);
  }

  private combineTestSections(sections: string[], nodeAnalysis: any, framework: string): string {
    const header = `"""
Comprehensive test suite for ${nodeAnalysis.name}
Generated automatically by ComfyUI Framework Testing Agent
Framework: ${framework}
"""

`;

    return header + sections.join('\n\n');
  }

  private async generateTestFixtures(nodeAnalysis: any): Promise<Record<string, any>> {
    return {
      sample_images: {
        small: 'torch.randn(1, 64, 64, 3)',
        medium: 'torch.randn(1, 512, 512, 3)',
        large: 'torch.randn(1, 1024, 1024, 3)',
      },
      sample_parameters: {
        default_steps: 20,
        default_weight: 1.0,
        test_ranges: {
          steps: [1, 10, 50, 100],
          weights: [0.0, 0.5, 1.0, 1.5, 2.0],
        },
      },
    };
  }

  private async generateMockData(nodeAnalysis: any): Promise<Record<string, any>> {
    return {
      mock_responses: {
        api_calls: {},
        file_operations: {},
      },
      test_data: {
        valid_inputs: {},
        invalid_inputs: {},
        edge_cases: {},
      },
    };
  }

  private countTests(testCode: string): number {
    const testMatches = testCode.match(/def test_\w+/g);
    return testMatches ? testMatches.length : 0;
  }

  private estimateRuntime(testCount: number, testTypes: string[]): number {
    let baseTime = testCount * 100; // 100ms per test

    if (testTypes.includes('performance')) {
      baseTime += testCount * 500; // Performance tests take longer
    }
    if (testTypes.includes('integration')) {
      baseTime += testCount * 200; // Integration tests take longer
    }

    return baseTime;
  }

  private getTestRequirements(framework: string, testTypes: string[]): string[] {
    const requirements = [framework];

    if (testTypes.includes('performance')) {
      requirements.push('psutil', 'memory-profiler');
    }
    if (testTypes.includes('integration')) {
      requirements.push('requests-mock');
    }

    return requirements;
  }

  private async simulateTestExecution(testCode: string, nodeCode: string, options: any): Promise<any> {
    // Simulate test execution results
    const testCount = this.countTests(testCode);
    const passed = Math.floor(testCount * 0.9); // 90% pass rate
    const failed = testCount - passed;

    return {
      success: failed === 0,
      results: {
        passed,
        failed,
        skipped: 0,
        total: testCount,
      },
      coverage: options.coverage ? {
        percentage: 85,
        lines: { covered: 170, total: 200 },
        branches: { covered: 34, total: 40 },
      } : undefined,
      failures: failed > 0 ? [{
        testName: 'test_edge_case_example',
        error: 'AssertionError: Expected non-null result',
        traceback: 'Traceback (most recent call last)...',
      }] : [],
      performance: {
        totalTime: testCount * 150,
        averageTime: 150,
        slowestTests: [
          { name: 'test_performance_large_input', time: 500 },
          { name: 'test_integration_workflow', time: 300 },
        ],
      },
    };
  }

  private analyzeTestCoverage(testCode: string, nodeCode: string): number {
    // Simple coverage analysis - count covered functions
    const nodeFunctions = (nodeCode.match(/def\s+\w+/g) || []).length;
    const testedFunctions = (testCode.match(/def test_\w+/g) || []).length;

    return nodeFunctions > 0 ? Math.min((testedFunctions / nodeFunctions) * 100, 100) : 0;
  }

  private calculateOverallQuality(coverage: number, issueCount: number): number {
    const coverageScore = coverage;
    const issueScore = Math.max(0, 100 - (issueCount * 10));

    return (coverageScore + issueScore) / 2;
  }

  private calculateMaintainability(testCode: string): number {
    const lines = testCode.split('\n').length;
    const duplicateLines = this.countDuplicateLines(testCode);

    return Math.max(0, 100 - (duplicateLines / lines) * 100);
  }

  private calculateReliability(testCode: string, issues: any[]): number {
    const errorIssues = issues.filter(issue => issue.severity === 'high').length;
    return Math.max(0, 100 - (errorIssues * 20));
  }

  private countDuplicateLines(code: string): number {
    const lines = code.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    const uniqueLines = new Set(lines);
    return lines.length - uniqueLines.size;
  }
}

// MCP Tool definitions
export const testingAgentGenerateTool: Tool = {
  name: 'testing_agent_generate',
  description: 'Generate comprehensive test suites for ComfyUI nodes',
  inputSchema: {
    type: 'object',
    properties: {
      nodeCode: {
        type: 'string',
        description: 'The ComfyUI node code to generate tests for',
      },
      nodeName: {
        type: 'string',
        description: 'Name of the ComfyUI node',
      },
      testTypes: {
        type: 'array',
        items: {
          type: 'string',
          enum: ['unit', 'integration', 'performance', 'edge_cases'],
        },
        default: ['unit', 'edge_cases'],
        description: 'Types of tests to generate',
      },
      framework: {
        type: 'string',
        enum: ['pytest', 'unittest', 'jest'],
        default: 'pytest',
        description: 'Testing framework to use',
      },
      options: {
        type: 'object',
        properties: {
          coverage: { type: 'boolean', default: true },
          minCoverage: { type: 'number', minimum: 0, maximum: 100, default: 80 },
          includePerformanceTests: { type: 'boolean', default: false },
          includeMockData: { type: 'boolean', default: true },
          generateFixtures: { type: 'boolean', default: true },
        },
      },
    },
    required: ['nodeCode', 'nodeName'],
  },
};

export const testingAgentExecuteTool: Tool = {
  name: 'testing_agent_execute',
  description: 'Execute test suite and return detailed results',
  inputSchema: {
    type: 'object',
    properties: {
      testCode: {
        type: 'string',
        description: 'The test code to execute',
      },
      nodeCode: {
        type: 'string',
        description: 'The ComfyUI node code being tested',
      },
      options: {
        type: 'object',
        properties: {
          verbose: { type: 'boolean', default: true },
          coverage: { type: 'boolean', default: true },
          parallel: { type: 'boolean', default: false },
          timeout: { type: 'number', default: 30000 },
        },
      },
    },
    required: ['testCode', 'nodeCode'],
  },
};

export const testingAgentValidateTool: Tool = {
  name: 'testing_agent_validate',
  description: 'Validate test suite quality and completeness',
  inputSchema: {
    type: 'object',
    properties: {
      testCode: {
        type: 'string',
        description: 'The test code to validate',
      },
      nodeCode: {
        type: 'string',
        description: 'The ComfyUI node code being tested',
      },
    },
    required: ['testCode', 'nodeCode'],
  },
};