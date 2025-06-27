import { Resource } from '@modelcontextprotocol/sdk/types.js';
import { promises as fs } from 'fs';
import { join } from 'path';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('Resources');
const config = getConfig();

/**
 * Get framework configuration and settings
 */
export async function getFrameworkConfig(): Promise<string> {
  try {
    const frameworkConfig = {
      version: '1.0.0',
      name: 'ComfyUI Framework MCP Server',
      description: 'Professional MCP server for AI-powered ComfyUI node generation',
      
      agents: {
        available: config.framework.supportedAgents,
        default: config.framework.defaultAgents,
        configurations: {
          research: {
            timeout: config.agents.research.timeout,
            supportedSources: config.agents.research.supportedSources,
            cacheEnabled: config.agents.research.cacheResults,
          },
          coding: {
            timeout: config.agents.coding.timeout,
            outputFormat: config.agents.coding.outputFormat,
            codeStyle: config.agents.coding.codeStyle,
          },
          testing: {
            timeout: config.agents.testing.timeout,
            framework: config.agents.testing.framework,
            coverage: config.agents.testing.coverage,
          },
          documentation: {
            timeout: config.agents.documentation.timeout,
            format: config.agents.documentation.format,
            includeExamples: config.agents.documentation.includeExamples,
          },
          devops: {
            timeout: config.agents.devops.timeout,
            ciProvider: config.agents.devops.ciProvider,
            packageManager: config.agents.devops.packageManager,
          },
        },
      },
      
      workflows: {
        maxConcurrentExecutions: config.framework.maxConcurrentExecutions,
        defaultTimeout: config.framework.executionTimeout,
        templates: [
          {
            id: 'quick-prototype',
            name: 'Quick Prototype',
            agents: ['research', 'coding'],
            description: 'Fast prototyping for development',
          },
          {
            id: 'production-ready',
            name: 'Production Ready',
            agents: ['research', 'coding', 'testing', 'documentation', 'devops'],
            description: 'Complete production workflow',
          },
          {
            id: 'research-only',
            name: 'Research Analysis',
            agents: ['research'],
            description: 'Research and analysis only',
          },
        ],
      },
      
      scraping: {
        supportedMethods: config.scraping.supportedMethods,
        rateLimitDelay: config.scraping.rateLimitDelay,
        timeout: config.scraping.timeout,
        userAgent: config.scraping.userAgent,
      },
      
      llm: {
        provider: config.llm.provider,
        model: config.llm.model,
        maxTokens: config.llm.maxTokens,
        temperature: config.llm.temperature,
      },
      
      storage: {
        type: config.storage.type,
        path: config.storage.path,
        cleanup: config.storage.cleanup,
      },
    };

    return JSON.stringify(frameworkConfig, null, 2);
  } catch (error) {
    logger.error('Failed to get framework config:', error);
    return JSON.stringify({ error: 'Failed to load configuration' });
  }
}

/**
 * Get available agent templates and their capabilities
 */
export async function getAgentTemplates(): Promise<string> {
  try {
    const templates = {
      research: {
        name: 'Research Agent',
        description: 'Extract and analyze research content from papers, repositories, and web sources',
        capabilities: [
          'ArXiv paper extraction',
          'GitHub repository analysis',
          'Web content scraping',
          'PDF document parsing',
          'Content analysis and summarization',
          'Technology detection',
          'Citation extraction',
        ],
        supportedSources: ['arxiv', 'github', 'web', 'pdf', 'file'],
        outputFormats: ['json', 'markdown'],
        configuration: {
          timeout: '5 minutes',
          maxContentLength: '1MB',
          cacheResults: true,
          extractCode: true,
        },
      },
      
      coding: {
        name: 'Coding Agent',
        description: 'Generate high-quality ComfyUI node code with proper patterns and optimization',
        capabilities: [
          'ComfyUI node generation',
          'Code optimization',
          'Pattern recognition',
          'Dependency management',
          'Code validation',
          'Multiple output formats',
        ],
        supportedFrameworks: ['comfyui', 'pytorch', 'tensorflow'],
        codeStyles: ['standard', 'google', 'airbnb'],
        configuration: {
          timeout: '10 minutes',
          includeTests: true,
          optimizationLevel: 'standard',
          outputFormat: 'python',
        },
      },
      
      testing: {
        name: 'Testing Agent',
        description: 'Create comprehensive test suites and validation frameworks',
        capabilities: [
          'Unit test generation',
          'Integration test creation',
          'Performance testing',
          'Edge case testing',
          'Test execution',
          'Coverage analysis',
        ],
        supportedFrameworks: ['pytest', 'unittest', 'jest'],
        testTypes: ['unit', 'integration', 'performance', 'edge_cases'],
        configuration: {
          timeout: '5 minutes',
          coverage: true,
          minCoverage: 80,
          includePerformanceTests: false,
        },
      },
      
      documentation: {
        name: 'Documentation Agent',
        description: 'Generate professional documentation, API docs, and usage examples',
        capabilities: [
          'README generation',
          'API documentation',
          'Tutorial creation',
          'Example generation',
          'Changelog creation',
          'Multi-format output',
        ],
        supportedFormats: ['markdown', 'rst', 'html'],
        documentationTypes: ['readme', 'api', 'tutorial', 'examples', 'changelog'],
        configuration: {
          timeout: '3 minutes',
          includeExamples: true,
          generateToc: true,
          style: 'user-friendly',
        },
      },
      
      devops: {
        name: 'DevOps Agent',
        description: 'Handle packaging, deployment, CI/CD setup, and infrastructure management',
        capabilities: [
          'Package creation',
          'Docker containerization',
          'CI/CD pipeline setup',
          'Deployment automation',
          'Infrastructure as code',
          'Monitoring setup',
        ],
        supportedPlatforms: ['docker', 'kubernetes', 'github', 'gitlab'],
        packageManagers: ['pip', 'npm', 'conda'],
        configuration: {
          timeout: '2 minutes',
          includeDockerfile: true,
          includeCI: true,
          ciProvider: 'github',
        },
      },
    };

    return JSON.stringify(templates, null, 2);
  } catch (error) {
    logger.error('Failed to get agent templates:', error);
    return JSON.stringify({ error: 'Failed to load agent templates' });
  }
}

/**
 * Get artifacts from a specific execution
 */
export async function getExecutionArtifacts(executionId: string): Promise<string> {
  try {
    const artifactsPath = join('./artifacts', executionId);
    
    try {
      await fs.access(artifactsPath);
    } catch {
      return JSON.stringify({ error: 'Execution artifacts not found' });
    }

    const artifacts: Record<string, any> = {};
    const files = await fs.readdir(artifactsPath);
    
    for (const file of files) {
      if (file.endsWith('.json')) {
        try {
          const filePath = join(artifactsPath, file);
          const content = await fs.readFile(filePath, 'utf-8');
          const agentName = file.replace('.json', '');
          artifacts[agentName] = JSON.parse(content);
        } catch (error) {
          logger.warn(`Failed to read artifact file ${file}:`, error);
          artifacts[file] = { error: 'Failed to read file' };
        }
      }
    }

    return JSON.stringify({
      executionId,
      artifactsFound: Object.keys(artifacts).length,
      artifacts,
    }, null, 2);

  } catch (error) {
    logger.error(`Failed to get execution artifacts for ${executionId}:`, error);
    return JSON.stringify({ error: 'Failed to load execution artifacts' });
  }
}

/**
 * Get execution logs for debugging and monitoring
 */
export async function getExecutionLogs(executionId: string): Promise<string> {
  try {
    const logsPath = join('./logs', `${executionId}.log`);
    
    try {
      const logContent = await fs.readFile(logsPath, 'utf-8');
      const lines = logContent.split('\n').filter(line => line.trim().length > 0);
      
      return JSON.stringify({
        executionId,
        logLines: lines.length,
        logs: lines.slice(-100), // Last 100 lines
        fullLogPath: logsPath,
      }, null, 2);
      
    } catch {
      // If specific execution log doesn't exist, return general logs
      const generalLogPath = config.logging.file;
      if (generalLogPath) {
        try {
          const logContent = await fs.readFile(generalLogPath, 'utf-8');
          const lines = logContent.split('\n')
            .filter(line => line.includes(executionId))
            .slice(-50); // Last 50 relevant lines
          
          return JSON.stringify({
            executionId,
            logLines: lines.length,
            logs: lines,
            source: 'general_log',
          }, null, 2);
        } catch {
          return JSON.stringify({ error: 'No logs found for execution' });
        }
      }
      
      return JSON.stringify({ error: 'Execution logs not found' });
    }

  } catch (error) {
    logger.error(`Failed to get execution logs for ${executionId}:`, error);
    return JSON.stringify({ error: 'Failed to load execution logs' });
  }
}

/**
 * Get example ComfyUI node structures and patterns
 */
export async function getNodeExamples(): Promise<string> {
  try {
    const examples = {
      basic_node: {
        description: 'Basic ComfyUI node structure',
        structure: {
          required_methods: ['INPUT_TYPES', 'RETURN_TYPES', 'FUNCTION'],
          optional_attributes: ['CATEGORY', 'OUTPUT_NODE', 'DEPRECATED'],
        },
        example: `class ExampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "custom"
    
    def process(self, image, strength, mask=None):
        # Process the image
        result = image * strength
        return (result,)`,
        patterns: {
          input_validation: 'Always validate inputs in the process method',
          error_handling: 'Use try-catch blocks for robust error handling',
          tensor_operations: 'Ensure proper tensor format (BHWC for images)',
          return_format: 'Return tuple matching RETURN_TYPES',
        },
      },
      
      advanced_node: {
        description: 'Advanced ComfyUI node with multiple features',
        features: [
          'Multiple input types',
          'Optional parameters',
          'Custom validation',
          'Progress tracking',
          'Memory optimization',
        ],
        example: `class AdvancedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["fast", "quality", "balanced"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "advanced_process"
    CATEGORY = "advanced"
    OUTPUT_NODE = False
    
    def advanced_process(self, image, mode, steps, seed=-1, batch_size=1):
        # Advanced processing with validation
        if seed == -1:
            seed = random.randint(0, 2**32-1)
        
        results = []
        masks = []
        
        for i in range(batch_size):
            # Process each item in batch
            result, mask = self._process_single(image, mode, steps, seed + i)
            results.append(result)
            masks.append(mask)
        
        return (torch.cat(results, dim=0), torch.cat(masks, dim=0))`,
      },
      
      integration_patterns: {
        description: 'Common integration patterns for ComfyUI nodes',
        patterns: {
          node_registration: {
            description: 'How to register nodes with ComfyUI',
            example: `NODE_CLASS_MAPPINGS = {
    "ExampleNode": ExampleNode,
    "AdvancedNode": AdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExampleNode": "Example Node",
    "AdvancedNode": "Advanced Processing Node",
}`,
          },
          
          custom_types: {
            description: 'Defining custom input/output types',
            example: `# Custom type definition
CUSTOM_TYPES = {
    "CUSTOM_DATA": "CUSTOM_DATA",
}

# Usage in node
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "custom_input": ("CUSTOM_DATA",),
        }
    }`,
          },
          
          batch_processing: {
            description: 'Handling batch operations efficiently',
            example: `def process_batch(self, images, **kwargs):
    batch_size = images.shape[0]
    results = []
    
    for i in range(batch_size):
        result = self.process_single(images[i], **kwargs)
        results.append(result)
    
    return torch.stack(results, dim=0)`,
          },
        },
      },
      
      best_practices: {
        description: 'Best practices for ComfyUI node development',
        practices: [
          'Use descriptive parameter names and tooltips',
          'Implement proper input validation',
          'Handle edge cases gracefully',
          'Optimize for memory usage',
          'Provide clear error messages',
          'Support batch processing when possible',
          'Use appropriate data types',
          'Document your node thoroughly',
        ],
        performance_tips: [
          'Use torch.no_grad() for inference',
          'Minimize tensor copying',
          'Use in-place operations when safe',
          'Implement proper cleanup',
          'Consider GPU memory limitations',
        ],
      },
    };

    return JSON.stringify(examples, null, 2);
  } catch (error) {
    logger.error('Failed to get node examples:', error);
    return JSON.stringify({ error: 'Failed to load node examples' });
  }
}

/**
 * Get workflow templates and configurations
 */
export async function getWorkflowTemplates(): Promise<string> {
  try {
    const templatesPath = './workflow-templates';
    const templates: Record<string, any> = {};
    
    try {
      const files = await fs.readdir(templatesPath);
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          try {
            const filePath = join(templatesPath, file);
            const content = await fs.readFile(filePath, 'utf-8');
            const template = JSON.parse(content);
            templates[template.id] = template;
          } catch (error) {
            logger.warn(`Failed to read template file ${file}:`, error);
          }
        }
      }
    } catch {
      // If templates directory doesn't exist, return default templates
      templates['quick-prototype'] = {
        id: 'quick-prototype',
        name: 'Quick Prototype',
        description: 'Fast prototyping workflow for development',
        agents: ['research', 'coding'],
        estimatedTime: '30 minutes',
        complexity: 'low',
      };
      
      templates['production-ready'] = {
        id: 'production-ready',
        name: 'Production Ready',
        description: 'Complete workflow for production-ready nodes',
        agents: ['research', 'coding', 'testing', 'documentation', 'devops'],
        estimatedTime: '2 hours',
        complexity: 'high',
      };
      
      templates['research-only'] = {
        id: 'research-only',
        name: 'Research Analysis',
        description: 'Research and analysis only',
        agents: ['research'],
        estimatedTime: '15 minutes',
        complexity: 'low',
      };
    }

    return JSON.stringify({
      available: Object.keys(templates).length,
      templates,
    }, null, 2);

  } catch (error) {
    logger.error('Failed to get workflow templates:', error);
    return JSON.stringify({ error: 'Failed to load workflow templates' });
  }
}

// Export resource definitions for MCP server
export const resources: Resource[] = [
  {
    uri: 'config://framework',
    name: 'Framework Configuration',
    description: 'Get the current framework configuration and settings',
    mimeType: 'application/json',
  },
  {
    uri: 'agents://templates',
    name: 'Agent Templates',
    description: 'Get available agent templates and their capabilities',
    mimeType: 'application/json',
  },
  {
    uri: 'artifacts://{execution_id}',
    name: 'Execution Artifacts',
    description: 'Get artifacts from a specific execution',
    mimeType: 'application/json',
  },
  {
    uri: 'logs://{execution_id}',
    name: 'Execution Logs',
    description: 'Get execution logs for debugging and monitoring',
    mimeType: 'application/json',
  },
  {
    uri: 'examples://nodes',
    name: 'Node Examples',
    description: 'Get example ComfyUI node structures and patterns',
    mimeType: 'application/json',
  },
  {
    uri: 'workflows://templates',
    name: 'Workflow Templates',
    description: 'Get available workflow templates and configurations',
    mimeType: 'application/json',
  },
];

// Resource handler function
export async function handleResourceRequest(uri: string): Promise<string> {
  const url = new URL(uri);

  switch (url.protocol) {
    case 'config:':
      if (url.pathname === '//framework') {
        return await getFrameworkConfig();
      }
      break;

    case 'agents:':
      if (url.pathname === '//templates') {
        return await getAgentTemplates();
      }
      break;

    case 'artifacts:':
      const executionId = url.pathname.replace('//', '');
      return await getExecutionArtifacts(executionId);

    case 'logs:':
      const logExecutionId = url.pathname.replace('//', '');
      return await getExecutionLogs(logExecutionId);

    case 'examples:':
      if (url.pathname === '//nodes') {
        return await getNodeExamples();
      }
      break;

    case 'workflows:':
      if (url.pathname === '//templates') {
        return await getWorkflowTemplates();
      }
      break;
  }

  throw new Error(`Unknown resource: ${uri}`);
}
