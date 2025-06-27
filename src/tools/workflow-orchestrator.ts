import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';
import { join } from 'path';

import { Workflow, ExecutionContext, ExecutionError, ValidationError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';
import { ResearchAgent } from './research-agent';
import { CodingAgent } from './coding-agent';
import { TestingAgent } from './testing-agent';
import { DocumentationAgent } from './documentation-agent';
import { DevOpsAgent } from './devops-agent';

const logger = createLogger('WorkflowOrchestrator');

// Input validation schemas
const WorkflowExecutionInputSchema = z.object({
  workflowId: z.string().optional(),
  workflowConfig: z.object({
    name: z.string().min(1, 'Workflow name is required'),
    agents: z.array(z.enum(['research', 'coding', 'testing', 'documentation', 'devops'])).min(1),
    parallel: z.boolean().default(false),
    timeout: z.number().default(3600000), // 1 hour
    retryPolicy: z.object({
      maxRetries: z.number().min(0).max(5).default(3),
      backoffStrategy: z.enum(['linear', 'exponential', 'fixed']).default('exponential'),
      baseDelay: z.number().default(1000),
      maxDelay: z.number().default(30000),
    }).optional(),
  }),
  input: z.object({
    source: z.string().min(1, 'Input source is required'),
    outputDirectory: z.string().default('./output'),
    focusAreas: z.array(z.string()).optional(),
    qualityLevel: z.enum(['draft', 'development', 'production']).default('production'),
  }),
  options: z.object({
    saveIntermediateResults: z.boolean().default(true),
    continueOnError: z.boolean().default(false),
    notifyOnCompletion: z.boolean().default(false),
    generateReport: z.boolean().default(true),
  }).optional(),
});

const WorkflowTemplateInputSchema = z.object({
  name: z.string().min(1, 'Template name is required'),
  description: z.string().max(500),
  agents: z.array(z.string()).min(1),
  configuration: z.record(z.unknown()).default({}),
  tags: z.array(z.string()).default([]),
});

type WorkflowExecutionInput = z.infer<typeof WorkflowExecutionInputSchema>;
type WorkflowTemplateInput = z.infer<typeof WorkflowTemplateInputSchema>;

export class WorkflowOrchestrator {
  private config = getConfig();
  private activeExecutions = new Map<string, ExecutionContext>();
  private agents = new Map<string, any>();
  private workflows = new Map<string, Workflow>();

  constructor() {
    this.initializeAgents();
    this.loadWorkflowTemplates();
  }

  /**
   * Execute a complete workflow with multiple agents
   */
  async executeWorkflow(input: WorkflowExecutionInput): Promise<{
    executionId: string;
    status: 'completed' | 'failed' | 'partial';
    results: Record<string, any>;
    metrics: {
      totalTime: number;
      agentTimes: Record<string, number>;
      successRate: number;
      errors: string[];
    };
    artifacts: Record<string, any>;
  }> {
    const validatedInput = WorkflowExecutionInputSchema.parse(input);
    const { workflowConfig, input: workflowInput, options = {} } = validatedInput;

    const executionId = uuidv4();
    logger.info(`Starting workflow execution: ${executionId}`);

    try {
      // Create execution context
      const context: ExecutionContext = {
        id: executionId,
        inputSource: workflowInput.source,
        inputType: this.detectInputType(workflowInput.source),
        outputDirectory: workflowInput.outputDirectory,
        focusAreas: workflowInput.focusAreas,
        qualityLevel: workflowInput.qualityLevel,
        agentsCompleted: [],
        artifacts: {},
        metrics: {},
        startTime: new Date().toISOString(),
        status: 'running',
      };

      this.activeExecutions.set(executionId, context);

      // Execute agents
      const results: Record<string, any> = {};
      const agentTimes: Record<string, number> = {};
      const errors: string[] = [];
      const startTime = Date.now();

      if (workflowConfig.parallel) {
        // Execute agents in parallel
        const agentPromises = workflowConfig.agents.map(async (agentName) => {
          const agentStartTime = Date.now();
          try {
            const result = await this.executeAgent(agentName, context, options);
            agentTimes[agentName] = Date.now() - agentStartTime;
            return { agentName, result, success: true };
          } catch (error) {
            agentTimes[agentName] = Date.now() - agentStartTime;
            const errorMessage = error instanceof Error ? error.message : String(error);
            errors.push(`${agentName}: ${errorMessage}`);
            return { agentName, error: errorMessage, success: false };
          }
        });

        const agentResults = await Promise.all(agentPromises);
        
        agentResults.forEach(({ agentName, result, success }) => {
          if (success) {
            results[agentName] = result;
            context.agentsCompleted.push(agentName);
          }
        });

      } else {
        // Execute agents sequentially
        for (const agentName of workflowConfig.agents) {
          const agentStartTime = Date.now();
          
          try {
            const result = await this.executeAgent(agentName, context, options);
            results[agentName] = result;
            context.agentsCompleted.push(agentName);
            agentTimes[agentName] = Date.now() - agentStartTime;

            // Update context with results
            context.artifacts[agentName] = result;

          } catch (error) {
            agentTimes[agentName] = Date.now() - agentStartTime;
            const errorMessage = error instanceof Error ? error.message : String(error);
            errors.push(`${agentName}: ${errorMessage}`);
            
            if (!options?.continueOnError) {
              break;
            }
          }
        }
      }

      // Calculate metrics
      const totalTime = Date.now() - startTime;
      const successRate = context.agentsCompleted.length / workflowConfig.agents.length;
      const status: 'running' | 'completed' | 'failed' | 'cancelled' = errors.length === 0 ? 'completed' : 'failed';

      // Update context
      context.status = status;
      context.endTime = new Date().toISOString();

      // Generate report if requested
      if (options?.generateReport) {
        await this.generateExecutionReport(context, results, agentTimes, errors);
      }

      // Save artifacts if requested
      if (options?.saveIntermediateResults) {
        await this.saveArtifacts(executionId, context.artifacts);
      }

      logger.info(`Workflow execution completed: ${executionId} (${status})`);

      return {
        executionId,
        status,
        results,
        metrics: {
          totalTime,
          agentTimes,
          successRate,
          errors,
        },
        artifacts: context.artifacts,
      };

    } catch (error) {
      logger.error(`Workflow execution failed: ${error}`);
      throw new ExecutionError(`Workflow execution failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  /**
   * Create and save a workflow template
   */
  async createWorkflowTemplate(input: WorkflowTemplateInput): Promise<Workflow> {
    const validatedInput = WorkflowTemplateInputSchema.parse(input);
    const { name, description, agents, configuration, tags } = validatedInput;

    logger.info(`Creating workflow template: ${name}`);

    const workflow: Workflow = {
      id: uuidv4(),
      name,
      description,
      agents,
      configuration: {
        parallel: false,
        timeout: 3600000,
        retryPolicy: {
          maxRetries: 3,
          backoffStrategy: 'exponential',
          baseDelay: 1000,
          maxDelay: 30000,
        },
        notifications: {
          onSuccess: false,
          onFailure: true,
          onProgress: false,
          channels: [],
        },
        quality: {
          codeReview: true,
          testing: true,
          documentation: true,
          performance: false,
          security: false,
        },
        ...configuration,
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      version: '1.0.0',
    };

    // Save workflow template
    this.workflows.set(workflow.id, workflow);
    await this.saveWorkflowTemplate(workflow);

    logger.info(`Created workflow template: ${workflow.id}`);
    return workflow;
  }

  /**
   * Get workflow execution status
   */
  async getExecutionStatus(executionId: string): Promise<{
    status: ExecutionContext['status'];
    progress: {
      completed: number;
      total: number;
      currentAgent?: string;
    };
    metrics: Record<string, any>;
    errors: string[];
  }> {
    const context = this.activeExecutions.get(executionId);
    
    if (!context) {
      throw new ValidationError(`Execution not found: ${executionId}`);
    }

    return {
      status: context.status,
      progress: {
        completed: context.agentsCompleted.length,
        total: Object.keys(this.agents).length,
        currentAgent: this.getCurrentAgent(context),
      },
      metrics: context.metrics,
      errors: [], // Would track errors in real implementation
    };
  }

  /**
   * Cancel a running workflow execution
   */
  async cancelExecution(executionId: string): Promise<void> {
    const context = this.activeExecutions.get(executionId);
    
    if (!context) {
      throw new ValidationError(`Execution not found: ${executionId}`);
    }

    context.status = 'cancelled';
    this.activeExecutions.delete(executionId);
    
    logger.info(`Cancelled workflow execution: ${executionId}`);
  }

  private initializeAgents(): void {
    this.agents.set('research', new ResearchAgent());
    this.agents.set('coding', new CodingAgent());
    this.agents.set('testing', new TestingAgent());
    this.agents.set('documentation', new DocumentationAgent());
    this.agents.set('devops', new DevOpsAgent());
  }

  private async loadWorkflowTemplates(): Promise<void> {
    // Load predefined workflow templates
    const templates = [
      {
        id: 'quick-prototype',
        name: 'Quick Prototype',
        description: 'Fast prototyping workflow for development',
        agents: ['research', 'coding'],
        configuration: { parallel: false, timeout: 1800000 }, // 30 minutes
      },
      {
        id: 'production-ready',
        name: 'Production Ready',
        description: 'Complete workflow for production-ready nodes',
        agents: ['research', 'coding', 'testing', 'documentation', 'devops'],
        configuration: { parallel: false, timeout: 7200000 }, // 2 hours
      },
      {
        id: 'research-only',
        name: 'Research Analysis',
        description: 'Research and analysis only',
        agents: ['research'],
        configuration: { parallel: false, timeout: 900000 }, // 15 minutes
      },
    ];

    templates.forEach(template => {
      const workflow: Workflow = {
        ...template,
        configuration: {
          ...template.configuration,
          parallel: template.configuration?.parallel ?? false,
          timeout: template.configuration?.timeout ?? 3600000,
          retryPolicy: {
            maxRetries: 3,
            backoffStrategy: 'exponential',
            baseDelay: 1000,
            maxDelay: 30000,
            ...(template.configuration as any)?.retryPolicy,
          },
          notifications: {
            onSuccess: false,
            onFailure: true,
            onProgress: false,
            channels: [],
            ...(template.configuration as any)?.notifications,
          },
          quality: {
            codeReview: true,
            testing: true,
            documentation: true,
            performance: false,
            security: false,
            ...(template.configuration as any)?.quality,
          },
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        version: '1.0.0',
      };
      
      this.workflows.set(workflow.id, workflow);
    });
  }

  private detectInputType(source: string): ExecutionContext['inputType'] {
    if (source.includes('arxiv.org')) return 'arxiv';
    if (source.includes('github.com')) return 'github';
    if (source.startsWith('http')) return 'url';
    return 'file';
  }

  private async executeAgent(agentName: string, context: ExecutionContext, options: any): Promise<any> {
    const agent = this.agents.get(agentName);
    if (!agent) {
      throw new Error(`Unknown agent: ${agentName}`);
    }

    logger.info(`Executing agent: ${agentName}`);

    try {
      let result: any;

      switch (agentName) {
        case 'research':
          result = await agent.extractResearchContent({
            source: context.inputSource,
            method: 'auto',
            options: { extractCode: true, maxContentLength: 1000000 },
          });
          break;

        case 'coding':
          const researchResult = context.artifacts.research;
          if (!researchResult) {
            throw new Error('Research results required for coding agent');
          }
          
          result = await agent.generateComfyUINode({
            researchContent: researchResult,
            nodeSpec: {
              name: this.generateNodeName((researchResult as any)?.title || 'Generated Node'),
              description: (researchResult as any)?.title || 'Generated Node',
              category: 'custom',
            },
            options: {
              includeTests: true,
              includeDocumentation: true,
              optimizationLevel: 'standard',
            },
          });
          break;

        case 'testing':
          const codingResult = context.artifacts.coding;
          if (!codingResult) {
            throw new Error('Coding results required for testing agent');
          }
          
          result = await agent.generateTestSuite({
            nodeCode: (codingResult as any)?.nodeCode || '',
            nodeName: this.generateNodeName((context.artifacts.research as any)?.title || 'Node'),
            testTypes: ['unit', 'edge_cases'],
            framework: 'pytest',
          });
          break;

        case 'documentation':
          const nodeCode = (context.artifacts.coding as any)?.nodeCode;
          if (!nodeCode) {
            throw new Error('Node code required for documentation agent');
          }

          result = await agent.generateDocumentation({
            nodeCode,
            nodeName: this.generateNodeName((context.artifacts.research as any)?.title || 'Node'),
            researchContent: context.artifacts.research,
            documentationTypes: ['readme', 'api'],
          });
          break;

        case 'devops':
          const packageCode = (context.artifacts.coding as any)?.nodeCode;
          if (!packageCode) {
            throw new Error('Node code required for devops agent');
          }

          result = await agent.packageNode({
            nodeCode: packageCode,
            nodeName: this.generateNodeName((context.artifacts.research as any)?.title || 'Node'),
            version: '1.0.0',
            dependencies: (context.artifacts.coding as any)?.dependencies || [],
          });
          break;

        default:
          throw new Error(`Unsupported agent: ${agentName}`);
      }

      return result;

    } catch (error) {
      logger.error(`Agent ${agentName} execution failed:`, error);
      throw error;
    }
  }

  private generateNodeName(title: string): string {
    return title
      .replace(/[^a-zA-Z0-9\s]/g, '')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join('')
      .slice(0, 50);
  }

  private getCurrentAgent(context: ExecutionContext): string | undefined {
    // In a real implementation, this would track the currently executing agent
    return undefined;
  }

  private async generateExecutionReport(
    context: ExecutionContext,
    results: Record<string, any>,
    agentTimes: Record<string, number>,
    errors: string[]
  ): Promise<void> {
    const report = {
      executionId: context.id,
      startTime: context.startTime,
      endTime: context.endTime,
      status: context.status,
      inputSource: context.inputSource,
      outputDirectory: context.outputDirectory,
      agentsCompleted: context.agentsCompleted,
      agentTimes,
      errors,
      summary: {
        totalAgents: Object.keys(results).length,
        successfulAgents: context.agentsCompleted.length,
        totalTime: agentTimes ? Object.values(agentTimes).reduce((sum, time) => sum + time, 0) : 0,
      },
      artifacts: Object.keys(context.artifacts),
    };

    const reportPath = join(context.outputDirectory, 'execution-report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    logger.info(`Generated execution report: ${reportPath}`);
  }

  private async saveArtifacts(executionId: string, artifacts: Record<string, any>): Promise<void> {
    const artifactsPath = join('./artifacts', executionId);
    
    try {
      await fs.mkdir(artifactsPath, { recursive: true });
      
      for (const [agentName, artifact] of Object.entries(artifacts)) {
        const artifactPath = join(artifactsPath, `${agentName}.json`);
        await fs.writeFile(artifactPath, JSON.stringify(artifact, null, 2));
      }
      
      logger.info(`Saved artifacts for execution: ${executionId}`);
    } catch (error) {
      logger.warn(`Failed to save artifacts:`, error);
    }
  }

  private async saveWorkflowTemplate(workflow: Workflow): Promise<void> {
    const templatesPath = './workflow-templates';
    
    try {
      await fs.mkdir(templatesPath, { recursive: true });
      const templatePath = join(templatesPath, `${workflow.id}.json`);
      await fs.writeFile(templatePath, JSON.stringify(workflow, null, 2));
      
      logger.info(`Saved workflow template: ${templatePath}`);
    } catch (error) {
      logger.warn(`Failed to save workflow template:`, error);
    }
  }
}

// MCP Tool definitions
export const workflowOrchestratorExecuteTool: Tool = {
  name: 'workflow_orchestrator_execute',
  description: 'Execute a complete workflow with multiple agents for ComfyUI node generation',
  inputSchema: {
    type: 'object',
    properties: {
      workflowConfig: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          agents: {
            type: 'array',
            items: {
              type: 'string',
              enum: ['research', 'coding', 'testing', 'documentation', 'devops'],
            },
            minItems: 1,
          },
          parallel: { type: 'boolean', default: false },
          timeout: { type: 'number', default: 3600000 },
        },
        required: ['name', 'agents'],
      },
      input: {
        type: 'object',
        properties: {
          source: { type: 'string' },
          outputDirectory: { type: 'string', default: './output' },
          focusAreas: { type: 'array', items: { type: 'string' } },
          qualityLevel: { type: 'string', enum: ['draft', 'development', 'production'], default: 'production' },
        },
        required: ['source'],
      },
      options: {
        type: 'object',
        properties: {
          saveIntermediateResults: { type: 'boolean', default: true },
          continueOnError: { type: 'boolean', default: false },
          generateReport: { type: 'boolean', default: true },
        },
      },
    },
    required: ['workflowConfig', 'input'],
  },
};

export const workflowOrchestratorTemplateTool: Tool = {
  name: 'workflow_orchestrator_template',
  description: 'Create and save a workflow template for reuse',
  inputSchema: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      description: { type: 'string', maxLength: 500 },
      agents: {
        type: 'array',
        items: { type: 'string' },
        minItems: 1,
      },
      configuration: {
        type: 'object',
        additionalProperties: true,
      },
      tags: {
        type: 'array',
        items: { type: 'string' },
        default: [],
      },
    },
    required: ['name', 'description', 'agents'],
  },
};

export const workflowOrchestratorStatusTool: Tool = {
  name: 'workflow_orchestrator_status',
  description: 'Get the status of a running workflow execution',
  inputSchema: {
    type: 'object',
    properties: {
      executionId: {
        type: 'string',
        description: 'ID of the workflow execution to check',
      },
    },
    required: ['executionId'],
  },
};
