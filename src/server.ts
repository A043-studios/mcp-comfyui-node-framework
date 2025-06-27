#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

// Import all tools
import { ResearchAgent, researchAgentTool } from '@/tools/research-agent';
import { CodingAgent, codingAgentGenerateTool, codingAgentOptimizeTool, codingAgentValidateTool } from '@/tools/coding-agent';
import { TestingAgent, testingAgentGenerateTool, testingAgentExecuteTool, testingAgentValidateTool } from '@/tools/testing-agent';
import { DocumentationAgent, documentationAgentGenerateTool, documentationAgentApiTool, documentationAgentGuideTool } from '@/tools/documentation-agent';
import { DevOpsAgent, devopsAgentPackageTool, devopsAgentDeployTool, devopsAgentCIPipelineTool } from '@/tools/devops-agent';
import { AdvancedWebScraper, webScraperTool, webScraperBatchTool, webScraperStructuredTool } from '@/tools/web-scraper';
import { CodeAnalyzer, codeAnalyzerTool, codeAnalyzerFilesTool } from '@/tools/code-analyzer';
import { WorkflowOrchestrator, workflowOrchestratorExecuteTool, workflowOrchestratorTemplateTool, workflowOrchestratorStatusTool } from '@/tools/workflow-orchestrator';

// Import resources and prompts
import { resources, handleResourceRequest } from '@/resources';
import { prompts, handlePromptRequest } from '@/prompts';

const logger = createLogger('MCPServer');
const config = getConfig();

class ComfyUIFrameworkServer {
  private server: Server;
  private researchAgent: ResearchAgent;
  private codingAgent: CodingAgent;
  private testingAgent: TestingAgent;
  private documentationAgent: DocumentationAgent;
  private devopsAgent: DevOpsAgent;
  private webScraper: AdvancedWebScraper;
  private codeAnalyzer: CodeAnalyzer;
  private workflowOrchestrator: WorkflowOrchestrator;

  constructor() {
    this.server = new Server(
      {
        name: config.server.name,
        version: config.server.version,
      },
      {
        capabilities: {
          tools: {},
          resources: {},
          prompts: {},
        },
      }
    );

    // Initialize agents
    this.researchAgent = new ResearchAgent();
    this.codingAgent = new CodingAgent();
    this.testingAgent = new TestingAgent();
    this.documentationAgent = new DocumentationAgent();
    this.devopsAgent = new DevOpsAgent();
    this.webScraper = new AdvancedWebScraper();
    this.codeAnalyzer = new CodeAnalyzer();
    this.workflowOrchestrator = new WorkflowOrchestrator();

    this.setupHandlers();
  }

  private setupHandlers(): void {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          // Research Agent Tools
          researchAgentTool,
          
          // Coding Agent Tools
          codingAgentGenerateTool,
          codingAgentOptimizeTool,
          codingAgentValidateTool,
          
          // Testing Agent Tools
          testingAgentGenerateTool,
          testingAgentExecuteTool,
          testingAgentValidateTool,
          
          // Documentation Agent Tools
          documentationAgentGenerateTool,
          documentationAgentApiTool,
          documentationAgentGuideTool,
          
          // DevOps Agent Tools
          devopsAgentPackageTool,
          devopsAgentDeployTool,
          devopsAgentCIPipelineTool,
          
          // Web Scraper Tools
          webScraperTool,
          webScraperBatchTool,
          webScraperStructuredTool,
          
          // Code Analyzer Tools
          codeAnalyzerTool,
          codeAnalyzerFilesTool,
          
          // Workflow Orchestrator Tools
          workflowOrchestratorExecuteTool,
          workflowOrchestratorTemplateTool,
          workflowOrchestratorStatusTool,
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        let result: any;

        // Validate args exists
        if (!args) {
          throw new Error(`Missing arguments for tool: ${name}`);
        }

        switch (name) {
          // Research Agent
          case 'research_agent_extract':
            result = await this.researchAgent.extractResearchContent(args as any);
            break;

          // Coding Agent
          case 'coding_agent_generate':
            result = await this.codingAgent.generateComfyUINode(args as any);
            break;
          case 'coding_agent_optimize':
            if (typeof (args as any).code !== 'string') {
              throw new Error('Missing or invalid code parameter');
            }
            result = await this.codingAgent.optimizeNodeCode((args as any).code, (args as any).optimizationLevel);
            break;
          case 'coding_agent_validate':
            if (typeof (args as any).code !== 'string') {
              throw new Error('Missing or invalid code parameter');
            }
            result = await this.codingAgent.validateNodeCode((args as any).code);
            break;

          // Testing Agent
          case 'testing_agent_generate':
            result = await this.testingAgent.generateTestSuite(args as any);
            break;
          case 'testing_agent_execute':
            result = await this.testingAgent.executeTests(args as any);
            break;
          case 'testing_agent_validate':
            if (typeof (args as any).testCode !== 'string' || typeof (args as any).nodeCode !== 'string') {
              throw new Error('Missing or invalid testCode/nodeCode parameters');
            }
            result = await this.testingAgent.validateTestSuite((args as any).testCode, (args as any).nodeCode);
            break;

          // Documentation Agent
          case 'documentation_agent_generate':
            result = await this.documentationAgent.generateDocumentation(args as any);
            break;
          case 'documentation_agent_api':
            if (typeof (args as any).nodeCode !== 'string' || typeof (args as any).nodeName !== 'string') {
              throw new Error('Missing or invalid nodeCode/nodeName parameters');
            }
            result = await this.documentationAgent.generateApiReference((args as any).nodeCode, (args as any).nodeName);
            break;
          case 'documentation_agent_guide':
            if (typeof (args as any).nodeCode !== 'string' || typeof (args as any).nodeName !== 'string') {
              throw new Error('Missing or invalid nodeCode/nodeName parameters');
            }
            result = await this.documentationAgent.generateUserGuide((args as any).nodeCode, (args as any).nodeName, (args as any).userLevel);
            break;

          // DevOps Agent
          case 'devops_agent_package':
            result = await this.devopsAgent.packageNode(args as any);
            break;
          case 'devops_agent_deploy':
            result = await this.devopsAgent.deployNode(args as any);
            break;
          case 'devops_agent_ci_pipeline':
            if (typeof (args as any).nodeName !== 'string' || typeof (args as any).repository !== 'string') {
              throw new Error('Missing or invalid nodeName/repository parameters');
            }
            result = await this.devopsAgent.setupCIPipeline((args as any).nodeName, (args as any).repository, (args as any).provider);
            break;

          // Web Scraper
          case 'web_scraper_scrape':
            result = await this.webScraper.scrapeUrl(args as any);
            break;
          case 'web_scraper_batch':
            result = await this.webScraper.scrapeUrls(args as any);
            break;
          case 'web_scraper_structured':
            if (typeof (args as any).url !== 'string') {
              throw new Error('Missing or invalid url parameter');
            }
            result = await this.webScraper.extractStructuredData((args as any).url, (args as any).selectors);
            break;

          // Code Analyzer
          case 'code_analyzer_analyze':
            result = await this.codeAnalyzer.analyzeCode(args as any);
            break;
          case 'code_analyzer_files':
            result = await this.codeAnalyzer.analyzeFiles(args as any);
            break;

          // Workflow Orchestrator
          case 'workflow_orchestrator_execute':
            result = await this.workflowOrchestrator.executeWorkflow(args as any);
            break;
          case 'workflow_orchestrator_template':
            result = await this.workflowOrchestrator.createWorkflowTemplate(args as any);
            break;
          case 'workflow_orchestrator_status':
            if (typeof (args as any).executionId !== 'string') {
              throw new Error('Missing or invalid executionId parameter');
            }
            result = await this.workflowOrchestrator.getExecutionStatus((args as any).executionId);
            break;

          default:
            throw new Error(`Unknown tool: ${name}`);
        }

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };

      } catch (error) {
        logger.error(`Tool execution failed for ${name}:`, error);
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                error: error instanceof Error ? error.message : String(error),
                tool: name,
                timestamp: new Date().toISOString(),
              }, null, 2),
            },
          ],
          isError: true,
        };
      }
    });

    // List available resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return { resources };
    });

    // Handle resource requests
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      try {
        const content = await handleResourceRequest(uri);
        return {
          contents: [
            {
              uri,
              mimeType: 'application/json',
              text: content,
            },
          ],
        };
      } catch (error) {
        logger.error(`Resource request failed for ${uri}:`, error);
        throw error;
      }
    });

    // List available prompts
    this.server.setRequestHandler(ListPromptsRequestSchema, async () => {
      return { prompts };
    });

    // Handle prompt requests
    this.server.setRequestHandler(GetPromptRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        const messages = handlePromptRequest(name, args || {});
        return {
          description: `Guided workflow for ${name}`,
          messages: messages.map(msg => ({
            role: msg.role,
            content: {
              type: 'text',
              text: msg.content,
            },
          })),
        };
      } catch (error) {
        logger.error(`Prompt request failed for ${name}:`, error);
        throw error;
      }
    });

    // Error handling
    this.server.onerror = (error) => {
      logger.error('Server error:', error);
    };

    process.on('SIGINT', async () => {
      logger.info('Shutting down server...');
      await this.server.close();
      process.exit(0);
    });
  }

  async start(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    logger.info(`ComfyUI Framework MCP Server started (${config.server.version})`);
  }
}

// Start the server
async function main(): Promise<void> {
  try {
    const server = new ComfyUIFrameworkServer();
    await server.start();
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch((error) => {
    logger.error('Unhandled error:', error);
    process.exit(1);
  });
}

export { ComfyUIFrameworkServer };
