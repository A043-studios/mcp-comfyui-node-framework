import { z } from 'zod';
import { readFileSync } from 'fs';
import { join } from 'path';

// Configuration schema
const ConfigSchema = z.object({
  server: z.object({
    name: z.string().default('ComfyUI Framework MCP Server'),
    version: z.string().default('1.0.0'),
    description: z.string().default('Professional MCP server for ComfyUI node generation'),
    port: z.number().optional(),
    host: z.string().default('localhost'),
  }),
  
  framework: z.object({
    defaultOutputDir: z.string().default('./output'),
    defaultQuality: z.enum(['draft', 'development', 'production']).default('production'),
    supportedAgents: z.array(z.string()).default([
      'research',
      'coding',
      'testing',
      'documentation',
      'devops',
    ]),
    defaultAgents: z.array(z.string()).default([
      'research',
      'coding',
      'testing',
      'documentation',
    ]),
    maxConcurrentExecutions: z.number().default(5),
    executionTimeout: z.number().default(3600000), // 1 hour
  }),
  
  agents: z.object({
    research: z.object({
      timeout: z.number().default(300000), // 5 minutes
      maxContentLength: z.number().default(1000000), // 1MB
      supportedSources: z.array(z.string()).default(['arxiv', 'github', 'url', 'file']),
      cacheResults: z.boolean().default(true),
      cacheTtl: z.number().default(86400000), // 24 hours
    }),
    
    coding: z.object({
      timeout: z.number().default(600000), // 10 minutes
      templatePath: z.string().default('./templates/comfyui'),
      outputFormat: z.enum(['typescript', 'python']).default('python'),
      includeTests: z.boolean().default(true),
      codeStyle: z.enum(['standard', 'google', 'airbnb']).default('standard'),
    }),
    
    testing: z.object({
      timeout: z.number().default(300000), // 5 minutes
      framework: z.enum(['jest', 'pytest', 'unittest']).default('pytest'),
      coverage: z.boolean().default(true),
      minCoverage: z.number().default(80),
      includeIntegrationTests: z.boolean().default(true),
    }),
    
    documentation: z.object({
      timeout: z.number().default(180000), // 3 minutes
      format: z.enum(['markdown', 'rst', 'html']).default('markdown'),
      includeExamples: z.boolean().default(true),
      includeApiDocs: z.boolean().default(true),
      generateReadme: z.boolean().default(true),
    }),
    
    devops: z.object({
      timeout: z.number().default(120000), // 2 minutes
      includeDockerfile: z.boolean().default(true),
      includeCi: z.boolean().default(true),
      ciProvider: z.enum(['github', 'gitlab', 'jenkins']).default('github'),
      packageManager: z.enum(['npm', 'pip', 'conda']).default('pip'),
    }),
  }),
  
  scraping: z.object({
    rateLimitDelay: z.number().default(1000),
    timeout: z.number().default(30000),
    userAgent: z.string().default('ComfyUI-Framework-MCP/1.0'),
    maxRetries: z.number().default(3),
    supportedMethods: z.array(z.string()).default([
      'auto',
      'arxiv',
      'github',
      'puppeteer',
      'cheerio',
      'jsdom',
    ]),
    puppeteer: z.object({
      headless: z.boolean().default(true),
      timeout: z.number().default(30000),
      viewport: z.object({
        width: z.number().default(1920),
        height: z.number().default(1080),
      }),
    }),
  }),
  
  llm: z.object({
    provider: z.enum(['openrouter', 'anthropic', 'openai']).default('openrouter'),
    model: z.string().default('anthropic/claude-3.5-sonnet'),
    maxTokens: z.number().default(4000),
    temperature: z.number().default(0.1),
    apiKey: z.string().optional(),
    baseUrl: z.string().optional(),
  }),
  
  logging: z.object({
    level: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
    format: z.enum(['json', 'simple', 'detailed']).default('detailed'),
    file: z.string().optional(),
    maxFiles: z.number().default(5),
    maxSize: z.string().default('10m'),
  }),
  
  storage: z.object({
    type: z.enum(['filesystem', 'memory', 's3']).default('filesystem'),
    path: z.string().default('./data'),
    cleanup: z.object({
      enabled: z.boolean().default(true),
      maxAge: z.number().default(604800000), // 7 days
      maxSize: z.string().default('1gb'),
    }),
  }),
  
  security: z.object({
    enableCors: z.boolean().default(true),
    allowedOrigins: z.array(z.string()).default(['*']),
    rateLimit: z.object({
      enabled: z.boolean().default(true),
      windowMs: z.number().default(900000), // 15 minutes
      maxRequests: z.number().default(100),
    }),
    sanitizeInputs: z.boolean().default(true),
    validateOutputs: z.boolean().default(true),
  }),
});

export type Config = z.infer<typeof ConfigSchema>;

class ConfigManager {
  private config: Config;
  private configPath: string;

  constructor(configPath?: string) {
    this.configPath = configPath || join(process.cwd(), 'config.json');
    this.config = this.loadConfig();
  }

  private loadConfig(): Config {
    try {
      // Try to load from file
      const configFile = readFileSync(this.configPath, 'utf-8');
      const rawConfig = JSON.parse(configFile);
      return ConfigSchema.parse(rawConfig);
    } catch (error) {
      // Fall back to environment variables and defaults
      return this.loadFromEnvironment();
    }
  }

  private loadFromEnvironment(): Config {
    const envConfig = {
      server: {
        name: process.env.MCP_SERVER_NAME,
        version: process.env.MCP_SERVER_VERSION,
        port: process.env.MCP_SERVER_PORT ? parseInt(process.env.MCP_SERVER_PORT) : undefined,
        host: process.env.MCP_SERVER_HOST,
      },
      framework: {
        defaultOutputDir: process.env.MCP_DEFAULT_OUTPUT_DIR,
        defaultQuality: process.env.MCP_DEFAULT_QUALITY as 'draft' | 'development' | 'production',
        maxConcurrentExecutions: process.env.MCP_MAX_CONCURRENT 
          ? parseInt(process.env.MCP_MAX_CONCURRENT) 
          : undefined,
      },
      llm: {
        provider: process.env.LLM_PROVIDER as 'openrouter' | 'anthropic' | 'openai',
        model: process.env.LLM_MODEL,
        apiKey: process.env.LLM_API_KEY || process.env.OPENROUTER_API_KEY,
        baseUrl: process.env.LLM_BASE_URL,
      },
      logging: {
        level: process.env.LOG_LEVEL as 'error' | 'warn' | 'info' | 'debug',
        file: process.env.LOG_FILE,
      },
      storage: {
        type: process.env.STORAGE_TYPE as 'filesystem' | 'memory' | 's3',
        path: process.env.STORAGE_PATH,
      },
    };

    // Remove undefined values
    const cleanConfig = JSON.parse(JSON.stringify(envConfig));
    
    return ConfigSchema.parse(cleanConfig);
  }

  public getConfig(): Config {
    return this.config;
  }

  public updateConfig(updates: Partial<Config>): void {
    this.config = ConfigSchema.parse({ ...this.config, ...updates });
  }

  public getAgentConfig(agentName: string): Config['agents'][keyof Config['agents']] {
    const agentConfig = this.config.agents[agentName as keyof Config['agents']];
    if (!agentConfig) {
      throw new Error(`Unknown agent: ${agentName}`);
    }
    return agentConfig;
  }

  public validateConfig(): { valid: boolean; errors: string[] } {
    try {
      ConfigSchema.parse(this.config);
      return { valid: true, errors: [] };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return {
          valid: false,
          errors: error.errors.map(err => `${err.path.join('.')}: ${err.message}`),
        };
      }
      return { valid: false, errors: [String(error)] };
    }
  }
}

// Singleton instance
let configManager: ConfigManager;

export function getConfig(configPath?: string): Config {
  if (!configManager) {
    configManager = new ConfigManager(configPath);
  }
  return configManager.getConfig();
}

export function updateConfig(updates: Partial<Config>): void {
  if (!configManager) {
    configManager = new ConfigManager();
  }
  configManager.updateConfig(updates);
}

export function getAgentConfig(agentName: string): Config['agents'][keyof Config['agents']] {
  if (!configManager) {
    configManager = new ConfigManager();
  }
  return configManager.getAgentConfig(agentName);
}

export function validateConfig(): { valid: boolean; errors: string[] } {
  if (!configManager) {
    configManager = new ConfigManager();
  }
  return configManager.validateConfig();
}

export { ConfigSchema };
