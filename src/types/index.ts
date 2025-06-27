import { z } from 'zod';

// Core execution types
export interface ExecutionContext {
  id: string;
  inputSource: string;
  inputType: 'arxiv' | 'github' | 'url' | 'file';
  outputDirectory: string;
  focusAreas?: string[];
  qualityLevel: 'draft' | 'development' | 'production';
  agentsCompleted: string[];
  artifacts: Record<string, unknown>;
  metrics: Record<string, unknown>;
  startTime: string;
  endTime?: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  error?: string;
}

// Agent configuration
export interface AgentConfig {
  name: string;
  enabled: boolean;
  priority: number;
  timeout: number;
  retries: number;
  dependencies: string[];
  configuration: Record<string, unknown>;
}

// Research content types
export interface ResearchContent {
  url: string;
  title: string;
  content: string;
  contentType: string;
  metadata: Record<string, unknown>;
  extractedAt: string;
  wordCount: number;
  language?: string;
}

// Code analysis types
export interface CodeAnalysis {
  language: string;
  framework: string;
  patterns: CodePattern[];
  dependencies: Dependency[];
  complexity: ComplexityMetrics;
  suggestions: string[];
  comfyuiCompatibility: CompatibilityCheck;
}

export interface CodePattern {
  type: 'class' | 'function' | 'import' | 'constant' | 'interface';
  name: string;
  location: { line: number; column: number };
  signature?: string;
  description?: string;
  usage: string[];
}

export interface Dependency {
  name: string;
  version?: string;
  type: 'runtime' | 'development' | 'peer';
  source: 'npm' | 'pip' | 'conda' | 'github' | 'local';
  required: boolean;
}

export interface ComplexityMetrics {
  cyclomaticComplexity: number;
  cognitiveComplexity: number;
  linesOfCode: number;
  maintainabilityIndex: number;
}

export interface CompatibilityCheck {
  isCompatible: boolean;
  version: string;
  issues: CompatibilityIssue[];
  recommendations: string[];
}

export interface CompatibilityIssue {
  severity: 'error' | 'warning' | 'info';
  message: string;
  location?: { line: number; column: number };
  fix?: string;
}

// ComfyUI node types
export interface ComfyUINode {
  name: string;
  category: string;
  description: string;
  inputTypes: Record<string, NodeInput>;
  returnTypes: string[];
  function: string;
  outputNode?: boolean;
  deprecated?: boolean;
  experimental?: boolean;
}

export interface NodeInput {
  type: string;
  required: boolean;
  default?: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  tooltip?: string;
}

// Workflow types
export interface Workflow {
  id: string;
  name: string;
  description: string;
  agents: string[];
  configuration: WorkflowConfig;
  createdAt: string;
  updatedAt: string;
  version: string;
}

export interface WorkflowConfig {
  parallel: boolean;
  timeout: number;
  retryPolicy: RetryPolicy;
  notifications: NotificationConfig;
  quality: QualityConfig;
}

export interface RetryPolicy {
  maxRetries: number;
  backoffStrategy: 'linear' | 'exponential' | 'fixed';
  baseDelay: number;
  maxDelay: number;
}

export interface NotificationConfig {
  onSuccess: boolean;
  onFailure: boolean;
  onProgress: boolean;
  channels: string[];
}

export interface QualityConfig {
  codeReview: boolean;
  testing: boolean;
  documentation: boolean;
  performance: boolean;
  security: boolean;
}

// Validation schemas
export const ExecutionRequestSchema = z.object({
  inputSource: z.string().url().or(z.string().min(1)),
  outputDirectory: z.string().optional(),
  focusAreas: z.array(z.string()).optional(),
  qualityLevel: z.enum(['draft', 'development', 'production']).default('production'),
  agents: z.array(z.string()).optional(),
  workflow: z.string().optional(),
  configuration: z.record(z.unknown()).optional(),
});

export const WorkflowSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500),
  agents: z.array(z.string()).min(1),
  configuration: z.object({
    parallel: z.boolean().default(false),
    timeout: z.number().positive().default(3600),
    retryPolicy: z.object({
      maxRetries: z.number().min(0).max(5).default(3),
      backoffStrategy: z.enum(['linear', 'exponential', 'fixed']).default('exponential'),
      baseDelay: z.number().positive().default(1000),
      maxDelay: z.number().positive().default(30000),
    }),
  }),
});

export const CodeAnalysisRequestSchema = z.object({
  content: z.string().min(1),
  language: z.string().optional(),
  framework: z.string().optional(),
  analysisType: z.enum(['full', 'patterns', 'dependencies', 'compatibility']).default('full'),
});

export const ValidationRequestSchema = z.object({
  path: z.string().min(1),
  type: z.enum(['file', 'directory']).optional(),
  strict: z.boolean().default(false),
});

// Error types
export class MCPError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'MCPError';
  }
}

export class ValidationError extends MCPError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 'VALIDATION_ERROR', details);
    this.name = 'ValidationError';
  }
}

export class ExecutionError extends MCPError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 'EXECUTION_ERROR', details);
    this.name = 'ExecutionError';
  }
}

export class NetworkError extends MCPError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 'NETWORK_ERROR', details);
    this.name = 'NetworkError';
  }
}

// Export type utilities
export type ExecutionRequest = z.infer<typeof ExecutionRequestSchema>;
export type WorkflowRequest = z.infer<typeof WorkflowSchema>;
export type CodeAnalysisRequest = z.infer<typeof CodeAnalysisRequestSchema>;
export type ValidationRequest = z.infer<typeof ValidationRequestSchema>;
