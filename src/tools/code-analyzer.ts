import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { promises as fs } from 'fs';
import { join, extname } from 'path';

import { CodeAnalysis, CodePattern, Dependency, ComplexityMetrics, CompatibilityCheck, ExecutionError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('CodeAnalyzer');

// Input validation schemas
const CodeAnalysisInputSchema = z.object({
  code: z.string().min(1, 'Code content is required'),
  language: z.enum(['python', 'javascript', 'typescript', 'auto']).default('auto'),
  analysisType: z.enum(['full', 'patterns', 'dependencies', 'complexity', 'compatibility']).default('full'),
  options: z.object({
    includeMetrics: z.boolean().default(true),
    checkComfyUICompatibility: z.boolean().default(true),
    extractDocstrings: z.boolean().default(true),
    analyzeImports: z.boolean().default(true),
    detectFrameworks: z.boolean().default(true),
  }).optional(),
});

const FileAnalysisInputSchema = z.object({
  filePath: z.string().min(1, 'File path is required'),
  recursive: z.boolean().default(false),
  includeTests: z.boolean().default(true),
  filePatterns: z.array(z.string()).default(['*.py', '*.js', '*.ts']),
});

type CodeAnalysisInput = z.infer<typeof CodeAnalysisInputSchema>;
type FileAnalysisInput = z.infer<typeof FileAnalysisInputSchema>;

export class CodeAnalyzer {
  private config = getConfig();

  /**
   * Analyze code for patterns, dependencies, and compatibility
   */
  async analyzeCode(input: CodeAnalysisInput): Promise<CodeAnalysis> {
    const validatedInput = CodeAnalysisInputSchema.parse(input);
    const { code, language, analysisType, options } = validatedInput;

    logger.info(`Analyzing code (${code.length} characters) for ${analysisType}`);

    try {
      // Detect language if auto
      const detectedLanguage = language === 'auto' ? this.detectLanguage(code) : language;
      
      // Detect framework
      const framework = options?.detectFrameworks ? this.detectFramework(code, detectedLanguage) : 'unknown';

      let patterns: CodePattern[] = [];
      let dependencies: Dependency[] = [];
      let complexity: ComplexityMetrics = { cyclomaticComplexity: 0, cognitiveComplexity: 0, linesOfCode: 0, maintainabilityIndex: 0 };
      let comfyuiCompatibility: CompatibilityCheck = { isCompatible: false, version: '', issues: [], recommendations: [] };

      // Perform requested analysis
      if (analysisType === 'full' || analysisType === 'patterns') {
        patterns = await this.extractPatterns(code, detectedLanguage, options);
      }

      if (analysisType === 'full' || analysisType === 'dependencies') {
        dependencies = await this.analyzeDependencies(code, detectedLanguage, options);
      }

      if (analysisType === 'full' || analysisType === 'complexity') {
        complexity = await this.calculateComplexity(code, detectedLanguage);
      }

      if (analysisType === 'full' || analysisType === 'compatibility') {
        if (options?.checkComfyUICompatibility) {
          comfyuiCompatibility = await this.checkComfyUICompatibility(code, patterns);
        }
      }

      // Generate suggestions
      const suggestions = this.generateSuggestions(patterns, dependencies, complexity, comfyuiCompatibility);

      const result: CodeAnalysis = {
        language: detectedLanguage,
        framework,
        patterns,
        dependencies,
        complexity,
        suggestions,
        comfyuiCompatibility,
      };

      logger.info(`Analysis completed: ${patterns.length} patterns, ${dependencies.length} dependencies`);
      return result;

    } catch (error) {
      logger.error(`Code analysis failed:`, error);
      throw new ExecutionError(`Code analysis failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Analyze files and directories for code patterns
   */
  async analyzeFiles(input: FileAnalysisInput): Promise<{
    files: Array<{
      path: string;
      analysis: CodeAnalysis;
      size: number;
      lastModified: string;
    }>;
    summary: {
      totalFiles: number;
      totalLines: number;
      languages: Record<string, number>;
      frameworks: Record<string, number>;
      averageComplexity: number;
      compatibilityScore: number;
    };
  }> {
    const validatedInput = FileAnalysisInputSchema.parse(input);
    const { filePath, recursive, includeTests, filePatterns } = validatedInput;

    logger.info(`Analyzing files in: ${filePath}`);

    try {
      const files = await this.findFiles(filePath, recursive, filePatterns, includeTests);
      const results: Array<{
        path: string;
        analysis: CodeAnalysis;
        size: number;
        lastModified: string;
      }> = [];

      for (const file of files) {
        try {
          const code = await fs.readFile(file, 'utf-8');
          const stats = await fs.stat(file);
          
          const analysis = await this.analyzeCode({
            code,
            language: 'auto',
            analysisType: 'full',
          });

          results.push({
            path: file,
            analysis,
            size: stats.size,
            lastModified: stats.mtime.toISOString(),
          });

        } catch (error) {
          logger.warn(`Failed to analyze file ${file}:`, error);
        }
      }

      // Generate summary
      const summary = this.generateSummary(results);

      logger.info(`Analyzed ${results.length} files`);
      return { files: results, summary };

    } catch (error) {
      logger.error(`File analysis failed:`, error);
      throw new ExecutionError(`File analysis failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Extract specific patterns from code
   */
  async extractPatterns(code: string, language: string, options: any): Promise<CodePattern[]> {
    const patterns: CodePattern[] = [];
    const lines = code.split('\n');

    // Extract classes
    const classRegex = language === 'python' 
      ? /^class\s+(\w+)(?:\([^)]*\))?:/
      : /^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?/;

    lines.forEach((line, index) => {
      const classMatch = line.trim().match(classRegex);
      if (classMatch) {
        patterns.push({
          type: 'class',
          name: classMatch[1],
          location: { line: index + 1, column: line.indexOf(classMatch[1]) },
          signature: line.trim(),
          description: this.extractDescription(lines, index, options.extractDocstrings),
          usage: this.findUsages(code, classMatch[1]),
        });
      }
    });

    // Extract functions/methods
    const functionRegex = language === 'python'
      ? /^(?:\s*)def\s+(\w+)\s*\([^)]*\):/
      : /^(?:\s*)(?:async\s+)?(?:function\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+\s*)?[{=]/;

    lines.forEach((line, index) => {
      const funcMatch = line.trim().match(functionRegex);
      if (funcMatch) {
        patterns.push({
          type: 'function',
          name: funcMatch[1],
          location: { line: index + 1, column: line.indexOf(funcMatch[1]) },
          signature: line.trim(),
          description: this.extractDescription(lines, index, options.extractDocstrings),
          usage: this.findUsages(code, funcMatch[1]),
        });
      }
    });

    // Extract imports
    const importRegex = language === 'python'
      ? /^(?:from\s+[\w.]+\s+)?import\s+([\w\s,.*]+)/
      : /^import\s+([\w\s,{}.*]+)(?:\s+from\s+['"][^'"]+['"])?/;

    lines.forEach((line, index) => {
      const importMatch = line.trim().match(importRegex);
      if (importMatch) {
        patterns.push({
          type: 'import',
          name: importMatch[1].trim(),
          location: { line: index + 1, column: 0 },
          signature: line.trim(),
          description: 'Import statement',
          usage: [],
        });
      }
    });

    // Extract constants
    const constantRegex = language === 'python'
      ? /^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)/
      : /^(?:const|let|var)\s+([A-Z_][A-Z0-9_]*)\s*=\s*(.+)/;

    lines.forEach((line, index) => {
      const constMatch = line.trim().match(constantRegex);
      if (constMatch) {
        patterns.push({
          type: 'constant',
          name: constMatch[1],
          location: { line: index + 1, column: line.indexOf(constMatch[1]) },
          signature: line.trim(),
          description: 'Constant definition',
          usage: this.findUsages(code, constMatch[1]),
        });
      }
    });

    return patterns;
  }

  /**
   * Analyze code dependencies
   */
  async analyzeDependencies(code: string, language: string, options: any): Promise<Dependency[]> {
    const dependencies: Dependency[] = [];
    const lines = code.split('\n');

    if (language === 'python') {
      // Python imports
      lines.forEach(line => {
        const trimmed = line.trim();
        
        // Standard imports
        const importMatch = trimmed.match(/^import\s+([\w.]+)/);
        if (importMatch) {
          const module = importMatch[1].split('.')[0];
          dependencies.push({
            name: module,
            type: 'runtime',
            source: this.getPackageSource(module, 'python'),
            required: true,
          });
        }

        // From imports
        const fromMatch = trimmed.match(/^from\s+([\w.]+)\s+import/);
        if (fromMatch) {
          const module = fromMatch[1].split('.')[0];
          dependencies.push({
            name: module,
            type: 'runtime',
            source: this.getPackageSource(module, 'python'),
            required: true,
          });
        }
      });
    } else if (language === 'javascript' || language === 'typescript') {
      // JavaScript/TypeScript imports
      lines.forEach(line => {
        const trimmed = line.trim();
        
        // ES6 imports
        const importMatch = trimmed.match(/^import\s+.*\s+from\s+['"]([^'"]+)['"]/);
        if (importMatch) {
          const module = importMatch[1];
          dependencies.push({
            name: module,
            type: 'runtime',
            source: this.getPackageSource(module, 'javascript'),
            required: true,
          });
        }

        // Require statements
        const requireMatch = trimmed.match(/require\(['"]([^'"]+)['"]\)/);
        if (requireMatch) {
          const module = requireMatch[1];
          dependencies.push({
            name: module,
            type: 'runtime',
            source: this.getPackageSource(module, 'javascript'),
            required: true,
          });
        }
      });
    }

    // Remove duplicates and add version info
    const uniqueDeps = dependencies.reduce((acc, dep) => {
      const existing = acc.find(d => d.name === dep.name);
      if (!existing) {
        acc.push({
          ...dep,
          version: this.getPackageVersion(dep.name, language),
        });
      }
      return acc;
    }, [] as Dependency[]);

    return uniqueDeps;
  }

  /**
   * Calculate code complexity metrics
   */
  async calculateComplexity(code: string, language: string): Promise<ComplexityMetrics> {
    const lines = code.split('\n');
    const nonEmptyLines = lines.filter(line => line.trim().length > 0);
    
    // Cyclomatic complexity (simplified)
    const cyclomaticComplexity = this.calculateCyclomaticComplexity(code, language);
    
    // Cognitive complexity (simplified)
    const cognitiveComplexity = this.calculateCognitiveComplexity(code, language);
    
    // Maintainability index (simplified)
    const maintainabilityIndex = this.calculateMaintainabilityIndex(code, cyclomaticComplexity);

    return {
      cyclomaticComplexity,
      cognitiveComplexity,
      linesOfCode: nonEmptyLines.length,
      maintainabilityIndex,
    };
  }

  /**
   * Check ComfyUI compatibility
   */
  async checkComfyUICompatibility(code: string, patterns: CodePattern[]): Promise<CompatibilityCheck> {
    const issues: CompatibilityCheck['issues'] = [];
    const recommendations: string[] = [];
    let isCompatible = true;

    // Check for required ComfyUI patterns
    const hasInputTypes = patterns.some(p => p.name === 'INPUT_TYPES');
    const hasReturnTypes = code.includes('RETURN_TYPES');
    const hasFunction = code.includes('FUNCTION');

    if (!hasInputTypes) {
      issues.push({
        severity: 'error',
        message: 'Missing INPUT_TYPES class method',
        fix: 'Add @classmethod INPUT_TYPES method',
      });
      isCompatible = false;
    }

    if (!hasReturnTypes) {
      issues.push({
        severity: 'error',
        message: 'Missing RETURN_TYPES attribute',
        fix: 'Add RETURN_TYPES class attribute',
      });
      isCompatible = false;
    }

    if (!hasFunction) {
      issues.push({
        severity: 'error',
        message: 'Missing FUNCTION attribute',
        fix: 'Add FUNCTION class attribute',
      });
      isCompatible = false;
    }

    // Check for recommended patterns
    if (!code.includes('CATEGORY')) {
      issues.push({
        severity: 'warning',
        message: 'Missing CATEGORY attribute',
        fix: 'Add CATEGORY class attribute for better organization',
      });
      recommendations.push('Add CATEGORY attribute to organize your node');
    }

    // Check for common issues
    if (code.includes('print(') && !code.includes('# debug')) {
      issues.push({
        severity: 'warning',
        message: 'Print statements found in code',
        fix: 'Remove print statements or use logging',
      });
      recommendations.push('Use logging instead of print statements');
    }

    return {
      isCompatible,
      version: '1.0.0',
      issues,
      recommendations,
    };
  }

  private detectLanguage(code: string): 'python' | 'javascript' | 'typescript' {
    if (code.includes('def ') || code.includes('import ') || code.includes('class ')) {
      return 'python';
    }
    if (code.includes('interface ') || code.includes(': string') || code.includes(': number')) {
      return 'typescript';
    }
    return 'javascript';
  }

  private detectFramework(code: string, language: string): string {
    const frameworks = {
      python: {
        'torch': /import torch|from torch/,
        'tensorflow': /import tensorflow|from tensorflow/,
        'numpy': /import numpy|from numpy/,
        'opencv': /import cv2|from cv2/,
        'comfyui': /INPUT_TYPES|RETURN_TYPES|FUNCTION/,
      },
      javascript: {
        'react': /import.*react|from.*react/,
        'vue': /import.*vue|from.*vue/,
        'express': /import.*express|from.*express/,
        'node': /require\(|module\.exports/,
      },
      typescript: {
        'react': /import.*react|from.*react/,
        'vue': /import.*vue|from.*vue/,
        'express': /import.*express|from.*express/,
        'node': /import.*from.*node/,
      },
    };

    const langFrameworks = frameworks[language as keyof typeof frameworks] || {};
    
    for (const [framework, regex] of Object.entries(langFrameworks)) {
      if ((regex as RegExp).test(code)) {
        return framework;
      }
    }

    return 'unknown';
  }

  private extractDescription(lines: string[], index: number, extractDocstrings: boolean): string {
    if (!extractDocstrings) return '';

    // Look for docstring after the definition
    for (let i = index + 1; i < Math.min(index + 5, lines.length); i++) {
      const line = lines[i].trim();
      if (line.startsWith('"""') || line.startsWith("'''")) {
        // Multi-line docstring
        let docstring = line.replace(/^['"]/, '').replace(/['"]$/, '');
        if (!line.endsWith('"""') && !line.endsWith("'''")) {
          // Continue reading until closing quotes
          for (let j = i + 1; j < lines.length; j++) {
            const nextLine = lines[j].trim();
            docstring += ' ' + nextLine.replace(/['"]$/, '');
            if (nextLine.endsWith('"""') || nextLine.endsWith("'''")) {
              break;
            }
          }
        }
        return docstring.trim();
      }
    }

    return '';
  }

  private findUsages(code: string, name: string): string[] {
    const usages: string[] = [];
    const lines = code.split('\n');
    
    lines.forEach((line, index) => {
      if (line.includes(name) && !line.trim().startsWith('def ') && !line.trim().startsWith('class ')) {
        usages.push(`Line ${index + 1}: ${line.trim()}`);
      }
    });

    return usages.slice(0, 5); // Limit to first 5 usages
  }

  private getPackageSource(moduleName: string, language: string): 'npm' | 'pip' | 'conda' | 'github' | 'local' {
    const standardLibs = {
      python: ['os', 'sys', 'json', 'math', 'random', 'datetime', 'collections'],
      javascript: ['fs', 'path', 'http', 'https', 'url', 'crypto'],
    };

    const langStdLibs = standardLibs[language as keyof typeof standardLibs] || [];
    
    if (langStdLibs.includes(moduleName)) {
      return 'local';
    }

    if (moduleName.startsWith('.') || moduleName.startsWith('/')) {
      return 'local';
    }

    return language === 'python' ? 'pip' : 'npm';
  }

  private getPackageVersion(packageName: string, language: string): string {
    // In a real implementation, this would check actual package versions
    return 'latest';
  }

  private calculateCyclomaticComplexity(code: string, language: string): number {
    const complexityKeywords = language === 'python'
      ? ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
      : ['if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch', 'finally'];

    let complexity = 1; // Base complexity
    
    complexityKeywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'g');
      const matches = code.match(regex);
      if (matches) {
        complexity += matches.length;
      }
    });

    return complexity;
  }

  private calculateCognitiveComplexity(code: string, language: string): number {
    // Simplified cognitive complexity calculation
    let complexity = 0;
    const lines = code.split('\n');
    let nestingLevel = 0;

    lines.forEach(line => {
      const trimmed = line.trim();
      
      // Increase nesting for control structures
      if (trimmed.match(/^(if|for|while|try|with|def|class)/)) {
        complexity += 1 + nestingLevel;
        nestingLevel++;
      }
      
      // Decrease nesting
      if (trimmed === '' && nestingLevel > 0) {
        nestingLevel = Math.max(0, nestingLevel - 1);
      }
    });

    return complexity;
  }

  private calculateMaintainabilityIndex(code: string, cyclomaticComplexity: number): number {
    const linesOfCode = code.split('\n').filter(line => line.trim().length > 0).length;
    const halsteadVolume = Math.log2(linesOfCode) * linesOfCode; // Simplified
    
    // Simplified maintainability index calculation
    const mi = Math.max(0, 
      171 - 5.2 * Math.log(halsteadVolume) - 0.23 * cyclomaticComplexity - 16.2 * Math.log(linesOfCode)
    );
    
    return Math.round(mi);
  }

  private generateSuggestions(
    patterns: CodePattern[], 
    dependencies: Dependency[], 
    complexity: ComplexityMetrics, 
    compatibility: CompatibilityCheck
  ): string[] {
    const suggestions: string[] = [];

    if (complexity.cyclomaticComplexity > 10) {
      suggestions.push('Consider breaking down complex functions to improve maintainability');
    }

    if (complexity.maintainabilityIndex < 50) {
      suggestions.push('Code maintainability is low - consider refactoring');
    }

    if (dependencies.length > 20) {
      suggestions.push('Large number of dependencies - consider reducing to improve performance');
    }

    if (!compatibility.isCompatible) {
      suggestions.push('Fix ComfyUI compatibility issues before deployment');
    }

    const functionCount = patterns.filter(p => p.type === 'function').length;
    if (functionCount === 0) {
      suggestions.push('Add functions to improve code organization');
    }

    return suggestions;
  }

  private async findFiles(
    path: string, 
    recursive: boolean, 
    patterns: string[], 
    includeTests: boolean
  ): Promise<string[]> {
    const files: string[] = [];
    
    try {
      const stats = await fs.stat(path);
      
      if (stats.isFile()) {
        if (this.matchesPattern(path, patterns)) {
          files.push(path);
        }
      } else if (stats.isDirectory()) {
        const entries = await fs.readdir(path);
        
        for (const entry of entries) {
          const fullPath = join(path, entry);
          const entryStats = await fs.stat(fullPath);
          
          if (entryStats.isFile()) {
            if (this.matchesPattern(fullPath, patterns)) {
              if (includeTests || !fullPath.includes('test')) {
                files.push(fullPath);
              }
            }
          } else if (entryStats.isDirectory() && recursive) {
            const subFiles = await this.findFiles(fullPath, recursive, patterns, includeTests);
            files.push(...subFiles);
          }
        }
      }
    } catch (error) {
      logger.warn(`Error accessing path ${path}:`, error);
    }
    
    return files;
  }

  private matchesPattern(filePath: string, patterns: string[]): boolean {
    const ext = extname(filePath);
    return patterns.some(pattern => {
      const regex = new RegExp(pattern.replace('*', '.*'));
      return regex.test(filePath) || pattern.includes(ext);
    });
  }

  private generateSummary(results: Array<{ path: string; analysis: CodeAnalysis; size: number; lastModified: string }>): any {
    const totalFiles = results.length;
    const totalLines = results.reduce((sum, r) => sum + r.analysis.complexity.linesOfCode, 0);
    
    const languages: Record<string, number> = {};
    const frameworks: Record<string, number> = {};
    let totalComplexity = 0;
    let compatibleFiles = 0;

    results.forEach(result => {
      languages[result.analysis.language] = (languages[result.analysis.language] || 0) + 1;
      frameworks[result.analysis.framework] = (frameworks[result.analysis.framework] || 0) + 1;
      totalComplexity += result.analysis.complexity.cyclomaticComplexity;
      if (result.analysis.comfyuiCompatibility.isCompatible) {
        compatibleFiles++;
      }
    });

    return {
      totalFiles,
      totalLines,
      languages,
      frameworks,
      averageComplexity: totalFiles > 0 ? Math.round(totalComplexity / totalFiles) : 0,
      compatibilityScore: totalFiles > 0 ? Math.round((compatibleFiles / totalFiles) * 100) : 0,
    };
  }
}

// MCP Tool definitions
export const codeAnalyzerTool: Tool = {
  name: 'code_analyzer_analyze',
  description: 'Analyze code for patterns, dependencies, complexity, and ComfyUI compatibility',
  inputSchema: {
    type: 'object',
    properties: {
      code: {
        type: 'string',
        description: 'Code content to analyze',
      },
      language: {
        type: 'string',
        enum: ['python', 'javascript', 'typescript', 'auto'],
        default: 'auto',
        description: 'Programming language of the code',
      },
      analysisType: {
        type: 'string',
        enum: ['full', 'patterns', 'dependencies', 'complexity', 'compatibility'],
        default: 'full',
        description: 'Type of analysis to perform',
      },
      options: {
        type: 'object',
        properties: {
          includeMetrics: { type: 'boolean', default: true },
          checkComfyUICompatibility: { type: 'boolean', default: true },
          extractDocstrings: { type: 'boolean', default: true },
          analyzeImports: { type: 'boolean', default: true },
          detectFrameworks: { type: 'boolean', default: true },
        },
      },
    },
    required: ['code'],
  },
};

export const codeAnalyzerFilesTool: Tool = {
  name: 'code_analyzer_files',
  description: 'Analyze files and directories for code patterns and metrics',
  inputSchema: {
    type: 'object',
    properties: {
      filePath: {
        type: 'string',
        description: 'Path to file or directory to analyze',
      },
      recursive: {
        type: 'boolean',
        default: false,
        description: 'Recursively analyze subdirectories',
      },
      includeTests: {
        type: 'boolean',
        default: true,
        description: 'Include test files in analysis',
      },
      filePatterns: {
        type: 'array',
        items: { type: 'string' },
        default: ['*.py', '*.js', '*.ts'],
        description: 'File patterns to include in analysis',
      },
    },
    required: ['filePath'],
  },
};
