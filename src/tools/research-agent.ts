import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import axios from 'axios';
import * as cheerio from 'cheerio';
import { JSDOM } from 'jsdom';
import pdfParse from 'pdf-parse';
import simpleGit from 'simple-git';
import { promises as fs } from 'fs';
import { join, extname } from 'path';
import { v4 as uuidv4 } from 'uuid';

import { ResearchContent, NetworkError, ValidationError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('ResearchAgent');

// Input validation schema
const ResearchInputSchema = z.object({
  source: z.string().min(1, 'Source URL or path is required'),
  method: z.enum(['auto', 'arxiv', 'github', 'web', 'pdf', 'file']).default('auto'),
  options: z.object({
    maxContentLength: z.number().default(1000000),
    includeMetadata: z.boolean().default(true),
    extractCode: z.boolean().default(true),
    followLinks: z.boolean().default(false),
    timeout: z.number().default(30000),
  }).optional(),
});

type ResearchInput = z.infer<typeof ResearchInputSchema>;

export class ResearchAgent {
  private config = getConfig();
  private cache = new Map<string, ResearchContent>();

  /**
   * Extract and analyze research content from various sources
   */
  async extractResearchContent(input: ResearchInput): Promise<ResearchContent> {
    const validatedInput = ResearchInputSchema.parse(input);
    const { source, method, options = {} } = validatedInput;

    logger.info(`Extracting research content from: ${source}`);

    // Check cache first
    const cacheKey = `${source}-${method}`;
    if (this.config.agents.research.cacheResults && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      const cacheAge = Date.now() - new Date(cached.extractedAt).getTime();
      if (cacheAge < this.config.agents.research.cacheTtl) {
        logger.info('Returning cached result');
        return cached;
      }
    }

    try {
      let content: ResearchContent;

      // Determine extraction method
      const detectedMethod = method === 'auto' ? this.detectSourceType(source) : method;

      switch (detectedMethod) {
        case 'arxiv':
          content = await this.extractFromArxiv(source, options);
          break;
        case 'github':
          content = await this.extractFromGithub(source, options);
          break;
        case 'pdf':
          content = await this.extractFromPdf(source, options);
          break;
        case 'file':
          content = await this.extractFromFile(source, options);
          break;
        case 'web':
        default:
          content = await this.extractFromWeb(source, options);
          break;
      }

      // Cache the result
      if (this.config.agents.research.cacheResults) {
        this.cache.set(cacheKey, content);
      }

      logger.info(`Successfully extracted ${content.wordCount} words from ${source}`);
      return content;

    } catch (error) {
      logger.error(`Failed to extract content from ${source}:`, error);
      throw new NetworkError(`Failed to extract research content: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Analyze research content for key insights and patterns
   */
  async analyzeResearchContent(content: ResearchContent): Promise<{
    summary: string;
    keyFindings: string[];
    methodologies: string[];
    technologies: string[];
    codePatterns: string[];
    citations: string[];
    relevanceScore: number;
  }> {
    logger.info('Analyzing research content for insights');

    const text = content.content.toLowerCase();
    const words = text.split(/\s+/);

    // Extract key findings (sentences with specific keywords)
    const keyFindings = this.extractKeyFindings(content.content);
    
    // Extract methodologies
    const methodologies = this.extractMethodologies(text);
    
    // Extract technologies and frameworks
    const technologies = this.extractTechnologies(text);
    
    // Extract code patterns if available
    const codePatterns = this.extractCodePatterns(content.content);
    
    // Extract citations
    const citations = this.extractCitations(content.content);
    
    // Calculate relevance score for ComfyUI
    const relevanceScore = this.calculateRelevanceScore(text);
    
    // Generate summary
    const summary = this.generateSummary(content.content, keyFindings);

    return {
      summary,
      keyFindings,
      methodologies,
      technologies,
      codePatterns,
      citations,
      relevanceScore,
    };
  }

  private detectSourceType(source: string): 'arxiv' | 'github' | 'pdf' | 'file' | 'web' {
    if (source.includes('arxiv.org')) return 'arxiv';
    if (source.includes('github.com')) return 'github';
    if (source.endsWith('.pdf') || source.includes('.pdf')) return 'pdf';
    if (source.startsWith('http')) return 'web';
    return 'file';
  }

  private async extractFromArxiv(source: string, options: any): Promise<ResearchContent> {
    // Extract arXiv ID from URL
    const arxivMatch = source.match(/arxiv\.org\/(?:abs|pdf)\/([0-9]{4}\.[0-9]{4,5})/);
    if (!arxivMatch) {
      throw new ValidationError('Invalid arXiv URL format');
    }

    const arxivId = arxivMatch[1];
    const pdfUrl = `https://arxiv.org/pdf/${arxivId}.pdf`;
    const absUrl = `https://arxiv.org/abs/${arxivId}`;

    // Get metadata from abstract page
    const absResponse = await axios.get(absUrl, { timeout: options.timeout });
    const $ = cheerio.load(absResponse.data);
    
    const title = $('.title').text().replace('Title:', '').trim();
    const authors = $('.authors').text().replace('Authors:', '').trim();
    const abstract = $('.abstract').text().replace('Abstract:', '').trim();

    // Download and parse PDF
    const pdfResponse = await axios.get(pdfUrl, { 
      responseType: 'arraybuffer',
      timeout: options.timeout 
    });
    
    const pdfData = await pdfParse(Buffer.from(pdfResponse.data));
    const content = `${title}\n\nAuthors: ${authors}\n\nAbstract: ${abstract}\n\n${pdfData.text}`;

    return {
      url: source,
      title,
      content: content.slice(0, options.maxContentLength),
      contentType: 'arxiv-paper',
      metadata: {
        arxivId,
        authors,
        abstract,
        pages: pdfData.numpages,
        publishedDate: this.extractArxivDate(arxivId),
      },
      extractedAt: new Date().toISOString(),
      wordCount: content.split(/\s+/).length,
      language: 'en',
    };
  }

  private async extractFromGithub(source: string, options: any): Promise<ResearchContent> {
    // Parse GitHub URL
    const githubMatch = source.match(/github\.com\/([^\/]+)\/([^\/]+)/);
    if (!githubMatch) {
      throw new ValidationError('Invalid GitHub URL format');
    }

    const [, owner, repo] = githubMatch;
    const apiUrl = `https://api.github.com/repos/${owner}/${repo}`;

    // Get repository metadata
    const repoResponse = await axios.get(apiUrl, { timeout: options.timeout });
    const repoData = repoResponse.data;

    // Get README content
    let readmeContent = '';
    try {
      const readmeResponse = await axios.get(`${apiUrl}/readme`, {
        headers: { Accept: 'application/vnd.github.v3.raw' },
        timeout: options.timeout,
      });
      readmeContent = readmeResponse.data;
    } catch (error) {
      logger.warn('Could not fetch README:', error);
    }

    // Get key files if extractCode is enabled
    let codeContent = '';
    if (options.extractCode) {
      codeContent = await this.extractKeyCodeFiles(owner, repo, options);
    }

    const content = `${repoData.name}\n\n${repoData.description || ''}\n\n${readmeContent}\n\n${codeContent}`;

    return {
      url: source,
      title: repoData.name,
      content: content.slice(0, options.maxContentLength),
      contentType: 'github-repository',
      metadata: {
        owner,
        repo,
        description: repoData.description,
        language: repoData.language,
        stars: repoData.stargazers_count,
        forks: repoData.forks_count,
        createdAt: repoData.created_at,
        updatedAt: repoData.updated_at,
        topics: repoData.topics,
      },
      extractedAt: new Date().toISOString(),
      wordCount: content.split(/\s+/).length,
      language: 'en',
    };
  }

  private async extractFromWeb(source: string, options: any): Promise<ResearchContent> {
    const response = await axios.get(source, { 
      timeout: options.timeout,
      headers: {
        'User-Agent': this.config.scraping.userAgent,
      },
    });

    const $ = cheerio.load(response.data);
    
    // Remove script and style elements
    $('script, style, nav, footer, aside').remove();
    
    const title = $('title').text().trim() || $('h1').first().text().trim() || 'Untitled';
    const content = $('body').text().replace(/\s+/g, ' ').trim();

    // Extract metadata
    const metadata: Record<string, any> = {};
    $('meta').each((_, el) => {
      const name = $(el).attr('name') || $(el).attr('property');
      const content = $(el).attr('content');
      if (name && content) {
        metadata[name] = content;
      }
    });

    return {
      url: source,
      title,
      content: content.slice(0, options.maxContentLength),
      contentType: 'webpage',
      metadata,
      extractedAt: new Date().toISOString(),
      wordCount: content.split(/\s+/).length,
      language: this.detectLanguage(content),
    };
  }

  private async extractFromPdf(source: string, options: any): Promise<ResearchContent> {
    let buffer: Buffer;

    if (source.startsWith('http')) {
      const response = await axios.get(source, { 
        responseType: 'arraybuffer',
        timeout: options.timeout 
      });
      buffer = Buffer.from(response.data);
    } else {
      buffer = await fs.readFile(source);
    }

    const pdfData = await pdfParse(buffer);
    
    return {
      url: source,
      title: pdfData.info?.Title || 'PDF Document',
      content: pdfData.text.slice(0, options.maxContentLength),
      contentType: 'pdf',
      metadata: {
        pages: pdfData.numpages,
        info: pdfData.info,
      },
      extractedAt: new Date().toISOString(),
      wordCount: pdfData.text.split(/\s+/).length,
      language: this.detectLanguage(pdfData.text),
    };
  }

  private async extractFromFile(source: string, options: any): Promise<ResearchContent> {
    const content = await fs.readFile(source, 'utf-8');
    const ext = extname(source);
    
    return {
      url: source,
      title: source.split('/').pop() || 'File',
      content: content.slice(0, options.maxContentLength),
      contentType: `file-${ext.slice(1) || 'txt'}`,
      metadata: {
        extension: ext,
        size: content.length,
      },
      extractedAt: new Date().toISOString(),
      wordCount: content.split(/\s+/).length,
      language: this.detectLanguage(content),
    };
  }

  private async extractKeyCodeFiles(owner: string, repo: string, options: any): Promise<string> {
    const keyFiles = ['setup.py', 'package.json', 'requirements.txt', 'README.md', 'main.py', 'index.js'];
    let codeContent = '';

    for (const file of keyFiles) {
      try {
        const response = await axios.get(
          `https://api.github.com/repos/${owner}/${repo}/contents/${file}`,
          {
            headers: { Accept: 'application/vnd.github.v3.raw' },
            timeout: options.timeout,
          }
        );
        codeContent += `\n\n--- ${file} ---\n${response.data}`;
      } catch (error) {
        // File doesn't exist, continue
      }
    }

    return codeContent;
  }

  // Analysis helper methods
  private extractKeyFindings(content: string): string[] {
    const sentences = content.split(/[.!?]+/);
    const keywords = ['propose', 'introduce', 'demonstrate', 'achieve', 'improve', 'novel', 'new', 'better'];
    
    return sentences
      .filter(sentence => keywords.some(keyword => sentence.toLowerCase().includes(keyword)))
      .map(sentence => sentence.trim())
      .filter(sentence => sentence.length > 20)
      .slice(0, 10);
  }

  private extractMethodologies(text: string): string[] {
    const methodKeywords = [
      'neural network', 'deep learning', 'machine learning', 'algorithm', 'model',
      'transformer', 'cnn', 'rnn', 'lstm', 'gru', 'attention', 'diffusion',
      'gan', 'vae', 'autoencoder', 'reinforcement learning'
    ];
    
    return methodKeywords.filter(keyword => text.includes(keyword));
  }

  private extractTechnologies(text: string): string[] {
    const techKeywords = [
      'pytorch', 'tensorflow', 'keras', 'numpy', 'opencv', 'pillow', 'scikit-learn',
      'pandas', 'matplotlib', 'seaborn', 'jupyter', 'python', 'javascript', 'typescript',
      'react', 'node.js', 'express', 'fastapi', 'flask', 'docker', 'kubernetes'
    ];
    
    return techKeywords.filter(keyword => text.includes(keyword));
  }

  private extractCodePatterns(content: string): string[] {
    const codeBlocks = content.match(/```[\s\S]*?```/g) || [];
    const patterns: string[] = [];
    
    codeBlocks.forEach((block: string) => {
      if (block.includes('class ')) patterns.push('class definition');
      if (block.includes('def ') || block.includes('function ')) patterns.push('function definition');
      if (block.includes('import ') || block.includes('from ')) patterns.push('imports');
      if (block.includes('torch.') || block.includes('tf.')) patterns.push('ml framework usage');
    });
    
    return [...new Set(patterns)];
  }

  private extractCitations(content: string): string[] {
    const citationPatterns = [
      /\[(\d+)\]/g,
      /\(([^)]+\d{4}[^)]*)\)/g,
      /doi:\s*[\w\-\.\/]+/gi,
      /arxiv:\d{4}\.\d{4,5}/gi,
    ];
    
    const citations: string[] = [];
    citationPatterns.forEach(pattern => {
      const matches = content.match(pattern);
      if (matches) citations.push(...matches);
    });
    
    return [...new Set(citations)].slice(0, 20);
  }

  private calculateRelevanceScore(text: string): number {
    const comfyuiKeywords = [
      'comfyui', 'stable diffusion', 'image generation', 'node', 'workflow',
      'diffusion model', 'latent', 'unet', 'vae', 'clip', 'controlnet',
      'lora', 'embedding', 'checkpoint', 'sampler', 'scheduler'
    ];
    
    const matches = comfyuiKeywords.filter(keyword => text.includes(keyword)).length;
    return Math.min(matches / comfyuiKeywords.length, 1.0);
  }

  private generateSummary(content: string, keyFindings: string[]): string {
    const sentences = content.split(/[.!?]+/).slice(0, 5);
    const summary = sentences.join('. ').trim();
    
    if (keyFindings.length > 0) {
      return `${summary}\n\nKey findings: ${keyFindings.slice(0, 3).join('; ')}`;
    }
    
    return summary;
  }

  private extractArxivDate(arxivId: string): string {
    const year = parseInt(arxivId.substring(0, 2));
    const month = parseInt(arxivId.substring(2, 4));
    const fullYear = year < 50 ? 2000 + year : 1900 + year;
    return `${fullYear}-${month.toString().padStart(2, '0')}-01`;
  }

  private detectLanguage(text: string): string {
    // Simple language detection - could be enhanced with a proper library
    const englishWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'];
    const words = text.toLowerCase().split(/\s+/).slice(0, 100);
    const englishCount = words.filter(word => englishWords.includes(word)).length;
    
    return englishCount > words.length * 0.1 ? 'en' : 'unknown';
  }
}

// MCP Tool definition
export const researchAgentTool: Tool = {
  name: 'research_agent_extract',
  description: 'Extract and analyze research content from papers, repositories, and web sources',
  inputSchema: {
    type: 'object',
    properties: {
      source: {
        type: 'string',
        description: 'URL or file path to extract content from (arXiv, GitHub, PDF, web page, etc.)',
      },
      method: {
        type: 'string',
        enum: ['auto', 'arxiv', 'github', 'web', 'pdf', 'file'],
        description: 'Extraction method (auto-detect if not specified)',
        default: 'auto',
      },
      options: {
        type: 'object',
        properties: {
          maxContentLength: {
            type: 'number',
            description: 'Maximum content length to extract',
            default: 1000000,
          },
          includeMetadata: {
            type: 'boolean',
            description: 'Include metadata in the extraction',
            default: true,
          },
          extractCode: {
            type: 'boolean',
            description: 'Extract code examples and patterns',
            default: true,
          },
          timeout: {
            type: 'number',
            description: 'Request timeout in milliseconds',
            default: 30000,
          },
        },
      },
    },
    required: ['source'],
  },
};
