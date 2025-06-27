import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import axios from 'axios';
import * as cheerio from 'cheerio';
import { JSDOM } from 'jsdom';
import puppeteer from 'puppeteer';
import { promises as fs } from 'fs';
import { join } from 'path';

import { ResearchContent, NetworkError, ValidationError } from '@/types';
import { getConfig } from '@/config';
import { createLogger } from '@/utils/logger';

const logger = createLogger('WebScraper');

// Input validation schemas
const ScrapingInputSchema = z.object({
  url: z.string().url('Valid URL is required'),
  method: z.enum(['auto', 'cheerio', 'jsdom', 'puppeteer', 'arxiv', 'github']).default('auto'),
  options: z.object({
    timeout: z.number().default(30000),
    maxRetries: z.number().default(3),
    rateLimitDelay: z.number().default(1000),
    userAgent: z.string().optional(),
    headers: z.record(z.string()).optional(),
    extractImages: z.boolean().default(false),
    extractLinks: z.boolean().default(false),
    extractCode: z.boolean().default(true),
    followRedirects: z.boolean().default(true),
    maxContentLength: z.number().default(1000000),
  }).optional(),
});

const BatchScrapingInputSchema = z.object({
  urls: z.array(z.string().url()).min(1, 'At least one URL is required'),
  method: z.enum(['auto', 'cheerio', 'jsdom', 'puppeteer']).default('auto'),
  options: z.object({
    concurrent: z.number().min(1).max(10).default(3),
    delay: z.number().default(1000),
    failFast: z.boolean().default(false),
    saveResults: z.boolean().default(false),
    outputPath: z.string().optional(),
  }).optional(),
});

type ScrapingInput = z.infer<typeof ScrapingInputSchema>;
type BatchScrapingInput = z.infer<typeof BatchScrapingInputSchema>;

export class AdvancedWebScraper {
  private config = getConfig();
  private rateLimiter = new Map<string, number>();
  private cache = new Map<string, ResearchContent>();

  /**
   * Scrape content from a single URL with advanced extraction capabilities
   */
  async scrapeUrl(input: ScrapingInput): Promise<ResearchContent> {
    const validatedInput = ScrapingInputSchema.parse(input);
    const { url, method, options = {} } = validatedInput;

    logger.info(`Scraping URL: ${url} using method: ${method}`);

    try {
      // Check cache first
      const cacheKey = `${url}-${method}`;
      if (this.cache.has(cacheKey)) {
        const cached = this.cache.get(cacheKey)!;
        const cacheAge = Date.now() - new Date(cached.extractedAt).getTime();
        if (cacheAge < 3600000) { // 1 hour cache
          logger.info('Returning cached result');
          return cached;
        }
      }

      // Apply rate limiting
      await this.applyRateLimit(url, options?.rateLimitDelay || 1000);

      // Determine scraping method
      const detectedMethod = method === 'auto' ? this.detectScrapingMethod(url) : method;

      let content: ResearchContent;

      switch (detectedMethod) {
        case 'arxiv':
          content = await this.scrapeArxiv(url, options);
          break;
        case 'github':
          content = await this.scrapeGithub(url, options);
          break;
        case 'puppeteer':
          content = await this.scrapePuppeteer(url, options);
          break;
        case 'jsdom':
          content = await this.scrapeJSDOM(url, options);
          break;
        case 'cheerio':
        default:
          content = await this.scrapeCheerio(url, options);
          break;
      }

      // Cache the result
      this.cache.set(cacheKey, content);

      logger.info(`Successfully scraped ${content.wordCount} words from ${url}`);
      return content;

    } catch (error) {
      logger.error(`Failed to scrape ${url}:`, error);
      throw new NetworkError(`Scraping failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Scrape multiple URLs concurrently with batch processing
   */
  async scrapeUrls(input: BatchScrapingInput): Promise<{
    results: ResearchContent[];
    failed: Array<{ url: string; error: string }>;
    summary: {
      total: number;
      successful: number;
      failed: number;
      totalWords: number;
      averageWords: number;
    };
  }> {
    const validatedInput = BatchScrapingInputSchema.parse(input);
    const { urls, method, options = {} } = validatedInput;

    const concurrent = options?.concurrent || 3;
    const delay = options?.delay || 1000;
    const failFast = options?.failFast || false;

    logger.info(`Batch scraping ${urls.length} URLs with ${concurrent} concurrent requests`);

    const results: ResearchContent[] = [];
    const failed: Array<{ url: string; error: string }> = [];

    try {
      // Process URLs in batches
      for (let i = 0; i < urls.length; i += concurrent) {
        const batch = urls.slice(i, i + concurrent);

        const batchPromises = batch.map(async (url) => {
          try {
            await new Promise(resolve => setTimeout(resolve, delay));
            const result = await this.scrapeUrl({
              url,
              method,
              options: {
                timeout: 30000,
                maxContentLength: 1000000,
                rateLimitDelay: 1000,
                maxRetries: 3,
                extractCode: true,
                extractImages: false,
                extractLinks: true,
                followRedirects: true
              }
            });
            return { success: true, result, url };
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            if (failFast) {
              throw error;
            }
            return { success: false, error: errorMessage, url };
          }
        });

        const batchResults = await Promise.all(batchPromises);

        for (const result of batchResults) {
          if (result.success && result.result) {
            results.push(result.result);
          } else {
            failed.push({ url: result.url, error: result.error || 'Unknown error' });
          }
        }

        logger.info(`Completed batch ${Math.floor(i / concurrent) + 1}/${Math.ceil(urls.length / concurrent)}`);
      }

      // Save results if requested
      if (options?.saveResults && options?.outputPath) {
        await this.saveResults(results, options.outputPath);
      }

      // Calculate summary
      const totalWords = results.reduce((sum, content) => sum + content.wordCount, 0);
      const summary = {
        total: urls.length,
        successful: results.length,
        failed: failed.length,
        totalWords,
        averageWords: results.length > 0 ? Math.round(totalWords / results.length) : 0,
      };

      logger.info(`Batch scraping completed: ${summary.successful}/${summary.total} successful`);

      return {
        results,
        failed,
        summary,
      };

    } catch (error) {
      logger.error(`Batch scraping failed:`, error);
      throw new NetworkError(`Batch scraping failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Extract structured data from web pages
   */
  async extractStructuredData(url: string, selectors: Record<string, string>): Promise<Record<string, any>> {
    logger.info(`Extracting structured data from: ${url}`);

    try {
      const response = await axios.get(url, {
        timeout: 30000,
        headers: {
          'User-Agent': this.config.scraping.userAgent,
        },
      });

      const $ = cheerio.load(response.data);
      const extractedData: Record<string, any> = {};

      for (const [key, selector] of Object.entries(selectors)) {
        const elements = $(selector);
        if (elements.length === 1) {
          extractedData[key] = elements.text().trim();
        } else if (elements.length > 1) {
          extractedData[key] = elements.map((_, el) => $(el).text().trim()).get();
        } else {
          extractedData[key] = null;
        }
      }

      return extractedData;

    } catch (error) {
      logger.error(`Structured data extraction failed:`, error);
      throw new NetworkError(`Structured data extraction failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private detectScrapingMethod(url: string): 'arxiv' | 'github' | 'puppeteer' | 'jsdom' | 'cheerio' {
    if (url.includes('arxiv.org')) return 'arxiv';
    if (url.includes('github.com')) return 'github';
    if (url.includes('javascript') || url.includes('spa') || url.includes('react')) return 'puppeteer';
    return 'cheerio';
  }

  private async applyRateLimit(url: string, delay: number): Promise<void> {
    const domain = new URL(url).hostname;
    const lastRequest = this.rateLimiter.get(domain) || 0;
    const timeSinceLastRequest = Date.now() - lastRequest;
    
    if (timeSinceLastRequest < delay) {
      const waitTime = delay - timeSinceLastRequest;
      logger.debug(`Rate limiting: waiting ${waitTime}ms for ${domain}`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
    
    this.rateLimiter.set(domain, Date.now());
  }

  private async scrapeCheerio(url: string, options: any): Promise<ResearchContent> {
    const response = await axios.get(url, {
      timeout: options.timeout,
      maxRedirects: options.followRedirects ? 5 : 0,
      headers: {
        'User-Agent': options.userAgent || this.config.scraping.userAgent,
        ...options.headers,
      },
    });

    const $ = cheerio.load(response.data);
    
    // Remove unwanted elements
    $('script, style, nav, footer, aside, .advertisement').remove();
    
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

    // Extract images if requested
    if (options.extractImages) {
      metadata.images = $('img').map((_, el) => $(el).attr('src')).get();
    }

    // Extract links if requested
    if (options.extractLinks) {
      metadata.links = $('a[href]').map((_, el) => ({
        text: $(el).text().trim(),
        href: $(el).attr('href'),
      })).get();
    }

    // Extract code blocks if requested
    if (options.extractCode) {
      metadata.codeBlocks = $('pre, code').map((_, el) => $(el).text()).get();
    }

    return {
      url,
      title,
      content: content.slice(0, options.maxContentLength),
      contentType: 'webpage',
      metadata,
      extractedAt: new Date().toISOString(),
      wordCount: content.split(/\s+/).length,
      language: this.detectLanguage(content),
    };
  }

  private async scrapeJSDOM(url: string, options: any): Promise<ResearchContent> {
    const response = await axios.get(url, {
      timeout: options.timeout,
      headers: {
        'User-Agent': options.userAgent || this.config.scraping.userAgent,
      },
    });

    const dom = new JSDOM(response.data, { url });
    const document = dom.window.document;

    // Remove unwanted elements
    const unwantedSelectors = ['script', 'style', 'nav', 'footer', 'aside'];
    unwantedSelectors.forEach(selector => {
      const elements = document.querySelectorAll(selector);
      elements.forEach(el => el.remove());
    });

    const title = document.title || document.querySelector('h1')?.textContent || 'Untitled';
    const content = document.body?.textContent?.replace(/\s+/g, ' ').trim() || '';

    // Extract metadata
    const metadata: Record<string, any> = {};
    const metaTags = document.querySelectorAll('meta');
    metaTags.forEach(meta => {
      const name = meta.getAttribute('name') || meta.getAttribute('property');
      const content = meta.getAttribute('content');
      if (name && content) {
        metadata[name] = content;
      }
    });

    return {
      url,
      title,
      content: content.slice(0, options.maxContentLength),
      contentType: 'webpage-jsdom',
      metadata,
      extractedAt: new Date().toISOString(),
      wordCount: content.split(/\s+/).length,
      language: this.detectLanguage(content),
    };
  }

  private async scrapePuppeteer(url: string, options: any): Promise<ResearchContent> {
    const browser = await puppeteer.launch({
      headless: this.config.scraping.puppeteer.headless,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });

    try {
      const page = await browser.newPage();
      
      await page.setUserAgent(options.userAgent || this.config.scraping.userAgent);
      await page.setViewport({
        width: this.config.scraping.puppeteer.viewport.width || 1920,
        height: this.config.scraping.puppeteer.viewport.height || 1080
      });

      // Set extra headers if provided
      if (options.headers) {
        await page.setExtraHTTPHeaders(options.headers);
      }

      await page.goto(url, {
        waitUntil: 'networkidle2',
        timeout: options.timeout,
      });

      // Wait for dynamic content to load
      await page.waitForTimeout(2000);

      // Extract content
      const pageData = await page.evaluate(() => {
        // Remove unwanted elements
        const unwantedSelectors = ['script', 'style', 'nav', 'footer', 'aside'];
        unwantedSelectors.forEach(selector => {
          const elements = document.querySelectorAll(selector);
          elements.forEach(el => el.remove());
        });

        return {
          title: document.title,
          content: document.body?.innerText || '',
          url: window.location.href,
        };
      });

      // Extract metadata
      const metadata = await page.evaluate(() => {
        const meta: Record<string, any> = {};
        const metaTags = document.querySelectorAll('meta');
        metaTags.forEach(tag => {
          const name = tag.getAttribute('name') || tag.getAttribute('property');
          const content = tag.getAttribute('content');
          if (name && content) {
            meta[name] = content;
          }
        });
        return meta;
      });

      return {
        url: pageData.url,
        title: pageData.title,
        content: pageData.content.slice(0, options.maxContentLength),
        contentType: 'webpage-puppeteer',
        metadata,
        extractedAt: new Date().toISOString(),
        wordCount: pageData.content.split(/\s+/).length,
        language: this.detectLanguage(pageData.content),
      };

    } finally {
      await browser.close();
    }
  }

  private async scrapeArxiv(url: string, options: any): Promise<ResearchContent> {
    // Extract arXiv ID from URL
    const arxivMatch = url.match(/arxiv\.org\/(?:abs|pdf)\/([0-9]{4}\.[0-9]{4,5})/);
    if (!arxivMatch) {
      throw new ValidationError('Invalid arXiv URL format');
    }

    const arxivId = arxivMatch[1];
    const absUrl = `https://arxiv.org/abs/${arxivId}`;

    // Get metadata from abstract page
    const response = await axios.get(absUrl, { timeout: options.timeout });
    const $ = cheerio.load(response.data);
    
    const title = $('.title').text().replace('Title:', '').trim();
    const authors = $('.authors').text().replace('Authors:', '').trim();
    const abstract = $('.abstract').text().replace('Abstract:', '').trim();
    const subjects = $('.subjects').text().replace('Subjects:', '').trim();

    const content = `${title}\n\nAuthors: ${authors}\n\nAbstract: ${abstract}\n\nSubjects: ${subjects}`;

    return {
      url,
      title,
      content,
      contentType: 'arxiv-paper',
      metadata: {
        arxivId,
        authors,
        abstract,
        subjects,
        publishedDate: this.extractArxivDate(arxivId),
      },
      extractedAt: new Date().toISOString(),
      wordCount: content.split(/\s+/).length,
      language: 'en',
    };
  }

  private async scrapeGithub(url: string, options: any): Promise<ResearchContent> {
    // Parse GitHub URL
    const githubMatch = url.match(/github\.com\/([^\/]+)\/([^\/]+)/);
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

    const content = `${repoData.name}\n\n${repoData.description || ''}\n\n${readmeContent}`;

    return {
      url,
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

  private extractArxivDate(arxivId: string): string {
    const year = parseInt(arxivId.substring(0, 2));
    const month = parseInt(arxivId.substring(2, 4));
    const fullYear = year < 50 ? 2000 + year : 1900 + year;
    return `${fullYear}-${month.toString().padStart(2, '0')}-01`;
  }

  private detectLanguage(text: string): string {
    // Simple language detection
    const englishWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'];
    const words = text.toLowerCase().split(/\s+/).slice(0, 100);
    const englishCount = words.filter(word => englishWords.includes(word)).length;
    
    return englishCount > words.length * 0.1 ? 'en' : 'unknown';
  }

  private async saveResults(results: ResearchContent[], outputPath: string): Promise<void> {
    const data = {
      timestamp: new Date().toISOString(),
      count: results.length,
      results,
    };

    await fs.writeFile(outputPath, JSON.stringify(data, null, 2));
    logger.info(`Saved ${results.length} results to ${outputPath}`);
  }
}

// MCP Tool definitions
export const webScraperTool: Tool = {
  name: 'web_scraper_scrape',
  description: 'Scrape content from web pages with advanced extraction capabilities',
  inputSchema: {
    type: 'object',
    properties: {
      url: {
        type: 'string',
        format: 'uri',
        description: 'URL to scrape content from',
      },
      method: {
        type: 'string',
        enum: ['auto', 'cheerio', 'jsdom', 'puppeteer', 'arxiv', 'github'],
        default: 'auto',
        description: 'Scraping method to use',
      },
      options: {
        type: 'object',
        properties: {
          timeout: { type: 'number', default: 30000 },
          maxRetries: { type: 'number', default: 3 },
          rateLimitDelay: { type: 'number', default: 1000 },
          extractImages: { type: 'boolean', default: false },
          extractLinks: { type: 'boolean', default: false },
          extractCode: { type: 'boolean', default: true },
          maxContentLength: { type: 'number', default: 1000000 },
        },
      },
    },
    required: ['url'],
  },
};

export const webScraperBatchTool: Tool = {
  name: 'web_scraper_batch',
  description: 'Scrape multiple URLs concurrently with batch processing',
  inputSchema: {
    type: 'object',
    properties: {
      urls: {
        type: 'array',
        items: { type: 'string', format: 'uri' },
        minItems: 1,
        description: 'Array of URLs to scrape',
      },
      method: {
        type: 'string',
        enum: ['auto', 'cheerio', 'jsdom', 'puppeteer'],
        default: 'auto',
        description: 'Scraping method to use',
      },
      options: {
        type: 'object',
        properties: {
          concurrent: { type: 'number', minimum: 1, maximum: 10, default: 3 },
          delay: { type: 'number', default: 1000 },
          failFast: { type: 'boolean', default: false },
          saveResults: { type: 'boolean', default: false },
          outputPath: { type: 'string' },
        },
      },
    },
    required: ['urls'],
  },
};

export const webScraperStructuredTool: Tool = {
  name: 'web_scraper_structured',
  description: 'Extract structured data from web pages using CSS selectors',
  inputSchema: {
    type: 'object',
    properties: {
      url: {
        type: 'string',
        format: 'uri',
        description: 'URL to extract data from',
      },
      selectors: {
        type: 'object',
        additionalProperties: { type: 'string' },
        description: 'CSS selectors for data extraction (key: selector pairs)',
      },
    },
    required: ['url', 'selectors'],
  },
};
