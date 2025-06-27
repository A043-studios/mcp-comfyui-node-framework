// Test setup file for Jest
import { jest } from '@jest/globals';

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  log: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Mock environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error';

// Global test timeout
jest.setTimeout(30000);

// Mock external dependencies that might not be available in test environment
jest.mock('puppeteer', () => ({
  launch: jest.fn(() => Promise.resolve({
    newPage: jest.fn(() => Promise.resolve({
      goto: jest.fn(),
      evaluate: jest.fn(),
      setUserAgent: jest.fn(),
      setViewport: jest.fn(),
      setExtraHTTPHeaders: jest.fn(),
      waitForTimeout: jest.fn(),
      waitForSelector: jest.fn(),
    })),
    close: jest.fn(),
  })),
}));

jest.mock('pdf-parse', () => jest.fn(() => Promise.resolve({
  text: 'Mock PDF content',
  numpages: 1,
  info: { Title: 'Mock PDF' },
})));

// Setup global test utilities
global.testUtils = {
  createMockExecutionContext: () => ({
    id: 'test-execution-id',
    inputSource: 'https://example.com/test',
    inputType: 'url' as const,
    outputDirectory: './test-output',
    qualityLevel: 'development' as const,
    agentsCompleted: [],
    artifacts: {},
    metrics: {},
    startTime: new Date().toISOString(),
    status: 'running' as const,
  }),
  
  createMockResearchContent: () => ({
    url: 'https://example.com/test',
    title: 'Test Research Content',
    content: 'This is test research content for testing purposes.',
    contentType: 'webpage',
    metadata: { test: true },
    extractedAt: new Date().toISOString(),
    wordCount: 10,
    language: 'en',
  }),
  
  createMockNodeCode: () => `
class TestNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "test"
    
    def process(self, image, strength):
        return (image * strength,)
`,
};

// Declare global types for TypeScript
declare global {
  var testUtils: {
    createMockExecutionContext: () => any;
    createMockResearchContent: () => any;
    createMockNodeCode: () => string;
  };
}
