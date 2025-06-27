import winston from 'winston';
import { getConfig } from '@/config';

const config = getConfig();

// Custom format for detailed logging
const detailedFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.printf(({ timestamp, level, message, stack, ...meta }) => {
    let log = `${timestamp} [${level.toUpperCase()}]`;
    
    if (meta.service) {
      log += ` [${meta.service}]`;
    }
    
    log += `: ${message}`;
    
    if (Object.keys(meta).length > 0) {
      log += ` ${JSON.stringify(meta)}`;
    }
    
    if (stack) {
      log += `\n${stack}`;
    }
    
    return log;
  })
);

// Create base logger
const baseLogger = winston.createLogger({
  level: config.logging.level,
  format: config.logging.format === 'json' 
    ? winston.format.json()
    : config.logging.format === 'simple'
    ? winston.format.simple()
    : detailedFormat,
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        detailedFormat
      ),
    }),
  ],
});

// Add file transport if configured
if (config.logging.file) {
  baseLogger.add(new winston.transports.File({
    filename: config.logging.file,
    maxsize: parseInt(config.logging.maxSize) * 1024 * 1024, // Convert MB to bytes
    maxFiles: config.logging.maxFiles,
    format: winston.format.json(),
  }));
}

export function createLogger(service: string): winston.Logger {
  return baseLogger.child({ service });
}

export { baseLogger as logger };
