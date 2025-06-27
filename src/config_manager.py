#!/usr/bin/env python3
"""
Configuration Manager for ComfyUI Framework MCP Server
Handles environment variables, validation, and configuration management
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration class for MCP Server"""
    
    # Server settings
    server_name: str = "comfyui-framework"
    server_version: str = "1.0.0"
    
    # API Keys
    openrouter_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Directories
    default_output_dir: str = "./output"
    storage_path: str = "./data"
    logs_dir: str = "./logs"
    
    # Generation settings
    default_quality: str = "production"
    max_concurrent_executions: int = 3
    execution_timeout: int = 3600
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Agent settings
    supported_agents: List[str] = None
    default_agents: List[str] = None
    
    # Web scraping
    rate_limit_delay: float = 1.0
    scraping_timeout: int = 30
    max_content_length: int = 1000000
    
    # Validation
    validate_inputs: bool = True
    strict_mode: bool = False
    
    def __post_init__(self):
        if self.supported_agents is None:
            self.supported_agents = ["research", "coding", "testing", "documentation", "devops"]
        if self.default_agents is None:
            self.default_agents = ["research", "coding", "testing", "documentation"]

class ConfigManager:
    """Manages configuration for the MCP Server"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = MCPServerConfig()
        self.read_only_mode = False
        self._load_configuration()
        self._validate_configuration()
        self._setup_directories_safe()
    
    def _load_configuration(self):
        """Load configuration from environment variables and config file"""
        
        # Load from environment variables first
        self._load_from_environment()
        
        # Load from config file if provided
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file()
        
        logger.info("Configuration loaded successfully")

    def _setup_directories_safe(self):
        """Setup directories with read-only environment handling"""
        import tempfile

        # Test if we can write to the default output directory
        try:
            test_dir = Path(self.config.default_output_dir)
            test_dir.mkdir(parents=True, exist_ok=True)
            # Try to create a test file
            test_file = test_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"Using configured output directory: {self.config.default_output_dir}")
        except (OSError, PermissionError) as e:
            # We're in a read-only environment
            self.read_only_mode = True
            logger.warning(f"Running in read-only environment, will use temporary directories when needed")
            logger.warning(f"Cannot create directories (read-only environment): {e}")

            # Use temporary directories
            temp_base = tempfile.mkdtemp(prefix="comfyui_mcp_")
            self.config.default_output_dir = temp_base
            self.config.storage_path = str(Path(temp_base) / "data")
            self.config.logs_dir = str(Path(temp_base) / "logs")

            logger.info(f"Using temporary directories: {temp_base}")

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # API Keys
        self.config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.config.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Directories
        self.config.default_output_dir = os.getenv("DEFAULT_OUTPUT_DIR", self.config.default_output_dir)
        self.config.storage_path = os.getenv("STORAGE_PATH", self.config.storage_path)
        self.config.logs_dir = os.getenv("LOGS_DIR", self.config.logs_dir)
        
        # Generation settings
        self.config.default_quality = os.getenv("DEFAULT_QUALITY", self.config.default_quality)
        self.config.max_concurrent_executions = int(os.getenv("MAX_CONCURRENT_EXECUTIONS", str(self.config.max_concurrent_executions)))
        self.config.execution_timeout = int(os.getenv("EXECUTION_TIMEOUT", str(self.config.execution_timeout)))
        
        # Logging
        self.config.log_level = os.getenv("LOG_LEVEL", self.config.log_level)
        
        # Agent settings
        default_agents_env = os.getenv("DEFAULT_AGENTS")
        if default_agents_env:
            self.config.default_agents = [a.strip() for a in default_agents_env.split(",")]
        
        # Web scraping
        self.config.rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY", str(self.config.rate_limit_delay)))
        self.config.scraping_timeout = int(os.getenv("SCRAPING_TIMEOUT", str(self.config.scraping_timeout)))
        self.config.max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", str(self.config.max_content_length)))
        
        # Validation
        self.config.validate_inputs = os.getenv("VALIDATE_INPUTS", "true").lower() == "true"
        self.config.strict_mode = os.getenv("STRICT_MODE", "false").lower() == "true"
    
    def _load_from_file(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update config with file values
            for key, value in file_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logger.info(f"Configuration loaded from file: {self.config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_file}: {str(e)}")
    
    def _validate_configuration(self):
        """Validate the configuration"""
        errors = []
        warnings = []
        
        # Check API keys
        api_keys = [
            self.config.openrouter_api_key,
            self.config.anthropic_api_key,
            self.config.openai_api_key
        ]
        
        if not any(api_keys):
            warnings.append("No API keys configured. LLM functionality will be limited.")
        
        # Validate directories (gracefully handle read-only file systems)
        try:
            # Try to create directories, but don't fail if we can't (e.g., in Augment environment)
            Path(self.config.default_output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)
            Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
            logger.info("Successfully validated/created all directories")
        except Exception as e:
            # In Augment or other restricted environments, we can't create directories
            # This is not a fatal error - we'll use temporary directories when needed
            warnings.append(f"Cannot create directories (read-only environment): {str(e)}")
            logger.warning(f"Running in read-only environment, will use temporary directories when needed")
        
        # Validate quality level
        valid_qualities = ["draft", "development", "production"]
        if self.config.default_quality not in valid_qualities:
            errors.append(f"Invalid quality level: {self.config.default_quality}. Must be one of: {valid_qualities}")
        
        # Validate agents
        for agent in self.config.default_agents:
            if agent not in self.config.supported_agents:
                errors.append(f"Invalid default agent: {agent}. Must be one of: {self.config.supported_agents}")
        
        # Validate numeric values
        if self.config.max_concurrent_executions < 1:
            errors.append("max_concurrent_executions must be at least 1")
        
        if self.config.execution_timeout < 60:
            warnings.append("execution_timeout is very low (< 60 seconds)")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        logger.info("Configuration validation completed successfully")
    
    def get_api_key_config(self) -> Dict[str, Optional[str]]:
        """Get API key configuration"""
        return {
            "openrouter": self.config.openrouter_api_key,
            "anthropic": self.config.anthropic_api_key,
            "openai": self.config.openai_api_key
        }
    
    def get_scraper_config(self) -> Dict[str, Any]:
        """Get web scraper configuration"""
        return {
            "rate_limit_delay": self.config.rate_limit_delay,
            "timeout": self.config.scraping_timeout,
            "max_content_length": self.config.max_content_length,
            "user_agent": f"{self.config.server_name}/{self.config.server_version}"
        }
    
    def get_framework_config(self, **overrides) -> Dict[str, Any]:
        """Get framework configuration with optional overrides"""
        base_config = {
            "default_output_dir": self.config.default_output_dir,
            "default_quality": self.config.default_quality,
            "supported_agents": self.config.supported_agents,
            "default_agents": self.config.default_agents,
            "max_concurrent_executions": self.config.max_concurrent_executions,
            "execution_timeout": self.config.execution_timeout,
            "validate_inputs": self.config.validate_inputs,
            "strict_mode": self.config.strict_mode
        }
        
        # Apply overrides
        base_config.update(overrides)
        return base_config
    
    def save_config(self, file_path: Optional[str] = None) -> str:
        """Save current configuration to file"""
        if not file_path:
            file_path = self.config_file or "mcp_server_config.json"
        
        # Convert config to dict, excluding sensitive data
        config_dict = asdict(self.config)
        
        # Remove sensitive API keys from saved config
        sensitive_keys = ["openrouter_api_key", "anthropic_api_key", "openai_api_key"]
        for key in sensitive_keys:
            if key in config_dict:
                config_dict[key] = "***REDACTED***" if config_dict[key] else None
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to: {file_path}")
        return file_path
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration with read-only environment support"""
        config = {
            "level": getattr(logging, self.config.log_level.upper()),
            "format": self.config.log_format,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.config.log_level.upper(),
                    "formatter": self.config.log_format
                }
            }
        }

        # Only add file handler if not in read-only mode
        if not self.read_only_mode:
            config["handlers"]["file"] = {
                "class": "logging.FileHandler",
                "filename": f"{self.config.logs_dir}/mcp_server.log",
                "level": self.config.log_level.upper(),
                "formatter": self.config.log_format
            }

        return config
    
    def __str__(self) -> str:
        """String representation of configuration"""
        config_dict = asdict(self.config)
        
        # Redact sensitive information
        sensitive_keys = ["openrouter_api_key", "anthropic_api_key", "openai_api_key"]
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                config_dict[key] = f"***{config_dict[key][-4:]}***"
        
        return json.dumps(config_dict, indent=2)

# Global config manager instance
config_manager: Optional[ConfigManager] = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get or create the global config manager instance"""
    global config_manager
    
    if config_manager is None:
        config_manager = ConfigManager(config_file)
    
    return config_manager

def initialize_config(config_file: Optional[str] = None) -> ConfigManager:
    """Initialize the configuration manager"""
    global config_manager
    config_manager = ConfigManager(config_file)
    return config_manager
