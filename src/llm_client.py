"""
LLM Client Infrastructure for MCP Multi-Agent Framework
Provides unified interface for different LLM providers (Claude, OpenAI, etc.)
"""

import os
import json
import asyncio
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import logging

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class LLMResponse:
    """Standardized response from LLM providers"""
    
    def __init__(self, content: str, model: str, usage: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "default")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4000)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Synchronous version of generate"""
        pass


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude client"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not available. Install with: pip install anthropic")
        
        api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or provide in config.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = config.get("model", "claude-3-sonnet-20240229")
    
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Generate response using Claude"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "messages": messages
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            response = self.client.messages.create(**request_params)
            
            content = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={"response_id": response.id}
            )
            
        except Exception as e:
            self.logger.error(f"Claude API error: {str(e)}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Synchronous version of generate"""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a loop, we need to use a different approach
            import concurrent.futures
            import threading

            def run_in_thread():
                # Create a new event loop in a separate thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.generate(prompt, system_prompt, **kwargs))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=60)  # 60 second timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.generate(prompt, system_prompt, **kwargs))


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available. Install with: pip install openai")
        
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide in config.")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4")
    
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Generate response using OpenAI"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )
            
            content = response.choices[0].message.content if response.choices else ""
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={"response_id": response.id}
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def generate_sync(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Synchronous version of generate"""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a loop, we need to use a different approach
            import concurrent.futures

            def run_in_thread():
                # Create a new event loop in a separate thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.generate(prompt, system_prompt, **kwargs))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=60)  # 60 second timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.generate(prompt, system_prompt, **kwargs))


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client for accessing multiple LLM models"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not HTTPX_AVAILABLE:
            raise ImportError("httpx package not available. Install with: pip install httpx")

        api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or provide in config.")

        self.api_key = api_key
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.model = config.get("model", "anthropic/claude-3-sonnet")
        self.site_url = config.get("site_url", "https://github.com/A043-studios/mcp-comfyui-framework")
        self.app_name = config.get("app_name", "MCP ComfyUI Framework")

    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Generate response using OpenRouter API"""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature)
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0
                )
                response.raise_for_status()

                data = response.json()

                content = ""
                if data.get("choices") and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]

                usage = data.get("usage", {})
                usage_info = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }

                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage_info,
                    metadata={"response_id": data.get("id", ""), "provider": "openrouter"}
                )

        except Exception as e:
            self.logger.error(f"OpenRouter API error: {str(e)}")
            raise

    def generate_sync(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Synchronous version of generate"""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a loop, we need to use a different approach
            import concurrent.futures

            def run_in_thread():
                # Create a new event loop in a separate thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.generate(prompt, system_prompt, **kwargs))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=60)  # 60 second timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.generate(prompt, system_prompt, **kwargs))


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for fallback when real clients fail"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "mock-model")

    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Generate mock response"""
        # Log the attempted request for debugging
        self.logger.warning(f"Mock LLM client called with prompt length: {len(prompt)}")
        if system_prompt:
            self.logger.warning(f"System prompt length: {len(system_prompt)}")

        return LLMResponse(
            content="Error: LLM client not properly configured. Please check your API keys and configuration.",
            model=self.model,
            usage={"input_tokens": len(prompt.split()) if prompt else 0, "output_tokens": 0, "total_tokens": len(prompt.split()) if prompt else 0},
            metadata={"error": True, "mock": True, "kwargs": kwargs}
        )

    def generate_sync(self, prompt: str, system_prompt: str = None, **kwargs) -> LLMResponse:
        """Synchronous version of generate"""
        return asyncio.run(self.generate(prompt, system_prompt, **kwargs))


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    CLIENTS = {
        "claude": ClaudeClient,
        "claude-3-sonnet": ClaudeClient,
        "claude-3-opus": ClaudeClient,
        "claude-3-haiku": ClaudeClient,
        "gpt-4": OpenAIClient,
        "gpt-4-turbo": OpenAIClient,
        "gpt-3.5-turbo": OpenAIClient,
        "openai": OpenAIClient,
        "openrouter": OpenRouterClient,
        "anthropic/claude-3-sonnet": OpenRouterClient,
        "anthropic/claude-3-opus": OpenRouterClient,
        "anthropic/claude-3-haiku": OpenRouterClient,
        "openai/gpt-4": OpenRouterClient,
        "openai/gpt-4-turbo": OpenRouterClient,
        "meta-llama/llama-3.1-405b": OpenRouterClient,
        "google/gemini-pro": OpenRouterClient,
        "mistralai/mixtral-8x7b": OpenRouterClient
    }
    
    @classmethod
    def create_client(cls, config: Dict[str, Any]) -> BaseLLMClient:
        """Create appropriate LLM client based on config"""
        model = config.get("model", "anthropic/claude-3-sonnet")

        # Check for explicit provider specification
        provider = config.get("provider", "").lower()
        if provider == "openrouter":
            return OpenRouterClient(config)

        # Check if model contains provider prefix (e.g., "anthropic/claude-3-sonnet")
        if "/" in model:
            return OpenRouterClient(config)

        # Determine client type from model name
        client_type = None
        for key, client_class in cls.CLIENTS.items():
            if model.startswith(key) or key in model:
                client_type = client_class
                break

        if not client_type:
            # Default to OpenRouter for maximum model availability
            client_type = OpenRouterClient

        return client_type(config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models"""
        models = []

        # OpenRouter models (always available if httpx is installed)
        if HTTPX_AVAILABLE:
            models.extend([
                "anthropic/claude-3-sonnet",
                "anthropic/claude-3-opus",
                "anthropic/claude-3-haiku",
                "openai/gpt-4",
                "openai/gpt-4-turbo",
                "openai/gpt-3.5-turbo",
                "meta-llama/llama-3.1-405b",
                "meta-llama/llama-3.1-70b",
                "google/gemini-pro",
                "mistralai/mixtral-8x7b",
                "cohere/command-r-plus"
            ])

        # Direct API models
        if ANTHROPIC_AVAILABLE:
            models.extend([
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307"
            ])

        if OPENAI_AVAILABLE:
            models.extend([
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ])

        return models


class LLMManager:
    """Manager for LLM operations with caching and error handling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Ensure proper API key configuration
        self._setup_api_keys(config)

        # Create client with validated configuration
        try:
            self.client = LLMClientFactory.create_client(config)
            self.logger = logging.getLogger(self.__class__.__name__)
            self.cache = {}  # Simple in-memory cache
            self.logger.info(f"LLMManager initialized with model: {self.client.model}")
        except Exception as e:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.error(f"Failed to initialize LLM client: {str(e)}")
            # Create a fallback configuration
            self.client = self._create_fallback_client(config)
            self.cache = {}

    def _setup_api_keys(self, config: Dict[str, Any]):
        """Setup API keys from environment variables if not provided in config"""
        # OpenRouter API key
        if not config.get("api_key"):
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                config["api_key"] = openrouter_key

        # Anthropic API key for direct Claude access
        if not config.get("anthropic_api_key"):
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                config["anthropic_api_key"] = anthropic_key

        # OpenAI API key for direct OpenAI access
        if not config.get("openai_api_key"):
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                config["openai_api_key"] = openai_key

    def _create_fallback_client(self, config: Dict[str, Any]) -> BaseLLMClient:
        """Create a fallback client when primary initialization fails"""
        # Use original config values where available, fallback to defaults
        fallback_config = {
            "model": config.get("model", "anthropic/claude-3.5-sonnet"),
            "provider": "openrouter",
            "api_key": config.get("api_key") or os.getenv("OPENROUTER_API_KEY", ""),
            "temperature": config.get("temperature", 0.1),
            "max_tokens": config.get("max_tokens", 4000)
        }

        try:
            return OpenRouterClient(fallback_config)
        except Exception as e:
            self.logger.error(f"Fallback client creation failed: {str(e)}")
            # Return a mock client that returns error responses
            return MockLLMClient(fallback_config)
        
    def generate(self, prompt: str, system_prompt: str = None, use_cache: bool = True, **kwargs) -> LLMResponse:
        """Generate response with caching and error handling"""
        
        # Create cache key
        cache_key = None
        if use_cache:
            cache_key = self._create_cache_key(prompt, system_prompt, kwargs)
            if cache_key in self.cache:
                self.logger.info("Using cached response")
                return self.cache[cache_key]
        
        try:
            self.logger.info(f"Generating response using {self.client.model}")
            response = self.client.generate_sync(prompt, system_prompt, **kwargs)
            
            # Cache the response
            if use_cache and cache_key:
                self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            # Return a fallback response
            return LLMResponse(
                content=f"Error: LLM generation failed - {str(e)}",
                model=self.client.model,
                usage={},
                metadata={"error": True}
            )
    
    def _create_cache_key(self, prompt: str, system_prompt: str, kwargs: Dict[str, Any]) -> str:
        """Create cache key from inputs"""
        import hashlib
        
        cache_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": self.client.model,
            "temperature": kwargs.get("temperature", self.client.temperature),
            "max_tokens": kwargs.get("max_tokens", self.client.max_tokens)
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "model": self.client.model
        }
