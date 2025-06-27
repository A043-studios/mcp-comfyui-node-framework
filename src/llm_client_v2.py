#!/usr/bin/env python3
"""
Professional LLM Client for ComfyUI MCP Server v2.0
Handles multiple LLM providers with proper error handling
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    model: str
    usage: Dict[str, Any]
    finish_reason: str
    success: bool = True
    error: Optional[str] = None

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "unknown")
        self.api_key = config.get("api_key", "")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4000)
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        pass

class OpenRouterClient(LLMClient):
    """OpenRouter API client supporting multiple models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/A043-studios/mcp-comfyui-node-framework",
            "X-Title": "ComfyUI MCP Server"
        }
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenRouter API"""
        if not HTTPX_AVAILABLE:
            return LLMResponse(
                content="Error: httpx not available",
                model=self.model,
                usage={},
                finish_reason="error",
                success=False,
                error="httpx dependency not installed"
            )
        
        if not self.api_key:
            return LLMResponse(
                content="Error: No API key configured",
                model=self.model,
                usage={},
                finish_reason="error",
                success=False,
                error="No API key provided"
            )
        
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": False
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"API request failed: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return LLMResponse(
                        content=f"API Error: {response.status_code}",
                        model=self.model,
                        usage={},
                        finish_reason="error",
                        success=False,
                        error=error_msg
                    )
                
                data = response.json()
                
                if "error" in data:
                    error_msg = data["error"].get("message", "Unknown API error")
                    logger.error(f"OpenRouter API error: {error_msg}")
                    return LLMResponse(
                        content=f"API Error: {error_msg}",
                        model=self.model,
                        usage={},
                        finish_reason="error",
                        success=False,
                        error=error_msg
                    )
                
                choice = data["choices"][0]
                content = choice["message"]["content"]
                usage = data.get("usage", {})
                finish_reason = choice.get("finish_reason", "stop")
                
                logger.info(f"Generated {len(content)} characters using {self.model}")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                    finish_reason=finish_reason,
                    success=True
                )
                
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                content=f"Generation Error: {str(e)}",
                model=self.model,
                usage={},
                finish_reason="error",
                success=False,
                error=error_msg
            )

class MockLLMClient(LLMClient):
    """Mock LLM client for testing and fallback"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response"""
        logger.warning("Using mock LLM client - no real AI generation")
        
        # Generate a basic mock response based on prompt content
        if "arxiv" in prompt.lower() or "paper" in prompt.lower():
            content = """# ComfyUI Node Analysis

Based on the research paper analysis, I would recommend creating the following ComfyUI nodes:

## LightControlNode
- **Purpose**: Control lighting in images using diffusion models
- **Inputs**: image, lighting_parameters
- **Outputs**: processed_image
- **Category**: image/lighting

## DiffusionLightingNode  
- **Purpose**: Apply diffusion-based lighting effects
- **Inputs**: image, light_direction, intensity
- **Outputs**: lit_image
- **Category**: image/effects

This is a mock response. For real analysis, configure an API key."""
        
        elif "github" in prompt.lower() or "repository" in prompt.lower():
            content = """# Repository Analysis

The repository appears to contain image processing functionality that could be adapted for ComfyUI:

## Key Components Identified:
- Image processing algorithms
- Model loading utilities
- Configuration management

## Suggested ComfyUI Nodes:
- ImageProcessorNode
- ModelLoaderNode
- ConfigNode

This is a mock response. For real analysis, configure an API key."""
        
        else:
            content = """# Analysis Complete

I've analyzed the provided content and generated appropriate ComfyUI node specifications.

**Note**: This is a mock response. To get real AI-powered analysis, please configure an OpenRouter API key.

## Generated Components:
- Node specifications
- Implementation templates
- Documentation

For production use, please set up proper LLM integration."""
        
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": len(prompt), "completion_tokens": len(content), "total_tokens": len(prompt) + len(content)},
            finish_reason="stop",
            success=True
        )

class LLMManager:
    """High-level LLM manager that handles multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._create_client(config)
    
    def _create_client(self, config: Dict[str, Any]) -> LLMClient:
        """Create appropriate LLM client based on configuration"""
        model = config.get("model", "")
        api_key = config.get("api_key", "")
        
        if not api_key:
            logger.warning("No API key provided, using mock LLM client")
            return MockLLMClient(config)
        
        if "anthropic" in model or "openai" in model or "meta" in model:
            # These models are available through OpenRouter
            logger.info(f"Using OpenRouter client for model: {model}")
            return OpenRouterClient(config)
        else:
            logger.warning(f"Unknown model {model}, using mock client")
            return MockLLMClient(config)
    
    async def generate(self, prompt: str, use_cache: bool = True, **kwargs) -> LLMResponse:
        """Generate response with optional caching"""
        # For now, we'll skip caching and generate directly
        return await self.client.generate(prompt, **kwargs)
    
    async def analyze_content(self, content: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze content and return structured analysis"""
        prompt = f"""Analyze the following content for ComfyUI node creation:

Content: {content[:2000]}...

Analysis Type: {analysis_type}

Please provide:
1. Key concepts and technologies
2. Suggested ComfyUI node types (as simple strings, not objects)
3. Technical requirements
4. Implementation complexity (0.0-1.0)
5. Recommended node categories

IMPORTANT: Format your response as valid JSON with simple string arrays:
{{
    "content_type": "research_paper",
    "key_concepts": ["diffusion models", "computer vision", "image processing"],
    "technical_requirements": ["Python", "PyTorch", "PIL"],
    "suggested_node_types": ["LightControlNode", "DiffusionProcessorNode", "ImageEnhancerNode"],
    "complexity_score": 0.7,
    "categories": ["image", "processing", "ai"],
    "summary": "Brief summary of the content and its potential for ComfyUI nodes"
}}

Only return valid JSON, no additional text."""

        response = await self.generate(prompt)

        if not response.success:
            return {
                "content_type": "error",
                "key_concepts": [],
                "technical_requirements": [],
                "suggested_node_types": [],
                "complexity_score": 0.0,
                "categories": [],
                "summary": f"Analysis failed: {response.error}"
            }

        try:
            # Clean the response to extract JSON
            content = response.content.strip()

            # Find JSON content between braces
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_content = content[start_idx:end_idx]
                analysis = json.loads(json_content)

                # Ensure all required fields are present and properly typed
                return {
                    "content_type": str(analysis.get("content_type", "unknown")),
                    "key_concepts": [str(item) for item in analysis.get("key_concepts", [])],
                    "technical_requirements": [str(item) for item in analysis.get("technical_requirements", [])],
                    "suggested_node_types": [str(item) for item in analysis.get("suggested_node_types", [])],
                    "complexity_score": float(analysis.get("complexity_score", 0.5)),
                    "categories": [str(item) for item in analysis.get("categories", [])],
                    "summary": str(analysis.get("summary", "Content analyzed"))
                }
            else:
                raise json.JSONDecodeError("No valid JSON found", content, 0)

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            # Fallback to text analysis with extracted keywords
            return self._extract_fallback_analysis(response.content, content)
    
    def _extract_fallback_analysis(self, llm_content: str, original_content: str) -> Dict[str, Any]:
        """Extract analysis from LLM content when JSON parsing fails"""
        content_lower = llm_content.lower() + " " + original_content.lower()

        # Extract key concepts using keyword matching
        key_concepts = []
        concept_keywords = [
            "diffusion", "neural network", "deep learning", "computer vision",
            "image processing", "machine learning", "ai", "pytorch", "tensorflow",
            "lighting", "control", "generation", "synthesis", "enhancement"
        ]

        for keyword in concept_keywords:
            if keyword in content_lower:
                key_concepts.append(keyword.replace("_", " ").title())

        # Extract technical requirements
        tech_requirements = ["Python", "ComfyUI"]
        tech_keywords = ["pytorch", "tensorflow", "opencv", "pil", "numpy", "scipy"]

        for keyword in tech_keywords:
            if keyword in content_lower:
                tech_requirements.append(keyword.upper() if keyword in ["pil", "opencv"] else keyword.title())

        # Generate suggested node types based on content
        suggested_nodes = []
        if "light" in content_lower or "illumination" in content_lower:
            suggested_nodes.append("LightControlNode")
        if "diffusion" in content_lower:
            suggested_nodes.append("DiffusionProcessorNode")
        if "image" in content_lower:
            suggested_nodes.append("ImageProcessorNode")
        if not suggested_nodes:
            suggested_nodes.append("CustomNode")

        return {
            "content_type": "text_analysis",
            "key_concepts": key_concepts[:5],  # Limit to 5 concepts
            "technical_requirements": list(set(tech_requirements)),
            "suggested_node_types": suggested_nodes,
            "complexity_score": 0.6,  # Default medium complexity
            "categories": ["image", "processing"],
            "summary": f"Fallback analysis extracted {len(key_concepts)} concepts from content"
        }

    async def generate_node_code(self, node_spec: Dict[str, Any]) -> str:
        """Generate ComfyUI node code from specifications"""
        prompt = f"""Generate a complete ComfyUI node implementation based on these specifications:

Node Name: {node_spec.get('name', 'CustomNode')}
Description: {node_spec.get('description', 'Custom ComfyUI node')}
Category: {node_spec.get('category', 'utils')}
Inputs: {node_spec.get('inputs', [])}
Outputs: {node_spec.get('outputs', [])}

Please generate a complete Python class that follows ComfyUI node conventions:
- Proper INPUT_TYPES class method
- RETURN_TYPES and RETURN_NAMES
- FUNCTION and CATEGORY class attributes
- Complete implementation of the main function

Make the code production-ready with proper error handling."""

        response = await self.generate(prompt)
        return response.content if response.success else f"# Error generating code: {response.error}"

# Factory function for easy instantiation
def create_llm_manager(config: Dict[str, Any]) -> LLMManager:
    """Create and configure LLM manager"""
    return LLMManager(config)
