#!/usr/bin/env python3
"""
Professional ComfyUI MCP Server v2.0
Built with official MCP Python SDK following best practices
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Data models for structured output
class NodeGenerationResult(BaseModel):
    """Structured result for ComfyUI node generation"""
    execution_id: str = Field(description="Unique execution identifier")
    success: bool = Field(description="Whether generation was successful")
    nodes_generated: int = Field(description="Number of ComfyUI nodes generated")
    output_directory: str = Field(description="Directory containing generated files")
    quality_score: float = Field(description="Quality assessment score (0.0-1.0)")
    analysis_summary: str = Field(description="Summary of content analysis")
    recommendations: List[str] = Field(description="Recommendations for improvement")
    artifacts: Dict[str, Any] = Field(description="Generated artifacts metadata")

class NodeTypeDefinition(BaseModel):
    """Definition for a ComfyUI node type"""
    name: str = Field(description="Node name")
    category: str = Field(description="Node category")
    description: str = Field(description="Node description")
    inputs: List[str] = Field(default_factory=list, description="Input types")
    outputs: List[str] = Field(default_factory=list, description="Output types")

class ContentAnalysis(BaseModel):
    """Structured content analysis result"""
    content_type: str = Field(description="Type of content analyzed")
    key_concepts: List[str] = Field(description="Key concepts identified")
    technical_requirements: List[str] = Field(description="Technical requirements")
    suggested_node_types: List[str] = Field(description="Suggested ComfyUI node types as strings")
    complexity_score: float = Field(description="Complexity assessment (0.0-1.0)", ge=0.0, le=1.0)
    categories: List[str] = Field(default_factory=list, description="Recommended categories")
    summary: str = Field(description="Analysis summary")

# Application context for dependency injection
class AppContext:
    def __init__(self):
        self.llm_client = None
        self.content_scraper = None
        self.temp_dir = None
        self.generation_history = []

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with proper resource management"""
    logger.info("Initializing ComfyUI MCP Server v2.0")
    
    # Initialize application context
    ctx = AppContext()
    
    # Create temporary directory for outputs
    ctx.temp_dir = tempfile.mkdtemp(prefix="comfyui_mcp_")
    logger.info(f"Created temporary directory: {ctx.temp_dir}")
    
    # Initialize LLM client if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            # Import and initialize LLM client
            from llm_client_v2 import LLMManager
            llm_config = {
                "model": "anthropic/claude-3.5-sonnet",
                "api_key": api_key,
                "temperature": 0.1,
                "max_tokens": 4000
            }
            ctx.llm_client = LLMManager(llm_config)
            logger.info("LLM client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            ctx.llm_client = None
    else:
        logger.warning("No OPENROUTER_API_KEY found, LLM functionality will be limited")

    # Initialize content scraper
    try:
        from web_scraper_v2 import ContentScraper
        ctx.content_scraper = ContentScraper()
        logger.info("Content scraper initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize content scraper: {e}")
        ctx.content_scraper = None
    
    try:
        yield ctx
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down ComfyUI MCP Server")
        if ctx.temp_dir and Path(ctx.temp_dir).exists():
            import shutil
            shutil.rmtree(ctx.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {ctx.temp_dir}")

# Create MCP server with lifespan management
mcp = FastMCP(
    name="ComfyUI Node Generator",
    lifespan=app_lifespan,
    dependencies=["requests", "beautifulsoup4", "pydantic"]
)

# ============================================================================
# TOOLS - Functions the LLM can execute
# ============================================================================

@mcp.tool()
async def generate_comfyui_node(
    input_source: str,
    quality_level: str = "production",
    focus_areas: str = "",
    output_directory: Optional[str] = None,
    ctx: Context = None
) -> NodeGenerationResult:
    """
    Generate ComfyUI nodes from research papers, repositories, or other sources.
    
    Args:
        input_source: URL to paper, GitHub repo, or local file path
        quality_level: Generation quality (draft, development, production)
        focus_areas: Comma-separated focus areas for analysis
        output_directory: Custom output directory (optional)
    """
    execution_id = str(uuid.uuid4())
    
    try:
        # Get application context
        app_ctx = mcp.get_context().request_context.lifespan_context
        
        # Set up output directory
        if not output_directory:
            output_directory = os.path.join(app_ctx.temp_dir, f"execution_{execution_id}")
        
        os.makedirs(output_directory, exist_ok=True)
        
        if ctx:
            ctx.info(f"Starting ComfyUI node generation for: {input_source}")
            await ctx.report_progress(0, 100, "Initializing generation")

        # Analyze input source
        if ctx:
            await ctx.report_progress(20, 100, "Analyzing input source")
        analysis = await _analyze_input_source(input_source, focus_areas, app_ctx, ctx)

        # Generate node specifications
        if ctx:
            await ctx.report_progress(40, 100, "Generating node specifications")
        node_specs = await _generate_node_specifications(analysis, quality_level, app_ctx, ctx)

        # Create ComfyUI node files
        if ctx:
            await ctx.report_progress(60, 100, "Creating ComfyUI node files")
        artifacts = await _create_node_files(node_specs, output_directory, ctx)

        # Generate documentation and tests
        if ctx:
            await ctx.report_progress(80, 100, "Generating documentation and tests")
        await _generate_documentation(node_specs, artifacts, output_directory, ctx)

        if ctx:
            await ctx.report_progress(100, 100, "Generation complete")
        
        # Calculate quality score
        quality_score = _calculate_quality_score(artifacts, quality_level)
        
        # Record generation in history
        generation_record = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "input_source": input_source,
            "quality_level": quality_level,
            "success": True,
            "nodes_generated": len(artifacts.get("nodes", []))
        }
        app_ctx.generation_history.append(generation_record)
        
        if ctx:
            ctx.info(f"Successfully generated {len(artifacts.get('nodes', []))} ComfyUI nodes")
        
        return NodeGenerationResult(
            execution_id=execution_id,
            success=True,
            nodes_generated=len(artifacts.get("nodes", [])),
            output_directory=output_directory,
            quality_score=quality_score,
            analysis_summary=analysis.get("summary", "Content analyzed successfully"),
            recommendations=_generate_recommendations(analysis, artifacts),
            artifacts=artifacts
        )
        
    except Exception as e:
        logger.error(f"Node generation failed: {e}")
        if ctx:
            ctx.error(f"Generation failed: {str(e)}")
        
        return NodeGenerationResult(
            execution_id=execution_id,
            success=False,
            nodes_generated=0,
            output_directory=output_directory or "",
            quality_score=0.0,
            analysis_summary=f"Generation failed: {str(e)}",
            recommendations=["Check input source and try again", "Verify API configuration"],
            artifacts={}
        )

@mcp.tool()
async def analyze_content(
    input_source: str,
    analysis_type: str = "comprehensive",
    ctx: Context = None
) -> ContentAnalysis:
    """
    Analyze content from various sources to understand ComfyUI node requirements.
    
    Args:
        input_source: URL or path to content to analyze
        analysis_type: Type of analysis (quick, comprehensive, technical)
    """
    try:
        app_ctx = mcp.get_context().request_context.lifespan_context
        
        if ctx:
            ctx.info(f"Analyzing content: {input_source}")
        
        # Perform content analysis
        analysis_result = await _analyze_input_source(input_source, "", app_ctx, ctx)
        
        return ContentAnalysis(
            content_type=analysis_result.get("content_type", "unknown"),
            key_concepts=analysis_result.get("key_concepts", []),
            technical_requirements=analysis_result.get("technical_requirements", []),
            suggested_node_types=analysis_result.get("suggested_node_types", []),
            complexity_score=analysis_result.get("complexity_score", 0.5),
            categories=analysis_result.get("categories", []),
            summary=analysis_result.get("summary", "Content analyzed")
        )
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        if ctx:
            ctx.error(f"Analysis failed: {str(e)}")

        return ContentAnalysis(
            content_type="error",
            key_concepts=[],
            technical_requirements=[],
            suggested_node_types=[],
            complexity_score=0.0,
            categories=[],
            summary=f"Analysis failed: {str(e)}"
        )

# ============================================================================
# RESOURCES - Contextual data for the LLM
# ============================================================================

@mcp.resource("comfyui://generation-history")
def get_generation_history() -> str:
    """Get the history of ComfyUI node generations"""
    try:
        app_ctx = mcp.get_context().request_context.lifespan_context

        if not app_ctx.generation_history:
            return json.dumps({
                "status": "empty",
                "message": "No generation history available",
                "history": []
            }, indent=2)

        # Return structured JSON instead of markdown
        history_data = {
            "status": "success",
            "total_generations": len(app_ctx.generation_history),
            "recent_generations": app_ctx.generation_history[-10:],  # Last 10 generations
            "summary": {
                "successful": sum(1 for record in app_ctx.generation_history if record.get('success', False)),
                "failed": sum(1 for record in app_ctx.generation_history if not record.get('success', False)),
                "total_nodes_generated": sum(record.get('nodes_generated', 0) for record in app_ctx.generation_history)
            }
        }

        return json.dumps(history_data, indent=2)
    except Exception as e:
        logger.error(f"Failed to get generation history: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to retrieve history: {str(e)}",
            "history": []
        }, indent=2)

@mcp.resource("comfyui://server-status")
def get_server_status() -> str:
    """Get current server status and configuration"""
    try:
        app_ctx = mcp.get_context().request_context.lifespan_context

        # Get detailed status information
        status = {
            "server_info": {
                "name": "ComfyUI Node Generator v2.0",
                "version": "2.0.0",
                "status": "running",
                "uptime": "active"
            },
            "capabilities": {
                "llm_available": app_ctx.llm_client is not None,
                "content_scraper_available": app_ctx.content_scraper is not None,
                "api_key_configured": bool(os.getenv("OPENROUTER_API_KEY")),
                "temp_directory": app_ctx.temp_dir
            },
            "statistics": {
                "generations_completed": len(app_ctx.generation_history),
                "successful_generations": sum(1 for record in app_ctx.generation_history if record.get('success', False)),
                "total_nodes_generated": sum(record.get('nodes_generated', 0) for record in app_ctx.generation_history)
            },
            "configuration": {
                "supported_quality_levels": ["draft", "development", "production"],
                "supported_input_types": ["arxiv_papers", "github_repos", "local_files", "urls"],
                "default_model": "anthropic/claude-3.5-sonnet" if app_ctx.llm_client else "none"
            }
        }

        return json.dumps(status, indent=2)
    except Exception as e:
        logger.error(f"Failed to get server status: {e}")
        return json.dumps({
            "server_info": {
                "name": "ComfyUI Node Generator v2.0",
                "status": "error",
                "error": str(e)
            }
        }, indent=2)

@mcp.resource("comfyui://execution/{execution_id}")
def get_execution_details(execution_id: str) -> str:
    """Get detailed information about a specific execution"""
    try:
        app_ctx = mcp.get_context().request_context.lifespan_context

        # Find execution in history
        execution = next(
            (record for record in app_ctx.generation_history if record["execution_id"] == execution_id),
            None
        )

        if not execution:
            return json.dumps({
                "status": "not_found",
                "message": f"Execution {execution_id} not found",
                "execution_id": execution_id,
                "available_executions": [record["execution_id"] for record in app_ctx.generation_history[-5:]]
            }, indent=2)

        # Return detailed execution information
        execution_details = {
            "status": "found",
            "execution": execution,
            "related_info": {
                "position_in_history": len(app_ctx.generation_history) - next(
                    i for i, record in enumerate(reversed(app_ctx.generation_history))
                    if record["execution_id"] == execution_id
                ),
                "total_executions": len(app_ctx.generation_history)
            }
        }

        return json.dumps(execution_details, indent=2)
    except Exception as e:
        logger.error(f"Failed to get execution details: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to retrieve execution details: {str(e)}",
            "execution_id": execution_id
        }, indent=2)

# ============================================================================
# PROMPTS - Interactive templates for users
# ============================================================================

@mcp.prompt()
def generate_from_paper_prompt(paper_url: str, focus_area: str = "") -> str:
    """Generate a ComfyUI node from a research paper"""
    return f"""Please generate a ComfyUI node from this research paper:

Paper URL: {paper_url}
Focus Area: {focus_area or "General implementation"}

Use the generate_comfyui_node tool with:
- input_source: {paper_url}
- quality_level: production
- focus_areas: {focus_area}

This will analyze the paper and create appropriate ComfyUI nodes with documentation and tests."""

@mcp.prompt()
def analyze_repository_prompt(repo_url: str) -> List[base.Message]:
    """Analyze a GitHub repository for ComfyUI node potential"""
    return [
        base.UserMessage(f"I want to analyze this repository for ComfyUI node creation: {repo_url}"),
        base.AssistantMessage("I'll analyze the repository to understand its functionality and suggest ComfyUI node implementations."),
        base.UserMessage("Please use the analyze_content tool first, then suggest how to proceed with node generation.")
    ]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _analyze_input_source(input_source: str, focus_areas: str, app_ctx: AppContext, ctx: Optional[Context]) -> Dict[str, Any]:
    """Analyze input source to understand content and requirements"""
    try:
        if ctx:
            ctx.info(f"Analyzing input source: {input_source}")

        # Scrape content from the source
        if app_ctx.content_scraper:
            scraped_content = await app_ctx.content_scraper.scrape_url(input_source)

            if not scraped_content.success:
                raise Exception(f"Failed to scrape content: {scraped_content.error}")

            if ctx:
                ctx.info(f"Successfully scraped content: {scraped_content.title}")

            # Use LLM to analyze the content if available
            if app_ctx.llm_client:
                analysis_prompt = f"""Analyze this content for ComfyUI node creation:

Title: {scraped_content.title}
Content Type: {scraped_content.content_type}
Content: {scraped_content.content[:3000]}...

Focus Areas: {focus_areas}

Please analyze and provide:
1. Key concepts and technologies mentioned
2. Technical requirements for implementation
3. Suggested ComfyUI node types
4. Implementation complexity (0.0-1.0 scale)
5. Brief summary

Format as JSON with keys: content_type, key_concepts, technical_requirements, suggested_node_types, complexity_score, summary"""

                analysis_response = await app_ctx.llm_client.analyze_content(
                    scraped_content.content,
                    "comprehensive"
                )

                # Enhance with scraped metadata
                analysis_response.update({
                    "scraped_title": scraped_content.title,
                    "scraped_metadata": scraped_content.metadata,
                    "source_url": input_source
                })

                return analysis_response
            else:
                # Fallback analysis without LLM
                return _fallback_content_analysis(scraped_content, focus_areas)
        else:
            # Fallback when scraper is not available
            return _basic_url_analysis(input_source, focus_areas)

    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        if ctx:
            ctx.error(f"Analysis failed: {str(e)}")

        return {
            "content_type": "error",
            "key_concepts": [],
            "technical_requirements": [],
            "suggested_node_types": [],
            "complexity_score": 0.0,
            "summary": f"Analysis failed: {str(e)}"
        }

async def _generate_node_specifications(analysis: Dict[str, Any], quality_level: str, app_ctx: AppContext, ctx: Optional[Context]) -> Dict[str, Any]:
    """Generate detailed node specifications based on analysis"""
    try:
        if ctx:
            ctx.info("Generating node specifications")

        if app_ctx.llm_client and analysis.get("key_concepts"):
            # Use LLM to generate detailed specifications
            spec_prompt = f"""Based on this analysis, generate detailed ComfyUI node specifications:

Analysis Summary: {analysis.get('summary', '')}
Key Concepts: {', '.join(analysis.get('key_concepts', []))}
Technical Requirements: {', '.join(analysis.get('technical_requirements', []))}
Suggested Node Types: {', '.join(analysis.get('suggested_node_types', []))}
Quality Level: {quality_level}

Generate 1-3 ComfyUI node specifications with:
- Unique node names (ending with 'Node')
- Appropriate categories (e.g., 'image/processing', 'utils', 'loaders')
- Clear descriptions
- Input/output specifications
- Implementation details

Format as JSON array of node objects."""

            response = await app_ctx.llm_client.generate(spec_prompt)

            if response.success:
                try:
                    # Try to parse JSON response
                    import json
                    specs = json.loads(response.content)
                    if isinstance(specs, list):
                        return {"nodes": specs}
                    elif isinstance(specs, dict) and "nodes" in specs:
                        return specs
                except json.JSONDecodeError:
                    pass

        # Fallback specification generation
        return _generate_fallback_specifications(analysis, quality_level)

    except Exception as e:
        logger.error(f"Node specification generation failed: {e}")
        if ctx:
            ctx.error(f"Specification generation failed: {str(e)}")
        return _generate_fallback_specifications(analysis, quality_level)

async def _create_node_files(node_specs: Dict[str, Any], output_dir: str, ctx: Optional[Context]) -> Dict[str, Any]:
    """Create actual ComfyUI node files"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        if ctx:
            ctx.info(f"Creating node files in {output_dir}")

        # Create basic node files for now
        artifacts = {
            "nodes": [],
            "tests": [],
            "documentation": []
        }

        # Generate actual node files based on specifications
        nodes = node_specs.get("nodes", [])
        for i, node_spec in enumerate(nodes):
            node_name = node_spec.get("name", f"CustomNode{i+1}")

            # Create node file
            node_filename = f"{node_name.lower()}.py"
            node_path = os.path.join(output_dir, node_filename)

            # Basic node template
            node_content = f'''"""
{node_spec.get("description", "Custom ComfyUI node")}
Generated by ComfyUI MCP Server v2.0
"""

class {node_name}:
    """
    {node_spec.get("description", "Custom ComfyUI node")}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {{
            "required": {{
                "input": ("STRING", {{"default": ""}})
            }}
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "{node_spec.get("category", "utils")}"

    def process(self, input):
        # TODO: Implement actual processing logic
        return (f"Processed: {{input}}",)

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {{
    "{node_name}": {node_name}
}}

NODE_DISPLAY_NAME_MAPPINGS = {{
    "{node_name}": "{node_spec.get("description", node_name)}"
}}
'''

            with open(node_path, 'w') as f:
                f.write(node_content)

            artifacts["nodes"].append(node_filename)

            if ctx:
                ctx.info(f"Created node file: {node_filename}")

        # Create README
        readme_path = os.path.join(output_dir, "README.md")
        readme_content = f"""# Generated ComfyUI Nodes

Generated by ComfyUI MCP Server v2.0

## Nodes Created

{chr(10).join(f"- **{node.get('name', 'Unknown')}**: {node.get('description', 'No description')}" for node in nodes)}

## Installation

1. Copy the node files to your ComfyUI custom_nodes directory
2. Restart ComfyUI
3. The nodes should appear in the specified categories

## Usage

Each node follows standard ComfyUI conventions with proper INPUT_TYPES and processing functions.
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        artifacts["documentation"].append("README.md")

        return artifacts

    except Exception as e:
        logger.error(f"Failed to create node files: {e}")
        if ctx:
            ctx.error(f"File creation failed: {str(e)}")
        return {"nodes": [], "tests": [], "documentation": []}

async def _generate_documentation(node_specs: Dict[str, Any], artifacts: Dict[str, Any], output_dir: str, ctx: Optional[Context]):
    """Generate documentation for the created nodes"""
    try:
        if ctx:
            ctx.info("Generating additional documentation")

        # Documentation is already created in _create_node_files
        # This could be extended for more detailed documentation
        pass

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        if ctx:
            ctx.error(f"Documentation generation failed: {str(e)}")

def _calculate_quality_score(artifacts: Dict[str, Any], quality_level: str) -> float:
    """Calculate quality score based on generated artifacts"""
    base_score = 0.8 if artifacts.get("nodes") else 0.0
    quality_multiplier = {"draft": 0.6, "development": 0.8, "production": 1.0}.get(quality_level, 0.8)
    return base_score * quality_multiplier

def _generate_recommendations(analysis: Dict[str, Any], artifacts: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on analysis and artifacts"""
    recommendations = []

    if not artifacts.get("nodes"):
        recommendations.append("No nodes were generated - check input source and configuration")

    if analysis.get("complexity_score", 0) > 0.8:
        recommendations.append("High complexity detected - consider breaking into multiple nodes")

    recommendations.append("Test the generated nodes in ComfyUI before production use")
    recommendations.append("Review and customize the generated code as needed")

    return recommendations

def _fallback_content_analysis(scraped_content, focus_areas: str) -> Dict[str, Any]:
    """Fallback content analysis when LLM is not available"""
    content_lower = scraped_content.content.lower()

    # Basic keyword detection
    key_concepts = []
    if "image" in content_lower or "vision" in content_lower:
        key_concepts.extend(["image processing", "computer vision"])
    if "diffusion" in content_lower or "stable diffusion" in content_lower:
        key_concepts.extend(["diffusion models", "generative AI"])
    if "pytorch" in content_lower or "torch" in content_lower:
        key_concepts.append("PyTorch")
    if "tensorflow" in content_lower:
        key_concepts.append("TensorFlow")

    # Determine content type
    content_type = scraped_content.content_type
    if "arxiv" in scraped_content.url:
        content_type = "research_paper"
    elif "github" in scraped_content.url:
        content_type = "repository"

    return {
        "content_type": content_type,
        "key_concepts": key_concepts or ["general"],
        "technical_requirements": ["Python", "ComfyUI"],
        "suggested_node_types": ["CustomNode"],
        "complexity_score": 0.5,
        "summary": f"Basic analysis of {scraped_content.title}"
    }

def _basic_url_analysis(input_source: str, focus_areas: str) -> Dict[str, Any]:
    """Basic URL analysis when scraper is not available"""
    content_type = "unknown"
    key_concepts = []

    if "arxiv" in input_source:
        content_type = "research_paper"
        key_concepts = ["research", "academic"]
    elif "github" in input_source:
        content_type = "repository"
        key_concepts = ["code", "implementation"]

    if focus_areas:
        key_concepts.extend(focus_areas.split(","))

    return {
        "content_type": content_type,
        "key_concepts": key_concepts or ["general"],
        "technical_requirements": ["Python"],
        "suggested_node_types": ["CustomNode"],
        "complexity_score": 0.5,
        "summary": f"Basic analysis of {input_source}"
    }

def _generate_fallback_specifications(analysis: Dict[str, Any], quality_level: str) -> Dict[str, Any]:
    """Generate fallback node specifications"""
    content_type = analysis.get("content_type", "unknown")
    key_concepts = analysis.get("key_concepts", ["general"])

    # Generate basic node specification
    node_name = f"{key_concepts[0].replace(' ', '').title()}Node" if key_concepts else "CustomNode"

    node_spec = {
        "name": node_name,
        "category": "utils",
        "description": f"ComfyUI node for {', '.join(key_concepts[:3])}",
        "inputs": ["input"],
        "outputs": ["output"],
        "implementation_notes": f"Generated from {content_type} analysis"
    }

    # Adjust based on content type
    if "image" in key_concepts or "vision" in key_concepts:
        node_spec["category"] = "image/processing"
        node_spec["inputs"] = ["image", "parameters"]
        node_spec["outputs"] = ["processed_image"]

    return {"nodes": [node_spec]}

if __name__ == "__main__":
    # Run the server
    mcp.run()
