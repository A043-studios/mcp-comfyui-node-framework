# API Reference

This document provides a comprehensive reference for the MCP ComfyUI Node Framework API.

## Core Classes

### ComfyUIMCPServer

The main server class for the MCP ComfyUI Node Framework.

```python
from src.comfyui_mcp_server_v2 import ComfyUIMCPServer

server = ComfyUIMCPServer(config_path="config/mcp-config.json")
```

#### Methods

##### `generate_node(source, quality_level="production", focus_areas="", output_directory=None)`

Generate ComfyUI nodes from a source.

**Parameters:**
- `source` (str): URL to paper, GitHub repo, or local file path
- `quality_level` (str): Generation quality ("draft", "development", "production")
- `focus_areas` (str): Comma-separated focus areas for analysis
- `output_directory` (str, optional): Custom output directory

**Returns:**
- `dict`: Generation result with metadata

**Example:**
```python
result = server.generate_node(
    source="https://github.com/danielgatis/rembg",
    quality_level="production",
    focus_areas="background removal, image segmentation"
)
```

##### `analyze_content(source, analysis_type="comprehensive")`

Analyze content to understand ComfyUI node requirements.

**Parameters:**
- `source` (str): URL or path to content
- `analysis_type` (str): Type of analysis ("quick", "comprehensive", "technical")

**Returns:**
- `dict`: Analysis results

**Example:**
```python
analysis = server.analyze_content(
    source="https://arxiv.org/abs/2301.12345",
    analysis_type="comprehensive"
)
```

## Agent Classes

### ResearchAgent

Handles research paper analysis and content extraction.

```python
from src.agents.research_agent import ResearchAgent

agent = ResearchAgent(config)
```

#### Methods

##### `analyze_paper(url, focus_areas=None)`

Analyze a research paper for ComfyUI node potential.

**Parameters:**
- `url` (str): URL to the research paper
- `focus_areas` (list, optional): Specific areas to focus on

**Returns:**
- `dict`: Analysis results

### CodingAgent

Generates ComfyUI node code from analysis results.

```python
from src.agents.coding_agent import CodingAgent

agent = CodingAgent(config)
```

#### Methods

##### `generate_node_code(analysis, quality_level="production")`

Generate ComfyUI node code from analysis.

**Parameters:**
- `analysis` (dict): Analysis results from ResearchAgent
- `quality_level` (str): Target quality level

**Returns:**
- `dict`: Generated code and metadata

### DocumentationAgent

Creates documentation for generated nodes.

```python
from src.agents.documentation_agent import DocumentationAgent

agent = DocumentationAgent(config)
```

#### Methods

##### `generate_documentation(node_info, examples=None)`

Generate comprehensive documentation for a node.

**Parameters:**
- `node_info` (dict): Node information and code
- `examples` (list, optional): Usage examples

**Returns:**
- `dict`: Generated documentation

## Utility Classes

### LLMClient

Handles communication with language models.

```python
from src.llm_client_v2 import LLMClient

client = LLMClient(config)
```

#### Methods

##### `generate_response(prompt, model=None, temperature=0.1)`

Generate a response using the configured LLM.

**Parameters:**
- `prompt` (str): Input prompt
- `model` (str, optional): Override default model
- `temperature` (float): Response randomness (0.0-1.0)

**Returns:**
- `str`: Generated response

### WebScraper

Extracts content from web sources.

```python
from src.web_scraper_v2 import WebScraper

scraper = WebScraper()
```

#### Methods

##### `scrape_github_repo(url)`

Extract information from a GitHub repository.

**Parameters:**
- `url` (str): GitHub repository URL

**Returns:**
- `dict`: Repository information

##### `scrape_arxiv_paper(url)`

Extract information from an arXiv paper.

**Parameters:**
- `url` (str): arXiv paper URL

**Returns:**
- `dict`: Paper information

## Configuration

### MCP Configuration

The main configuration file structure:

```json
{
  "llm": {
    "provider": "openrouter",
    "api_key": "your-api-key",
    "model": "anthropic/claude-3.5-sonnet",
    "temperature": 0.1,
    "max_tokens": 4000
  },
  "agents": {
    "research_agent": {
      "model": "anthropic/claude-3.5-sonnet",
      "temperature": 0.1
    },
    "coding_agent": {
      "model": "anthropic/claude-3.5-sonnet",
      "temperature": 0.0
    },
    "documentation_agent": {
      "model": "anthropic/claude-3.5-sonnet",
      "temperature": 0.2
    }
  },
  "quality_levels": {
    "draft": {
      "min_score": 0.3,
      "max_iterations": 1
    },
    "development": {
      "min_score": 0.5,
      "max_iterations": 2
    },
    "production": {
      "min_score": 0.7,
      "max_iterations": 3
    }
  },
  "output": {
    "base_directory": "output",
    "organize_by_date": true,
    "include_metadata": true
  }
}
```

### Agent Configuration

Individual agent settings:

```json
{
  "research_agent": {
    "model": "anthropic/claude-3.5-sonnet",
    "temperature": 0.1,
    "max_tokens": 4000,
    "timeout": 30,
    "retry_attempts": 3
  },
  "coding_agent": {
    "model": "anthropic/claude-3.5-sonnet",
    "temperature": 0.0,
    "max_tokens": 8000,
    "timeout": 60,
    "retry_attempts": 2
  },
  "documentation_agent": {
    "model": "anthropic/claude-3.5-sonnet",
    "temperature": 0.2,
    "max_tokens": 6000,
    "timeout": 45,
    "retry_attempts": 2
  }
}
```

## Error Handling

### Common Exceptions

#### `MCPServerError`

Base exception for MCP server errors.

#### `LLMClientError`

Raised when LLM communication fails.

#### `ContentAnalysisError`

Raised when content analysis fails.

#### `NodeGenerationError`

Raised when node generation fails.

### Example Error Handling

```python
from src.exceptions import MCPServerError, LLMClientError

try:
    result = server.generate_node(source="invalid-url")
except LLMClientError as e:
    print(f"LLM error: {e}")
except MCPServerError as e:
    print(f"Server error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Response Formats

### Generation Result

```python
{
    "execution_id": "uuid-string",
    "success": True,
    "nodes_generated": 2,
    "output_directory": "/path/to/output",
    "quality_score": 0.8,
    "analysis_summary": "Description of generated nodes",
    "recommendations": ["List of recommendations"],
    "artifacts": {
        "nodes": ["node1.py", "node2.py"],
        "tests": ["test_node1.py"],
        "documentation": ["README.md", "API.md"]
    }
}
```

### Analysis Result

```python
{
    "content_type": "github_repository",
    "key_concepts": ["concept1", "concept2"],
    "technical_requirements": ["python", "torch"],
    "suggested_node_types": ["NodeType1", "NodeType2"],
    "complexity_score": 0.6,
    "categories": ["image", "preprocessing"],
    "summary": "Analysis summary"
}
```

## Best Practices

### Performance Optimization

1. **Use session caching** for repeated operations
2. **Configure appropriate timeouts** for LLM calls
3. **Batch process** multiple sources when possible
4. **Monitor quality scores** and adjust parameters

### Quality Control

1. **Start with draft quality** for experimentation
2. **Use production quality** for final nodes
3. **Specify focus areas** for better results
4. **Review generated code** before deployment

### Configuration Management

1. **Use environment variables** for API keys
2. **Version control** configuration templates
3. **Document custom settings** for team use
4. **Test configurations** before deployment

---

For more information, see the [main documentation](../README.md) or [configuration guide](CONFIGURATION.md).
