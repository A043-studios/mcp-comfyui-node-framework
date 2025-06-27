# MCP ComfyUI Node Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)

**An advanced Model Context Protocol (MCP) server framework for automatically generating ComfyUI nodes from research papers, GitHub repositories, and other sources.**

## 🚀 Overview

The MCP ComfyUI Node Framework is a powerful tool that leverages AI to automatically analyze and generate ComfyUI nodes from various sources including:

- **Research Papers** (arXiv, academic publications)
- **GitHub Repositories** (open source projects)
- **Documentation** (API docs, technical specifications)
- **Code Examples** (implementation references)

### Key Features

- 🤖 **AI-Powered Analysis**: Intelligent content analysis using advanced LLM integration
- 🔧 **Automatic Node Generation**: Creates production-ready ComfyUI nodes with proper typing
- 📚 **Multi-Source Support**: Works with papers, repositories, and documentation
- 🎯 **Quality Control**: Multiple quality levels (draft, development, production)
- 🔄 **Batch Processing**: Generate multiple nodes efficiently
- 📖 **Comprehensive Documentation**: Auto-generates README, installation guides, and examples
- 🧪 **Testing Framework**: Includes test generation and validation
- 🔌 **MCP Integration**: Full Model Context Protocol compatibility

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher
- ComfyUI installation (for testing generated nodes)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/A043-studios/mcp-comfyui-node-framework.git
cd mcp-comfyui-node-framework

# Run the installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

### Manual Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Set up environment
chmod +x scripts/setup-env.sh
./scripts/setup-env.sh
```

## ⚡ Quick Start

### 1. Configure the Framework

```bash
# Copy the template configuration
cp config/mcp-config.template.json config/mcp-config.json

# Edit configuration with your API keys
nano config/mcp-config.json
```

### 2. Start the MCP Server

```bash
# Start the server
chmod +x scripts/start-server.sh
./scripts/start-server.sh
```

### 3. Generate Your First Node

```python
from src.comfyui_mcp_server_v2 import ComfyUIMCPServer

# Initialize the server
server = ComfyUIMCPServer()

# Generate a node from a GitHub repository
result = server.generate_node(
    source="https://github.com/danielgatis/rembg",
    quality_level="production",
    focus_areas="background removal, image segmentation"
)

print(f"Generated {result['nodes_generated']} nodes!")
```

## 📋 Usage Examples

### Generate from Research Paper

```python
# Generate nodes from an arXiv paper
result = server.generate_node(
    source="https://arxiv.org/abs/2301.12345",
    quality_level="production",
    focus_areas="image processing, neural networks"
)
```

### Generate from GitHub Repository

```python
# Generate nodes from a repository
result = server.generate_node(
    source="https://github.com/user/awesome-project",
    quality_level="development",
    focus_areas="computer vision, preprocessing"
)
```

### Batch Generation

```python
# Generate multiple nodes
sources = [
    "https://github.com/danielgatis/rembg",
    "https://arxiv.org/abs/2301.12345",
    "https://github.com/another/project"
]

for source in sources:
    result = server.generate_node(source, quality_level="production")
    print(f"Generated from {source}: {result['nodes_generated']} nodes")
```

## 🎯 Generated Node Example

The framework generates complete, production-ready ComfyUI nodes. Here's an example of what gets created:

```python
class RembgBackgroundRemovalNode:
    """ComfyUI Node for AI-powered background removal using rembg"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["u2net", "birefnet-general", "isnet-anime"], 
                         {"default": "u2net"}),
                "return_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image/background"
    
    # ... implementation
```

## 📁 Project Structure

```
mcp-comfyui-node-framework/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── INSTALLATION.md             # Detailed installation guide
├── requirements.txt            # Python dependencies
├── package.json                # Node.js configuration
├── src/                        # Core framework source
│   ├── comfyui_mcp_server_v2.py    # Main MCP server
│   ├── agents/                     # AI agent implementations
│   ├── tools/                      # Generation tools
│   └── utils/                      # Utility functions
├── config/                     # Configuration files
│   ├── mcp-config.template.json   # Configuration template
│   └── agents.example.json        # Agent configuration
├── scripts/                    # Installation and setup scripts
├── examples/                   # Example nodes and workflows
│   ├── nodes/                      # Generated node examples
│   └── workflows/                  # Example ComfyUI workflows
├── docs/                       # Documentation
└── tests/                      # Test suite
```

## 🔧 Configuration

### MCP Configuration

Edit `config/mcp-config.json` to configure:

- **LLM API Keys**: OpenRouter, OpenAI, etc.
- **Agent Settings**: Research, coding, documentation agents
- **Quality Levels**: Draft, development, production settings
- **Output Preferences**: File organization, naming conventions

### Agent Configuration

Customize agent behavior in `config/agents.example.json`:

```json
{
  "research_agent": {
    "model": "anthropic/claude-3.5-sonnet",
    "temperature": 0.1,
    "max_tokens": 4000
  },
  "coding_agent": {
    "model": "anthropic/claude-3.5-sonnet", 
    "temperature": 0.0,
    "max_tokens": 8000
  }
}
```

## 🧪 Testing Generated Nodes

```bash
# Test a generated node
python tests/test_integration.py --node examples/nodes/rembg_background_removal_node.py

# Run full test suite
python -m pytest tests/
```

## 📚 Documentation

- [Installation Guide](INSTALLATION.md) - Detailed setup instructions
- [Configuration Guide](docs/CONFIGURATION.md) - Complete configuration reference
- [API Documentation](docs/API.md) - Framework API reference
- [Examples](examples/README.md) - Usage examples and tutorials

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [Model Context Protocol](https://modelcontextprotocol.io/)
- Designed for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Powered by advanced language models

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/A043-studios/mcp-comfyui-node-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/A043-studios/mcp-comfyui-node-framework/discussions)
- **Documentation**: [Wiki](https://github.com/A043-studios/mcp-comfyui-node-framework/wiki)

---

**Made with ❤️ for the ComfyUI community**
