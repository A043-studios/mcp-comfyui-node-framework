# ComfyUI MCP Framework - Installation Guide

## 📋 Prerequisites

### System Requirements:
- **Python 3.11+** (required)
- **Node.js 16+** (optional, for TypeScript tools)
- **Git** (for cloning repositories)
- **Internet connection** (for API access)

### API Keys:
You'll need at least one of these API keys:
- **OpenRouter API Key** (recommended) - `sk-or-v1-...`
- **Anthropic API Key** (optional) - `sk-ant-...`
- **OpenAI API Key** (optional) - `sk-...`

## 🚀 Installation Steps

### 1. Extract and Navigate
```bash
# Extract the package
unzip comfyui-mcp-package.zip
# or
tar -xzf comfyui-mcp-package.tar.gz

# Navigate to the directory
cd comfyui-mcp-package
```

### 2. Run Installation Script
```bash
# Make the script executable
chmod +x install.sh

# Run the installation
./install.sh
```

The installation script will:
- ✅ Check Python version (3.11+ required)
- ✅ Create virtual environment
- ✅ Install Python dependencies
- ✅ Install Node.js dependencies (if available)
- ✅ Create directory structure
- ✅ Set up configuration templates

### 3. Configure API Keys

#### Option A: Edit Configuration File
```bash
# Edit the main configuration
nano mcp-config.json

# Update these fields:
"OPENROUTER_API_KEY": "your-actual-api-key-here"
"ANTHROPIC_API_KEY": "your-anthropic-key-here"  # optional
"OPENAI_API_KEY": "your-openai-key-here"        # optional
```

#### Option B: Use Environment Variables
```bash
# Edit the environment setup script
nano setup-env.sh

# Set your API keys
export OPENROUTER_API_KEY="your-actual-api-key-here"

# Source the environment
source setup-env.sh
```

### 4. Setup Augment Integration (Optional)
```bash
# Configure for Augment
./setup-augment.sh
```

This creates:
- `augment-mcp-config.json` - Augment-specific configuration
- Updated paths and settings for Augment integration

### 5. Test Installation
```bash
# Start the server
./start-server.sh
```

You should see:
```
🚀 Starting ComfyUI MCP Server...
🔧 Activating virtual environment...
📁 Working directory: /path/to/installation
✅ MCP server started successfully!
```

## 🔧 Configuration Details

### MCP Configuration (`mcp-config.json`)
```json
{
  "mcpServers": {
    "comfyui-framework": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/src/comfyui_node_mcp_server.py"],
      "cwd": "/path/to/installation",
      "env": {
        "PYTHONPATH": "/path/to/installation",
        "OPENROUTER_API_KEY": "your-key-here",
        "LOG_LEVEL": "info",
        "DEFAULT_OUTPUT_DIR": "/path/to/output",
        "DEFAULT_QUALITY": "production"
      }
    }
  }
}
```

### Agent Configuration (`config/agents.json`)
The framework includes pre-configured agents:
- Research Agent (Claude 3.5 Sonnet)
- Coding Agent (Claude 3.5 Sonnet)
- Testing Agent (Claude 3.5 Sonnet)
- Documentation Agent (Claude 3.5 Sonnet)
- DevOps Agent (Claude 3 Haiku)

## 🔍 Troubleshooting

### Common Issues:

#### 1. Python Version Error
```bash
❌ Python 3.11+ is required. Found: 3.9.0
```
**Solution**: Install Python 3.11+ or use pyenv:
```bash
# Using pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

#### 2. Virtual Environment Issues
```bash
❌ Virtual environment not found
```
**Solution**: Re-run installation:
```bash
rm -rf venv
./install.sh
```

#### 3. API Key Issues
```bash
❌ API key validation failed
```
**Solution**: Check your API key format and permissions:
- OpenRouter keys start with `sk-or-v1-`
- Anthropic keys start with `sk-ant-`
- OpenAI keys start with `sk-`

#### 4. Permission Issues
```bash
❌ Permission denied
```
**Solution**: Make scripts executable:
```bash
chmod +x install.sh start-server.sh setup-augment.sh
```

#### 5. Import Errors
```bash
❌ ModuleNotFoundError: No module named 'mcp'
```
**Solution**: Activate virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## 📁 Directory Structure After Installation

```
comfyui-mcp-package/
├── venv/                  # Virtual environment
├── src/                   # Framework source code
├── config/                # Configuration files
├── output/                # Generated nodes output
├── logs/                  # Execution logs
├── data/                  # Storage for artifacts
├── mcp-config.json        # MCP server configuration
├── augment-mcp-config.json # Augment integration config
├── setup-env.sh           # Environment variables
└── *.sh                   # Installation/startup scripts
```

## 🔄 Updating

To update the framework:
1. **Backup configurations**:
   ```bash
   cp mcp-config.json mcp-config.backup.json
   cp config/agents.json config/agents.backup.json
   ```

2. **Replace framework files**:
   ```bash
   # Extract new version over existing installation
   # (keeping your config files)
   ```

3. **Update dependencies**:
   ```bash
   ./install.sh
   ```

4. **Restore configurations** if needed

## 🎯 Next Steps

After successful installation:

1. **Test the framework**:
   ```bash
   # Generate a simple node
   # Use MCP client to call generate_comfyui_node tool
   ```

2. **Explore examples**:
   - Check `output/` for generated nodes
   - Review `logs/` for execution details

3. **Customize configuration**:
   - Modify agent settings in `config/agents.json`
   - Adjust quality levels and timeouts

4. **Integrate with your workflow**:
   - Use with Augment or other MCP clients
   - Create custom node generation pipelines

---

**Installation complete! Ready to generate ComfyUI nodes!** 🎨
