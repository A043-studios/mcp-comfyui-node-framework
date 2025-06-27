#!/bin/bash

# ComfyUI MCP Framework Installation Script
# This script sets up the ComfyUI MCP framework in a new environment

set -e

echo "🚀 Installing ComfyUI MCP Framework..."

# Get the installation directory
INSTALL_DIR="$(pwd)"
echo "📁 Installation directory: $INSTALL_DIR"

# Check if Python 3.11+ is available
echo "🐍 Checking Python version..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
        PYTHON_CMD="python3"
    else
        echo "❌ Python 3.11+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "❌ Python 3.11+ is required but not found"
    exit 1
fi

echo "✅ Using Python: $PYTHON_CMD"

# Create virtual environment
echo "🔧 Creating virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing core dependencies..."
    pip install mcp anthropic openai requests beautifulsoup4 lxml trafilatura newspaper3k selenium
fi

# Install Node.js dependencies if package.json exists
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    if command -v npm &> /dev/null; then
        npm install
    elif command -v yarn &> /dev/null; then
        yarn install
    else
        echo "⚠️  Node.js package manager not found, skipping Node.js dependencies"
    fi
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p output logs data

# Set up configuration
echo "⚙️  Setting up configuration..."
if [ ! -f "mcp-config.json" ]; then
    cp mcp-config.template.json mcp-config.json
    echo "📝 Created mcp-config.json from template"
    echo "⚠️  Please edit mcp-config.json with your API keys and paths"
fi

# Make scripts executable
chmod +x start-server.sh
chmod +x setup-augment.sh

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit mcp-config.json with your API keys and correct paths"
echo "2. Run './setup-augment.sh' to configure Augment integration"
echo "3. Run './start-server.sh' to start the MCP server"
echo ""
echo "📚 Documentation:"
echo "- README.md - General usage and setup"
echo "- INSTALLATION.md - Detailed installation guide"
echo "- config/agents.json - Agent configuration"
echo ""
echo "🔑 Required API Keys:"
echo "- OPENROUTER_API_KEY (recommended)"
echo "- ANTHROPIC_API_KEY (optional)"
echo "- OPENAI_API_KEY (optional)"
