#!/bin/bash

# ComfyUI MCP Server Startup Script

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting ComfyUI MCP Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if configuration exists
if [ ! -f "mcp-config.json" ]; then
    echo "❌ mcp-config.json not found. Please run ./install.sh first"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR:-$SCRIPT_DIR/output}"
export DEFAULT_QUALITY="${DEFAULT_QUALITY:-production}"
export STORAGE_PATH="${STORAGE_PATH:-$SCRIPT_DIR/data}"
export MAX_CONCURRENT_EXECUTIONS="${MAX_CONCURRENT_EXECUTIONS:-3}"
export EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-3600}"

# Create output and logs directories if they don't exist
mkdir -p output logs data

echo "📁 Working directory: $SCRIPT_DIR"
echo "📁 Output directory: $DEFAULT_OUTPUT_DIR"
echo "📁 Storage path: $STORAGE_PATH"
echo "🔧 Quality level: $DEFAULT_QUALITY"

# Start the MCP server
echo "🎯 Starting MCP server..."
python src/comfyui_node_mcp_server.py

echo "✅ MCP server started successfully!"
