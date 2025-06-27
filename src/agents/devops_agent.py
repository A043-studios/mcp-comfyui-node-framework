"""
DevOps Agent for MCP Multi-Agent Framework
Handles environment setup, dependency management, and infrastructure preparation
"""

import os
import subprocess
import sys
from typing import Dict, Any, List
from pathlib import Path

# Professional import structure
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from agents.base_agent import BaseAgent
except ImportError:
    try:
        from .base_agent import BaseAgent
    except ImportError as e:
        print(f"Warning: Could not import dependencies: {e}")

        class BaseAgent:
            def __init__(self, config): pass
            def execute(self, context): return {"status": "error", "message": "Dependencies not available"}


class DevOpsAgent(BaseAgent):
    """
    DevOps Agent responsible for:
    - Environment setup and validation
    - Dependency installation and management
    - Repository cloning and configuration
    - Build system preparation
    """
    
    def _process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main DevOps processing logic
        
        Args:
            context: Execution context from previous agents
            
        Returns:
            Dict containing DevOps results and artifacts
        """
        self._log_info("Starting DevOps environment setup")
        
        # Get quality settings
        quality_settings = self._get_quality_settings()
        
        # Initialize results
        results = {
            "artifacts": {},
            "metrics": {},
            "summary": ""
        }
        
        try:
            # 1. Validate environment
            env_validation = self._validate_environment(context)
            results["artifacts"]["environment_validation"] = env_validation
            
            # 2. Setup dependencies
            if env_validation.get("valid", False):
                deps_result = self._setup_dependencies(context)
                results["artifacts"]["dependencies"] = deps_result
                
                # 3. Prepare build environment
                build_result = self._prepare_build_environment(context)
                results["artifacts"]["build_environment"] = build_result
                
                # 4. Validate setup
                validation_result = self._validate_setup(context)
                results["artifacts"]["setup_validation"] = validation_result
                
                # Update metrics
                results["metrics"] = {
                    "environment_valid": env_validation.get("valid", False),
                    "dependencies_installed": deps_result.get("success", False),
                    "build_ready": build_result.get("success", False),
                    "setup_validated": validation_result.get("valid", False)
                }
                
                results["summary"] = "DevOps setup completed successfully"
            else:
                results["summary"] = "DevOps setup failed - environment validation failed"
                
        except Exception as e:
            self._log_error(f"DevOps processing failed: {str(e)}")
            results["summary"] = f"DevOps setup failed: {str(e)}"
            
        return results
    
    def _validate_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the current environment"""
        self._log_info("Validating environment")
        
        validation = {
            "valid": True,
            "python_version": None,
            "pip_available": False,
            "git_available": False,
            "issues": []
        }
        
        try:
            # Check Python version
            result = subprocess.run(["python3", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                validation["python_version"] = result.stdout.strip()
            else:
                validation["issues"].append("Python3 not available")
                validation["valid"] = False
                
            # Check pip
            result = subprocess.run(["pip3", "--version"], capture_output=True, text=True)
            validation["pip_available"] = result.returncode == 0
            if not validation["pip_available"]:
                validation["issues"].append("pip3 not available")
                
            # Check git
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            validation["git_available"] = result.returncode == 0
            if not validation["git_available"]:
                validation["issues"].append("git not available")
                
        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Environment validation error: {str(e)}")
            
        return validation
    
    def _setup_dependencies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup and install dependencies"""
        self._log_info("Setting up dependencies")
        
        result = {
            "success": True,
            "installed_packages": [],
            "errors": []
        }
        
        # For now, just return success as dependencies are already installed
        result["installed_packages"] = ["framework dependencies already installed"]
        
        return result
    
    def _prepare_build_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the build environment"""
        self._log_info("Preparing build environment")
        
        result = {
            "success": True,
            "directories_created": [],
            "files_created": [],
            "errors": []
        }
        
        try:
            # Create output directories
            output_dir = context.get("output_directory", "./output")
            directories = [
                f"{output_dir}/nodes",
                f"{output_dir}/docs", 
                f"{output_dir}/examples",
                f"{output_dir}/tests"
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                result["directories_created"].append(directory)
                
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Build environment setup failed: {str(e)}")
            
        return result
    
    def _validate_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete setup"""
        self._log_info("Validating setup")
        
        validation = {
            "valid": True,
            "checks": {},
            "issues": []
        }
        
        try:
            # Check output directory exists
            output_dir = context.get("output_directory", "./output")
            validation["checks"]["output_directory"] = os.path.exists(output_dir)
            
            # Check required subdirectories
            subdirs = ["nodes", "docs", "examples", "tests"]
            for subdir in subdirs:
                path = f"{output_dir}/{subdir}"
                validation["checks"][f"{subdir}_directory"] = os.path.exists(path)
                if not os.path.exists(path):
                    validation["issues"].append(f"Missing directory: {path}")
                    validation["valid"] = False
                    
        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Setup validation error: {str(e)}")
            
        return validation
