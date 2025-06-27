#!/usr/bin/env python3
"""
Optimized Dependency Manager for ComfyUI Framework
Implements lazy loading, caching, and optional dependencies
"""

import logging
import importlib
import sys
import os
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import threading
from functools import wraps


class DependencyType(Enum):
    """Types of dependencies"""
    CORE = "core"
    OPTIONAL = "optional"
    HEAVY = "heavy"
    DEVELOPMENT = "development"


@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    import_name: str
    dependency_type: DependencyType
    description: str
    install_command: str
    fallback_available: bool
    lazy_load: bool
    cache_duration: int = 3600  # 1 hour default


class DependencyManager:
    """
    Optimized dependency manager with lazy loading and caching
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Dependency cache
        self._module_cache = {}
        self._availability_cache = {}
        self._cache_timestamps = {}
        self._lock = threading.Lock()
        
        # Dependency registry
        self.dependencies = {}
        self._initialize_dependency_registry()
        
        # Feature flags
        self.enable_lazy_loading = config.get("enable_lazy_loading", True)
        self.enable_caching = config.get("enable_caching", True)
        self.cache_duration = config.get("cache_duration", 3600)
        
        self.logger.info("Dependency manager initialized with lazy loading and caching")
    
    def _initialize_dependency_registry(self):
        """Initialize the dependency registry with all known dependencies"""
        
        # Core dependencies (always required)
        core_deps = [
            DependencyInfo("requests", "requests", DependencyType.CORE, 
                         "HTTP library", "pip install requests", False, False),
            DependencyInfo("pathlib", "pathlib", DependencyType.CORE,
                         "Path handling", "built-in", False, False),
            DependencyInfo("json", "json", DependencyType.CORE,
                         "JSON handling", "built-in", False, False),
        ]
        
        # Optional lightweight dependencies
        optional_deps = [
            DependencyInfo("beautifulsoup4", "bs4", DependencyType.OPTIONAL,
                         "HTML parsing", "pip install beautifulsoup4", True, True),
            DependencyInfo("lxml", "lxml", DependencyType.OPTIONAL,
                         "XML parsing", "pip install lxml", True, True),
            DependencyInfo("markdown", "markdown", DependencyType.OPTIONAL,
                         "Markdown processing", "pip install markdown", True, True),
        ]
        
        # Heavy dependencies (loaded on demand)
        heavy_deps = [
            DependencyInfo("torch", "torch", DependencyType.HEAVY,
                         "PyTorch deep learning", "pip install torch", True, True, 7200),
            DependencyInfo("transformers", "transformers", DependencyType.HEAVY,
                         "Hugging Face transformers", "pip install transformers", True, True, 7200),
            DependencyInfo("opencv-python", "cv2", DependencyType.HEAVY,
                         "Computer vision", "pip install opencv-python", True, True, 7200),
            DependencyInfo("librosa", "librosa", DependencyType.HEAVY,
                         "Audio processing", "pip install librosa", True, True, 7200),
            DependencyInfo("selenium", "selenium", DependencyType.HEAVY,
                         "Web automation", "pip install selenium", True, True, 3600),
            DependencyInfo("numpy", "numpy", DependencyType.HEAVY,
                         "Numerical computing", "pip install numpy", False, True, 7200),
            DependencyInfo("pandas", "pandas", DependencyType.HEAVY,
                         "Data analysis", "pip install pandas", True, True, 7200),
        ]
        
        # Development dependencies
        dev_deps = [
            DependencyInfo("pytest", "pytest", DependencyType.DEVELOPMENT,
                         "Testing framework", "pip install pytest", True, True),
            DependencyInfo("black", "black", DependencyType.DEVELOPMENT,
                         "Code formatting", "pip install black", True, True),
            DependencyInfo("flake8", "flake8", DependencyType.DEVELOPMENT,
                         "Code linting", "pip install flake8", True, True),
        ]
        
        # Register all dependencies
        all_deps = core_deps + optional_deps + heavy_deps + dev_deps
        for dep in all_deps:
            self.dependencies[dep.name] = dep
        
        self.logger.info(f"Registered {len(all_deps)} dependencies")
    
    def is_available(self, dependency_name: str, force_check: bool = False) -> bool:
        """Check if a dependency is available with caching"""
        
        if not self.enable_caching or force_check:
            return self._check_availability_direct(dependency_name)
        
        with self._lock:
            # Check cache
            if dependency_name in self._availability_cache:
                cache_time = self._cache_timestamps.get(dependency_name, 0)
                if time.time() - cache_time < self.cache_duration:
                    return self._availability_cache[dependency_name]
            
            # Check availability and cache result
            available = self._check_availability_direct(dependency_name)
            self._availability_cache[dependency_name] = available
            self._cache_timestamps[dependency_name] = time.time()
            
            return available
    
    def _check_availability_direct(self, dependency_name: str) -> bool:
        """Direct availability check without caching"""
        
        if dependency_name not in self.dependencies:
            self.logger.warning(f"Unknown dependency: {dependency_name}")
            return False
        
        dep_info = self.dependencies[dependency_name]
        
        try:
            importlib.import_module(dep_info.import_name)
            return True
        except ImportError:
            return False
    
    def import_dependency(self, dependency_name: str, required: bool = False) -> Optional[Any]:
        """Import a dependency with lazy loading and caching"""
        
        if dependency_name not in self.dependencies:
            if required:
                raise ImportError(f"Unknown required dependency: {dependency_name}")
            return None
        
        dep_info = self.dependencies[dependency_name]
        
        # Check if lazy loading is enabled and appropriate
        if self.enable_lazy_loading and dep_info.lazy_load:
            return self._lazy_import(dep_info, required)
        else:
            return self._direct_import(dep_info, required)
    
    def _lazy_import(self, dep_info: DependencyInfo, required: bool) -> Optional[Any]:
        """Lazy import with caching"""
        
        with self._lock:
            # Check cache first
            if dep_info.name in self._module_cache:
                cache_time = self._cache_timestamps.get(f"module_{dep_info.name}", 0)
                if time.time() - cache_time < dep_info.cache_duration:
                    return self._module_cache[dep_info.name]
            
            # Import and cache
            module = self._direct_import(dep_info, required)
            if module is not None:
                self._module_cache[dep_info.name] = module
                self._cache_timestamps[f"module_{dep_info.name}"] = time.time()
            
            return module
    
    def _direct_import(self, dep_info: DependencyInfo, required: bool) -> Optional[Any]:
        """Direct import without caching"""
        
        try:
            module = importlib.import_module(dep_info.import_name)
            self.logger.debug(f"Successfully imported {dep_info.name}")
            return module
        except ImportError as e:
            if required and not dep_info.fallback_available:
                self.logger.error(f"Required dependency {dep_info.name} not available: {e}")
                raise ImportError(f"Required dependency {dep_info.name} not available. Install with: {dep_info.install_command}")
            elif not dep_info.fallback_available:
                self.logger.warning(f"Optional dependency {dep_info.name} not available: {e}")
            else:
                self.logger.info(f"Dependency {dep_info.name} not available, fallback will be used")
            return None
    
    def require_dependencies(self, dependency_names: List[str]) -> Dict[str, Any]:
        """Import multiple required dependencies"""
        
        modules = {}
        missing = []
        
        for dep_name in dependency_names:
            try:
                module = self.import_dependency(dep_name, required=True)
                modules[dep_name] = module
            except ImportError:
                missing.append(dep_name)
        
        if missing:
            missing_info = []
            for dep_name in missing:
                if dep_name in self.dependencies:
                    dep_info = self.dependencies[dep_name]
                    missing_info.append(f"{dep_name}: {dep_info.install_command}")
                else:
                    missing_info.append(f"{dep_name}: unknown dependency")
            
            raise ImportError(f"Missing required dependencies:\n" + "\n".join(missing_info))
        
        return modules
    
    def get_optional_dependencies(self, dependency_names: List[str]) -> Dict[str, Any]:
        """Import multiple optional dependencies"""
        
        modules = {}
        
        for dep_name in dependency_names:
            module = self.import_dependency(dep_name, required=False)
            if module is not None:
                modules[dep_name] = module
        
        return modules
    
    def lazy_import_decorator(self, dependency_name: str, required: bool = False):
        """Decorator for lazy importing dependencies in functions"""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Import dependency when function is called
                module = self.import_dependency(dependency_name, required)
                
                # Add module to function's globals if successful
                if module is not None:
                    func.__globals__[dependency_name] = module
                elif required:
                    raise ImportError(f"Required dependency {dependency_name} not available")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Get status of all dependencies"""
        
        status = {
            "core": {"available": [], "missing": []},
            "optional": {"available": [], "missing": []},
            "heavy": {"available": [], "missing": []},
            "development": {"available": [], "missing": []}
        }
        
        for dep_name, dep_info in self.dependencies.items():
            category = dep_info.dependency_type.value
            if self.is_available(dep_name):
                status[category]["available"].append(dep_name)
            else:
                status[category]["missing"].append({
                    "name": dep_name,
                    "install_command": dep_info.install_command,
                    "description": dep_info.description
                })
        
        return status
    
    def install_missing_dependencies(self, dependency_types: List[DependencyType] = None) -> Dict[str, Any]:
        """Generate installation commands for missing dependencies"""
        
        if dependency_types is None:
            dependency_types = [DependencyType.CORE, DependencyType.OPTIONAL]
        
        missing_deps = []
        
        for dep_name, dep_info in self.dependencies.items():
            if dep_info.dependency_type in dependency_types and not self.is_available(dep_name):
                missing_deps.append(dep_info)
        
        if not missing_deps:
            return {"status": "all_available", "commands": []}
        
        # Group by install command to optimize installation
        install_groups = {}
        for dep in missing_deps:
            cmd = dep.install_command
            if cmd not in install_groups:
                install_groups[cmd] = []
            install_groups[cmd].append(dep.name)
        
        return {
            "status": "missing_dependencies",
            "missing_count": len(missing_deps),
            "commands": list(install_groups.keys()),
            "dependencies": [{"name": dep.name, "description": dep.description} for dep in missing_deps]
        }
    
    def clear_cache(self):
        """Clear all caches"""
        with self._lock:
            self._module_cache.clear()
            self._availability_cache.clear()
            self._cache_timestamps.clear()
        
        self.logger.info("Dependency cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "module_cache_size": len(self._module_cache),
                "availability_cache_size": len(self._availability_cache),
                "cache_hits": sum(1 for ts in self._cache_timestamps.values() 
                                if time.time() - ts < self.cache_duration),
                "cache_enabled": self.enable_caching,
                "lazy_loading_enabled": self.enable_lazy_loading
            }


# Global dependency manager instance
_dependency_manager = None


def get_dependency_manager(config: Dict[str, Any] = None) -> DependencyManager:
    """Get global dependency manager instance"""
    global _dependency_manager
    
    if _dependency_manager is None:
        if config is None:
            config = {}
        _dependency_manager = DependencyManager(config)
    
    return _dependency_manager


def lazy_import(dependency_name: str, required: bool = False):
    """Convenience function for lazy importing"""
    manager = get_dependency_manager()
    return manager.lazy_import_decorator(dependency_name, required)


def require(*dependency_names: str) -> Dict[str, Any]:
    """Convenience function for requiring multiple dependencies"""
    manager = get_dependency_manager()
    return manager.require_dependencies(list(dependency_names))


def optional(*dependency_names: str) -> Dict[str, Any]:
    """Convenience function for optional dependencies"""
    manager = get_dependency_manager()
    return manager.get_optional_dependencies(list(dependency_names))


def is_available(dependency_name: str) -> bool:
    """Convenience function to check dependency availability"""
    manager = get_dependency_manager()
    return manager.is_available(dependency_name)


# Example usage decorators
def requires_torch(func):
    """Decorator that ensures torch is available"""
    return lazy_import("torch", required=True)(func)


def requires_transformers(func):
    """Decorator that ensures transformers is available"""
    return lazy_import("transformers", required=True)(func)


def requires_opencv(func):
    """Decorator that ensures opencv is available"""
    return lazy_import("opencv-python", required=True)(func)


def optional_selenium(func):
    """Decorator that optionally imports selenium"""
    return lazy_import("selenium", required=False)(func)


# Context manager for temporary dependency requirements
class DependencyContext:
    """Context manager for temporary dependency requirements"""
    
    def __init__(self, *dependency_names: str, required: bool = True):
        self.dependency_names = dependency_names
        self.required = required
        self.modules = {}
        self.manager = get_dependency_manager()
    
    def __enter__(self):
        if self.required:
            self.modules = self.manager.require_dependencies(list(self.dependency_names))
        else:
            self.modules = self.manager.get_optional_dependencies(list(self.dependency_names))
        return self.modules
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass


# Example usage:
# with DependencyContext("torch", "transformers") as deps:
#     torch = deps["torch"]
#     transformers = deps["transformers"]
#     # Use the dependencies
