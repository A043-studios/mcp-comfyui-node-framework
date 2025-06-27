#!/usr/bin/env python3
"""
Performance Optimizer for ComfyUI Framework
Implements caching, memory management, and concurrent processing optimizations
"""

import logging
import asyncio
import time
import gc
import threading

# Optional dependencies with fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Mock psutil for basic functionality
    class MockPsutil:
        @staticmethod
        def Process():
            return MockProcess()
        @staticmethod
        def virtual_memory():
            return MockMemory()
        @staticmethod
        def cpu_percent():
            return 0.0

    class MockProcess:
        def memory_info(self):
            return MockMemoryInfo()
        def memory_percent(self):
            return 0.0

    class MockMemoryInfo:
        rss = 0

    class MockMemory:
        percent = 0.0
        available = 0

    psutil = MockPsutil()
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps, lru_cache
import hashlib
import json
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class CacheType(Enum):
    """Types of caching strategies"""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int
    cache_misses: int
    concurrent_tasks: int
    optimization_level: OptimizationLevel


class AdvancedCache:
    """Advanced caching system with multiple strategies"""
    
    def __init__(self, cache_type: CacheType = CacheType.HYBRID, max_size: int = 1000,
                 ttl: int = 3600, cache_dir: str = None):
        self.cache_type = cache_type
        self.max_size = max_size
        self.ttl = ttl
        # Set default cache directory to a safe location
        if cache_dir is None:
            import tempfile
            self.cache_dir = os.path.join(tempfile.gettempdir(), "comfyui_mcp_cache")
        else:
            self.cache_dir = cache_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Memory cache
        self._memory_cache = {}
        self._cache_timestamps = {}
        self._access_counts = {}
        self._lock = threading.RLock()
        
        # Disk cache setup
        if cache_type in [CacheType.DISK, CacheType.HYBRID]:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                self.logger.warning(f"Failed to create cache directory {self.cache_dir}: {e}. Disabling disk cache.")
                # Fall back to memory-only cache
                self.cache_type = CacheType.MEMORY
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                timestamp = self._cache_timestamps.get(key, 0)
                if time.time() - timestamp < self.ttl:
                    self._access_counts[key] = self._access_counts.get(key, 0) + 1
                    self.hits += 1
                    return self._memory_cache[key]
                else:
                    # Expired
                    self._remove_from_memory(key)
            
            # Check disk cache for hybrid/disk types
            if self.cache_type in [CacheType.DISK, CacheType.HYBRID]:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    # Load into memory cache if hybrid
                    if self.cache_type == CacheType.HYBRID:
                        self._store_in_memory(key, disk_value)
                    self.hits += 1
                    return disk_value
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        with self._lock:
            # Store in memory
            if self.cache_type in [CacheType.MEMORY, CacheType.HYBRID]:
                self._store_in_memory(key, value)
            
            # Store on disk
            if self.cache_type in [CacheType.DISK, CacheType.HYBRID]:
                self._store_on_disk(key, value)
    
    def _store_in_memory(self, key: str, value: Any) -> None:
        """Store item in memory cache"""
        # Evict if at capacity
        if len(self._memory_cache) >= self.max_size:
            self._evict_lru()
        
        self._memory_cache[key] = value
        self._cache_timestamps[key] = time.time()
        self._access_counts[key] = 1
    
    def _store_on_disk(self, key: str, value: Any) -> None:
        """Store item on disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            self.logger.warning(f"Failed to store cache item on disk: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get item from disk cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if time.time() - data['timestamp'] < self.ttl:
                        return data['value']
                    else:
                        # Remove expired file
                        os.remove(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to load cache item from disk: {e}")
        return None
    
    def _remove_from_memory(self, key: str) -> None:
        """Remove item from memory cache"""
        self._memory_cache.pop(key, None)
        self._cache_timestamps.pop(key, None)
        self._access_counts.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._access_counts:
            return
        
        lru_key = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
        self._remove_from_memory(lru_key)
    
    def clear(self) -> None:
        """Clear all cache"""
        with self._lock:
            self._memory_cache.clear()
            self._cache_timestamps.clear()
            self._access_counts.clear()
            
            # Clear disk cache
            if self.cache_type in [CacheType.DISK, CacheType.HYBRID]:
                try:
                    for file in os.listdir(self.cache_dir):
                        if file.endswith('.pkl'):
                            os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    self.logger.warning(f"Failed to clear disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "memory_items": len(self._memory_cache),
            "cache_type": self.cache_type.value,
            "max_size": self.max_size,
            "ttl": self.ttl
        }


class MemoryManager:
    """Advanced memory management and monitoring"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_threshold = self._get_memory_threshold()
        
    def _get_memory_threshold(self) -> float:
        """Get memory threshold based on optimization level"""
        thresholds = {
            OptimizationLevel.MINIMAL: 0.9,
            OptimizationLevel.STANDARD: 0.8,
            OptimizationLevel.AGGRESSIVE: 0.7,
            OptimizationLevel.MAXIMUM: 0.6
        }
        return thresholds.get(self.optimization_level, 0.8)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "process_memory_percent": process.memory_percent(),
            "system_memory_percent": system_memory.percent,
            "available_memory_mb": system_memory.available / 1024 / 1024,
            "threshold_percent": self.memory_threshold * 100
        }
    
    def should_optimize_memory(self) -> bool:
        """Check if memory optimization is needed"""
        memory_stats = self.get_memory_usage()
        return memory_stats["system_memory_percent"] / 100 > self.memory_threshold
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization"""
        before_stats = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Additional optimizations based on level
        if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
            # Clear caches
            self._clear_function_caches()
            
            # Additional GC passes
            for _ in range(3):
                gc.collect()
        
        after_stats = self.get_memory_usage()
        
        return {
            "before": before_stats,
            "after": after_stats,
            "objects_collected": collected,
            "memory_freed_mb": before_stats["process_memory_mb"] - after_stats["process_memory_mb"]
        }
    
    def _clear_function_caches(self) -> None:
        """Clear function caches"""
        # Clear lru_cache decorated functions
        for obj in gc.get_objects():
            if hasattr(obj, 'cache_clear'):
                try:
                    obj.cache_clear()
                except:
                    pass


class ConcurrencyManager:
    """Advanced concurrency and parallel processing management"""
    
    def __init__(self, max_workers: int = None, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Determine optimal worker counts
        cpu_count = os.cpu_count() or 4
        if max_workers is None:
            max_workers = self._get_optimal_workers(cpu_count)
        
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(max_workers, cpu_count))
        
        # Semaphores for resource control
        self.io_semaphore = asyncio.Semaphore(max_workers * 2)
        self.cpu_semaphore = asyncio.Semaphore(max_workers)
        
    def _get_optimal_workers(self, cpu_count: int) -> int:
        """Get optimal worker count based on optimization level"""
        multipliers = {
            OptimizationLevel.MINIMAL: 1,
            OptimizationLevel.STANDARD: 2,
            OptimizationLevel.AGGRESSIVE: 3,
            OptimizationLevel.MAXIMUM: 4
        }
        multiplier = multipliers.get(self.optimization_level, 2)
        return min(cpu_count * multiplier, 32)  # Cap at 32 workers
    
    async def run_io_bound(self, func: Callable, *args, **kwargs) -> Any:
        """Run IO-bound function with concurrency control"""
        async with self.io_semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
    
    async def run_cpu_bound(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-bound function with concurrency control"""
        async with self.cpu_semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.process_executor, func, *args, **kwargs)
    
    async def run_batch_io(self, func: Callable, items: List[Any], batch_size: int = None) -> List[Any]:
        """Run IO-bound function on batch of items"""
        if batch_size is None:
            batch_size = self.max_workers
        
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [self.run_io_bound(func, item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown executors"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class PerformanceOptimizer:
    """Main performance optimizer coordinating all optimization strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get optimization level
        optimization_level_str = config.get("optimization_level", "standard")
        self.optimization_level = OptimizationLevel(optimization_level_str)
        
        # Initialize components
        self.cache = AdvancedCache(
            cache_type=CacheType(config.get("cache_type", "hybrid")),
            max_size=config.get("cache_max_size", 1000),
            ttl=config.get("cache_ttl", 3600),
            cache_dir=config.get("cache_dir", os.path.join(os.path.expanduser("~"), ".comfyui_mcp_cache"))
        )
        
        self.memory_manager = MemoryManager(self.optimization_level)
        self.concurrency_manager = ConcurrencyManager(
            max_workers=config.get("max_workers"),
            optimization_level=self.optimization_level
        )
        
        # Performance tracking
        self.metrics = []
        self.start_time = time.time()
        
        self.logger.info(f"Performance optimizer initialized with {self.optimization_level.value} level")
    
    def cached(self, ttl: int = None):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = self.cache._generate_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                result = self.cache.get(key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.set(key, result)
                return result
            
            return wrapper
        return decorator
    
    def async_cached(self, ttl: int = None):
        """Decorator for caching async function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key = self.cache._generate_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                result = self.cache.get(key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.cache.set(key, result)
                return result
            
            return wrapper
        return decorator
    
    def monitor_performance(self, func: Callable) -> Callable:
        """Decorator for monitoring function performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.memory_manager.get_memory_usage()
            
            try:
                result = await func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self.memory_manager.get_memory_usage()
                
                # Record metrics
                metrics = PerformanceMetrics(
                    execution_time=end_time - start_time,
                    memory_usage=end_memory["process_memory_mb"] - start_memory["process_memory_mb"],
                    cpu_usage=psutil.cpu_percent(),
                    cache_hits=self.cache.hits,
                    cache_misses=self.cache.misses,
                    concurrent_tasks=len(asyncio.all_tasks()),
                    optimization_level=self.optimization_level
                )
                
                self.metrics.append(metrics)
                
                # Auto-optimize if needed
                if self.memory_manager.should_optimize_memory():
                    self.memory_manager.optimize_memory()
                
                return result
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                raise
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics:
            return {"status": "no_metrics", "message": "No performance data available"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in self.metrics]
        memory_usages = [m.memory_usage for m in self.metrics]
        
        cache_stats = self.cache.get_stats()
        memory_stats = self.memory_manager.get_memory_usage()
        
        return {
            "optimization_level": self.optimization_level.value,
            "uptime_seconds": time.time() - self.start_time,
            "total_operations": len(self.metrics),
            "execution_time": {
                "average": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "total": sum(execution_times)
            },
            "memory": {
                "average_usage_mb": sum(memory_usages) / len(memory_usages),
                "current_stats": memory_stats,
                "peak_usage_mb": max(memory_usages) if memory_usages else 0
            },
            "cache": cache_stats,
            "concurrency": {
                "max_workers": self.concurrency_manager.max_workers,
                "current_tasks": len(asyncio.all_tasks())
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not self.metrics:
            return recommendations
        
        # Cache performance
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache size or TTL for better hit rate")
        
        # Memory usage
        memory_stats = self.memory_manager.get_memory_usage()
        if memory_stats["system_memory_percent"] > 80:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        # Execution time
        execution_times = [m.execution_time for m in self.metrics]
        avg_time = sum(execution_times) / len(execution_times)
        if avg_time > 10:  # 10 seconds
            recommendations.append("Long execution times detected - consider increasing concurrency")
        
        # Optimization level
        if self.optimization_level == OptimizationLevel.MINIMAL:
            recommendations.append("Consider upgrading to STANDARD optimization level for better performance")
        
        return recommendations
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.concurrency_manager.shutdown()
        self.cache.clear()


# Global performance optimizer instance
_performance_optimizer = None


def get_performance_optimizer(config: Dict[str, Any] = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        if config is None:
            config = {"optimization_level": "standard"}
        _performance_optimizer = PerformanceOptimizer(config)
    
    return _performance_optimizer


# Convenience decorators
def cached(ttl: int = 3600):
    """Convenience decorator for caching"""
    optimizer = get_performance_optimizer()
    return optimizer.cached(ttl)


def async_cached(ttl: int = 3600):
    """Convenience decorator for async caching"""
    optimizer = get_performance_optimizer()
    return optimizer.async_cached(ttl)


def monitor_performance(func: Callable) -> Callable:
    """Convenience decorator for performance monitoring"""
    optimizer = get_performance_optimizer()
    return optimizer.monitor_performance(func)
