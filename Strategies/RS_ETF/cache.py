"""
Redis Cache Manager for RS Strategy
Provides caching utilities for market data, backtest results, and RS scores
"""

import redis
import json
import hashlib
import pickle
from functools import wraps
from typing import Optional, Any, Callable
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager for RS strategy data"""
    
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        """
        Initialize cache manager
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info(f"✓ Redis cache connected: {host}:{port}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None
            self.enabled = False
    
    def generate_cache_key(self, prefix: str, **kwargs) -> str:
        """
        Generate consistent cache key from parameters
        
        Args:
            prefix: Key prefix (e.g., 'stock_data', 'backtest')
            **kwargs: Parameters to include in key
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistency
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        hash_key = hashlib.md5(key_data.encode()).hexdigest()
        return f"rs:{prefix}:{hash_key}"
    
    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get cached DataFrame
        
        Args:
            key: Cache key
            
        Returns:
            DataFrame if found, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            data = self.redis_client.get(key)
            if data:
                # Deserialize using pickle for better DataFrame support
                df = pickle.loads(data)
                logger.debug(f"Cache HIT: {key}")
                return df
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting cached DataFrame: {e}")
            return None
    
    def set_dataframe(self, key: str, df: pd.DataFrame, ttl: int = 3600):
        """
        Cache DataFrame with TTL
        
        Args:
            key: Cache key
            df: DataFrame to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        if not self.enabled:
            return
            
        try:
            # Serialize using pickle for better DataFrame support
            data = pickle.dumps(df)
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Cached DataFrame: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Error caching DataFrame: {e}")
    
    def get_json(self, key: str) -> Optional[dict]:
        """
        Get cached JSON data
        
        Args:
            key: Cache key
            
        Returns:
            Dict if found, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            data = self.redis_client.get(key)
            if data:
                result = json.loads(data)
                logger.debug(f"Cache HIT: {key}")
                return result
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting cached JSON: {e}")
            return None
    
    def set_json(self, key: str, data: dict, ttl: int = 3600):
        """
        Cache JSON data with TTL
        
        Args:
            key: Cache key
            data: Dict to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        if not self.enabled:
            return
            
        try:
            json_data = json.dumps(data, default=str)
            self.redis_client.setex(key, ttl, json_data)
            logger.debug(f"Cached JSON: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Error caching JSON: {e}")
    
    def delete(self, key: str):
        """Delete cached item"""
        if not self.enabled:
            return
            
        try:
            self.redis_client.delete(key)
            logger.debug(f"Deleted cache: {key}")
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
    
    def clear_pattern(self, pattern: str):
        """
        Clear all keys matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., 'rs:stock_data:*')
        """
        if not self.enabled:
            return
            
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries matching: {pattern}")
        except Exception as e:
            logger.error(f"Error clearing cache pattern: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
            
        try:
            info = self.redis_client.info()
            return {
                "enabled": True,
                "used_memory": info.get('used_memory_human'),
                "total_keys": self.redis_client.dbsize(),
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


def cache_result(ttl: int = 3600, key_prefix: str = "result"):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        
    Usage:
        @cache_result(ttl=7200, key_prefix="backtest")
        def run_backtest(start_date, end_date):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if instance has cache_manager
            if not hasattr(self, 'cache_manager') or not self.cache_manager.enabled:
                return func(self, *args, **kwargs)
            
            # Generate cache key from function args
            cache_key = self.cache_manager.generate_cache_key(
                prefix=f"{key_prefix}:{func.__name__}",
                args=args,
                kwargs=kwargs
            )
            
            # Try to get from cache
            cached = self.cache_manager.get_json(cache_key)
            if cached is not None:
                return cached
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            self.cache_manager.set_json(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


# Global cache manager instance (initialized in __init__.py)
cache_manager = None


def init_cache_manager(host='localhost', port=6379, db=0, password=None):
    """Initialize global cache manager"""
    global cache_manager
    cache_manager = CacheManager(host=host, port=port, db=db, password=password)
    return cache_manager


def get_cache_manager() -> Optional[CacheManager]:
    """Get global cache manager instance"""
    return cache_manager
