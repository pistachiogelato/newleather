from loguru import logger
import time
from functools import wraps
import psutil
import os

def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        logger.info(f"函数 {func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        logger.info(f"内存使用: {end_memory - start_memory:.2f}MB")
        
        return result
    return wrapper

class PerformanceMonitor:
    """性能监控器"""
    
    @staticmethod
    def log_system_status():
        """记录系统状态"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logger.info(f"CPU使用率: {cpu_percent}%")
        logger.info(f"内存使用: {memory.percent}%")
        logger.info(f"磁盘使用: {disk.percent}%") 