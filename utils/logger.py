from loguru import logger
import sys
import os

def setup_logger(config):
    """
    设置日志配置
    
    Args:
        config: 配置对象
    """
    try:
        # 移除默认的处理器
        logger.remove()
        
        # 添加控制台处理器
        logger.add(
            sys.stderr,
            level=getattr(config, 'LOG_LEVEL', 'INFO'),  # 添加默认值
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # 添加文件处理器
        log_path = getattr(config, 'LOG_PATH', 'logs/leather_factory.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logger.add(
            log_path,
            level=getattr(config, 'LOG_LEVEL', 'INFO'),  # 添加默认值
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="500 MB",
            retention="30 days"
        )
        
        logger.info("日志系统初始化成功")
        return logger
        
    except Exception as e:
        print(f"日志系统初始化失败: {str(e)}")
        raise 