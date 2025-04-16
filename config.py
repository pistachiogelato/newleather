import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """基础配置类"""
    
    # 项目根目录
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # 数据目录
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_FILE = os.path.join(DATA_DIR, 'leather_market_data.csv')
    
    # 模型目录
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_SAVE_DIR = os.path.join(MODEL_DIR, 'saved')  # 用于保存训练好的模型
    #MODEL_BASE_PATH = MODEL_DIR  # 添加这一行，使其与dashboard.py兼容
    SARIMA_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'sarima_model.joblib')
    LSTM_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'lstm_model.pt')
    
    # 日志目录
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    LOG_FILE = os.path.join(LOG_DIR, 'leather_factory.log')
    
    # 创建必要的目录
    for dir_path in [DATA_DIR, MODEL_SAVE_DIR, LOG_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 数据库配置
    SQLITE_DB_PATH = os.path.join(DATA_DIR, 'leather_factory.db')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    
    # API配置
    API_VERSION = 'v1'
    API_PREFIX = f'/api/{API_VERSION}'
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    
    # SARIMA模型参数
    SARIMA_ORDER = (1, 1, 1)
    SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
    
    # LSTM模型参数
    LSTM_HIDDEN_SIZE = 64
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    LSTM_BATCH_SIZE = 32
    LSTM_EPOCHS = 50
    LSTM_LEARNING_RATE = 0.001
    
    # 预测配置
    PREDICTION_HORIZON_SHORT = 30
    PREDICTION_HORIZON_LONG = 90
    CONFIDENCE_INTERVAL = 0.95
    SEQUENCE_LENGTH = 30
    
    # 生产配置
    DAILY_PRODUCTION_CAPACITY = 1000
    
    # 预警阈值
    INVENTORY_WARNING_LOW = 7
    INVENTORY_WARNING_HIGH = 30
    PRICE_CHANGE_THRESHOLD = 0.1
    
    # 数据生成配置
    SIMULATION_START_DATE = '2016-01-01'
    SIMULATION_END_DATE = '2021-12-31'
    BASE_LEATHER_PRICE = 100.0
    
    # Web服务配置
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    SQLITE_DB_PATH = os.path.join(Config.DATA_DIR, 'test.db')

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# 环境配置映射
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 
