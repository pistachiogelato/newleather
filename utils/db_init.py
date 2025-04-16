import sqlite3
from loguru import logger
import os

def init_db(config):
    """
    初始化数据库
    
    Args:
        config: 配置对象，包含数据库路径等配置信息
    """
    logger.info("开始初始化数据库...")
    
    try:
        # 确保数据目录存在
        os.makedirs(os.path.dirname(config.SQLITE_DB_PATH), exist_ok=True)
        
        # 连接数据库
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # 创建价格数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            date TEXT PRIMARY KEY,
            price REAL NOT NULL,
            production REAL NOT NULL,
            sales REAL NOT NULL,
            inventory REAL NOT NULL,
            quality_score REAL NOT NULL
        )
        ''')
        
        # 创建供应商评级表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS supplier_ratings (
            supplier_id INTEGER PRIMARY KEY AUTOINCREMENT,
            supplier_name TEXT NOT NULL,
            quality_rating REAL NOT NULL,
            delivery_rating REAL NOT NULL,
            price_rating REAL NOT NULL,
            last_update TEXT NOT NULL
        )
        ''')
        
        # 创建库存预警表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_date TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            alert_message TEXT NOT NULL,
            is_active INTEGER NOT NULL
        )
        ''')
        
        # 创建预测记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT NOT NULL,
            target_date TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            confidence_lower REAL NOT NULL,
            confidence_upper REAL NOT NULL
        )
        ''')
        
        conn.commit()
        logger.info(f"数据库初始化完成: {config.SQLITE_DB_PATH}")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        raise
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    init_db() 