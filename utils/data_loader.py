import pandas as pd
from loguru import logger
import sqlite3

class DataLoader:
    """数据加载工具"""
    
    def __init__(self, config):
        """
        初始化数据加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            加载的DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"加载CSV数据失败: {str(e)}")
            raise
    
    def load_db_data(self, table_name: str) -> pd.DataFrame:
        """
        从数据库加载数据
        
        Args:
            table_name: 表名
            
        Returns:
            加载的DataFrame
        """
        try:
            conn = sqlite3.connect(self.config.SQLITE_DB_PATH)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            logger.error(f"加载数据库数据失败: {str(e)}")
            raise 