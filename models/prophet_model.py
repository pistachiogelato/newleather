import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib
import os
from loguru import logger

class ProphetModel:
    """Prophet模型实现"""
    
    def __init__(self, config):
        """
        初始化Prophet模型
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        
    def fit(self, data: pd.Series, date_index: pd.DatetimeIndex):
        """
        训练Prophet模型
        
        Args:
            data: 训练数据
            date_index: 日期索引
        """
        try:
            from prophet import Prophet
            logger.info("开始训练Prophet模型...")
            
            # 准备Prophet格式数据
            df = pd.DataFrame({
                'ds': date_index,
                'y': data.values
            })
            
            # 初始化并训练模型
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # 添加季度季节性
            self.model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=5
            )
            
            self.model.fit(df)
            logger.info("Prophet模型训练完成")
            
        except ImportError:
            logger.error("未安装prophet库，请使用pip install prophet安装")
            raise
    
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        进行预测
        
        Args:
            steps: 预测步数
            
        Returns:
            预测值、置信区间下界、置信区间上界
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 创建未来日期DataFrame
        future = self.model.make_future_dataframe(periods=steps)
        
        # 预测
        forecast = self.model.predict(future)
        
        # 获取预测结果
        predictions = forecast.iloc[-steps:]['yhat'].values
        lower_bound = forecast.iloc[-steps:]['yhat_lower'].values
        upper_bound = forecast.iloc[-steps:]['yhat_upper'].values
        
        return predictions, lower_bound, upper_bound
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                joblib.dump(self.model, f)
            logger.info(f"Prophet模型已保存到: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = joblib.load(f)
            logger.info(f"Prophet模型已从 {path} 加载")
        else:
            raise FileNotFoundError(f"模型文件 {path} 不存在")