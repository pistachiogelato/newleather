import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Dict
import joblib
import os
from loguru import logger

class SARIMAModel:
    """SARIMA模型实现"""
    
    def __init__(self, config):
        """
        初始化SARIMA模型
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: pd.Series, date_index: pd.DatetimeIndex):
        """
        训练SARIMA模型
        
        Args:
            data: 训练数据
            date_index: 日期索引
        """
        logger.info("开始训练SARIMA模型...")
        
        self.model = SARIMAX(
            data,
            order=self.config.SARIMA_ORDER,
            seasonal_order=self.config.SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False)
        logger.info("SARIMA模型训练完成")
        
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        进行预测
        
        Args:
            steps: 预测步数
            
        Returns:
            预测值、置信区间下界、置信区间上界
        """
        if self.fitted_model is None:
            raise ValueError("模型尚未训练")
            
        forecast = self.fitted_model.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=1-self.config.CONFIDENCE_INTERVAL)
        
        return mean_forecast, conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.fitted_model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.fitted_model, path)
            logger.info(f"SARIMA模型已保存到: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        if os.path.exists(path):
            self.fitted_model = joblib.load(path)
            logger.info(f"SARIMA模型已从 {path} 加载")
        else:
            raise FileNotFoundError(f"未找到模型文件: {path}") 