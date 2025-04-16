import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config):
        self.config = config
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            包含各种评估指标的字典
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
    def compare_models(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        比较多个模型的性能
        
        Args:
            y_true: 真实值
            predictions: 包含各模型预测值的字典
            
        Returns:
            包含各模型评估指标的嵌套字典
        """
        results = {}
        for model_name, y_pred in predictions.items():
            results[model_name] = self.evaluate_model(y_true, y_pred)
            
        return results

