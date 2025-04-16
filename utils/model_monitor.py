import pandas as pd
import numpy as np
from loguru import logger
import time
import os
from models.evaluator import ModelEvaluator
from typing import Dict

class ModelMonitor:
    """模型监控器"""
    
    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor
        self.evaluator = ModelEvaluator(config)
        self.performance_history = []
        
    def check_performance(self, df: pd.DataFrame) -> Dict:
        """
        检查模型性能
        
        Args:
            df: 包含最新数据的DataFrame
            
        Returns:
            性能评估结果
        """
        # 使用最近30天数据进行评估
        test_data = df.iloc[-30:].copy()
        
        # 获取实际值
        actual_values = test_data['price'].values
        
        # 使用前一天数据预测当天
        predictions = []
        for i in range(len(test_data) - 1):
            pred_df = df.iloc[:-(len(test_data) - i)]
            pred = self.predictor.predict_price(pred_df, 1)
            predictions.append(pred['predictions'][0])
        
        # 添加最后一天的预测
        predictions.append(self.predictor.predict_price(df.iloc[:-1], 1)['predictions'][0])
        
        # 评估性能
        performance = self.evaluator.evaluate_model(actual_values, np.array(predictions))
        
        # 记录性能历史
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': performance
        })
        
        return performance
        
    def should_retrain(self, current_performance: Dict) -> bool:
        """
        判断是否需要重新训练模型
        
        Args:
            current_performance: 当前性能指标
            
        Returns:
            是否需要重新训练
        """
        # 如果没有历史记录，不需要重新训练
        if len(self.performance_history) < 5:
            return False
            
        # 获取最近5次的MAPE
        recent_mapes = [record['metrics']['mape'] for record in self.performance_history[-5:]]
        avg_mape = np.mean(recent_mapes)
        
        # 如果当前MAPE比平均值高20%，触发重新训练
        if current_performance['mape'] > avg_mape * 1.2:
            logger.warning(f"模型性能下降，当前MAPE: {current_performance['mape']:.2f}，平均MAPE: {avg_mape:.2f}")
            return True
            
        return False