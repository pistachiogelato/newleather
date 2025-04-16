import numpy as np
import pandas as pd
from typing import Dict, Tuple
from loguru import logger
import torch
from models.sarima_model import SARIMAModel
from models.lstm_model import LSTMPredictor
from utils.data_processor import DataProcessor
from datetime import datetime, timedelta
import os

class HybridPredictor:
    """混合预测模型"""
    
    def __init__(self, config):
        """
        初始化混合预测器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.sarima_model = SARIMAModel(config)
        self.lstm_model = LSTMPredictor(config)
        self.data_processor = DataProcessor(config)
        
    def train(self, df: pd.DataFrame):
        """
        训练混合模型
        
        Args:
            df: 训练数据
        """
        logger.info("开始训练混合模型...")
        
        # 准备数据
        processed_df = self.data_processor.add_features(df)
        
        # 训练SARIMA模型
        self.sarima_model.fit(processed_df['price'], processed_df['date'])
        
        # 准备LSTM数据
        sequence_length = 30  # 使用30天数据预测
        X, y = self.data_processor.prepare_time_series_data(
            processed_df, 'price', sequence_length, 1
        )
        
        # 构建和训练LSTM模型
        self.lstm_model.build_model(input_size=1)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )
        
        # 训练LSTM模型
        self.lstm_model.train(train_loader, num_epochs=50, learning_rate=0.001)
        
        logger.info("混合模型训练完成")
    
    def predict_price(self, df: pd.DataFrame, horizon: int) -> Dict:
        """价格预测"""
        try:
            # SARIMA预测
            sarima_pred, lower_bound, upper_bound = self.sarima_model.predict(horizon)
            
            # LSTM预测
            last_sequence = df['price'].values[-30:].reshape(1, -1, 1)
            lstm_result = self.lstm_model.predict(last_sequence)
            
            # 组合预测结果
            combined_pred = (sarima_pred + lstm_result['mean'].flatten()) / 2
            
            # 生成预测日期
            last_date = df['date'].iloc[-1]
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=horizon, freq='B')
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in pred_dates],
                'predictions': combined_pred.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist(),
                'historical_dates': [d.strftime('%Y-%m-%d') for d in df['date'].tail(30)],
                'historical_values': df['price'].tail(30).tolist()
            }
        except Exception as e:
            logger.error(f"价格预测失败: {str(e)}")
            raise

    def predict_demand(self, df: pd.DataFrame, horizon: int) -> Dict:
        """需求预测"""
        try:
            # 使用历史销售数据的移动平均和趋势
            sales = df['sales'].values
            ma_30 = pd.Series(sales).rolling(window=30).mean().iloc[-1]
            ma_7 = pd.Series(sales).rolling(window=7).mean().iloc[-1]
            
            # 计算趋势
            trend = (ma_7 - ma_30) / ma_30
            
            # 生成预测
            base_demand = sales[-1]
            predictions = []
            for i in range(horizon):
                pred = base_demand * (1 + trend * (i + 1))
                predictions.append(max(0, pred))  # 确保需求非负
            
            # 生成日期
            last_date = df['date'].iloc[-1]
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=horizon, freq='B')
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in pred_dates],
                'predictions': predictions,
                'historical_dates': [d.strftime('%Y-%m-%d') for d in df['date'].tail(30)],
                'historical_values': df['sales'].tail(30).tolist()
            }
        except Exception as e:
            logger.error(f"需求预测失败: {str(e)}")
            raise

    def predict_inventory(self, df: pd.DataFrame, horizon: int) -> Dict:
        """库存预测"""
        try:
            # 获取当前库存水平
            current_inventory = df['inventory'].iloc[-1]
            
            # 预测需求
            demand_pred = self.predict_demand(df, horizon)
            demand_values = demand_pred['predictions']
            
            # 假设日生产能力固定
            daily_production = self.config.DAILY_PRODUCTION_CAPACITY
            
            # 计算预测库存
            inventory_pred = []
            inv = current_inventory
            for demand in demand_values:
                inv = inv + daily_production - demand
                inventory_pred.append(max(0, inv))  # 确保库存非负
            
            # 生成日期
            last_date = df['date'].iloc[-1]
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=horizon, freq='B')
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in pred_dates],
                'predictions': inventory_pred,
                'historical_dates': [d.strftime('%Y-%m-%d') for d in df['date'].tail(30)],
                'historical_values': df['inventory'].tail(30).tolist(),
                'warning_level': self.calculate_inventory_warning(inventory_pred)
            }
        except Exception as e:
            logger.error(f"库存预测失败: {str(e)}")
            raise

    def calculate_inventory_warning(self, inventory_pred: list) -> str:
        """计算库存预警级别"""
        min_inventory = min(inventory_pred)
        max_inventory = max(inventory_pred)
        
        if min_inventory < self.config.INVENTORY_WARNING_LOW:
            return "危险"
        elif max_inventory > self.config.INVENTORY_WARNING_HIGH:
            return "过高"
        else:
            return "正常"

    def save_models(self, base_path: str):
        """保存所有模型"""
        try:
            os.makedirs(base_path, exist_ok=True)
            sarima_path = os.path.join(base_path, 'sarima_model.joblib')
            lstm_path = os.path.join(base_path, 'lstm_model.pt')
            
            self.sarima_model.save_model(sarima_path)
            self.lstm_model.save_model(lstm_path)
            logger.info(f"模型已保存到目录: {base_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise

    def load_models(self, base_path: str):
        """加载所有模型"""
        try:
            # 直接使用传入的base_path，不再拼接
            sarima_path = os.path.join(base_path, 'sarima_model.joblib')
            lstm_path = os.path.join(base_path, 'lstm_model.pt')
            
            if not os.path.exists(sarima_path) or not os.path.exists(lstm_path):
                raise FileNotFoundError(f"模型文件不存在: {base_path}")
            
            self.sarima_model.load_model(sarima_path)
            self.lstm_model.load_model(lstm_path)
            logger.info("所有模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
