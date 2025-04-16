import pandas as pd
import numpy as np
from loguru import logger
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.sarima_model import SARIMAModel
from models.lstm_model import LSTMPredictor
from utils.data_processor import DataProcessor

def train_models(config):
    """
    训练所有模型
    
    Args:
        config: 配置对象，包含模型训练相关的配置信息
    """
    logger.info("开始训练模型...")
    
    try:
        # 初始化组件
        data_processor = DataProcessor(config)
        sarima_model = SARIMAModel(config)
        lstm_predictor = LSTMPredictor(config)
        
        # 加载数据
        df = data_processor.load_data(config.DATA_FILE)
        df = data_processor.add_features(df)
        
        # 训练SARIMA模型
        logger.info("训练SARIMA模型...")
        sarima_model.fit(df['price'], df['date'])
        
        # 准备LSTM数据
        X, y = data_processor.prepare_time_series_data(
            df, 
            'price', 
            config.SEQUENCE_LENGTH, 
            1
        )
        
        # 创建数据加载器
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        train_loader = DataLoader(
            dataset,
            batch_size=config.LSTM_BATCH_SIZE,
            shuffle=True
        )
        
        # 训练LSTM模型
        logger.info("训练LSTM模型...")
        lstm_predictor.build_model(input_size=1)
        lstm_predictor.train(
            train_loader, 
            num_epochs=config.LSTM_EPOCHS, 
            learning_rate=config.LSTM_LEARNING_RATE
        )
        
        # 保存模型
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        sarima_model.save_model(config.SARIMA_MODEL_PATH)
        lstm_predictor.save_model(config.LSTM_MODEL_PATH)
        
        logger.info("模型训练完成")
        
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        raise

if __name__ == '__main__':
    train_models() 
