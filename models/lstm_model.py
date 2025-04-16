import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import os
from loguru import logger

class LSTMModel(nn.Module):
    """LSTM模型实现"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 预测均值的全连接层
        self.fc_mean = nn.Linear(hidden_size, 1)
        # 预测方差的全连接层（用于计算置信区间）
        self.fc_var = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_size]
            
        Returns:
            预测均值和方差
        """
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        
        # 预测均值和方差
        mean = self.fc_mean(last_hidden)
        log_var = self.fc_var(last_hidden)  # 预测对数方差以确保方差为正
        var = torch.exp(log_var)
        
        return mean, var

class LSTMPredictor:
    """LSTM预测器封装类"""
    
    def __init__(self, config):
        """
        初始化LSTM预测器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def build_model(self, input_size: int):
        """
        构建LSTM模型
        
        Args:
            input_size: 输入特征维度
        """
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_layers=self.config.LSTM_NUM_LAYERS,
            dropout=self.config.LSTM_DROPOUT
        ).to(self.device)
        
    def train(self, train_loader: torch.utils.data.DataLoader, 
             num_epochs: int, learning_rate: float):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        if self.model is None:
            raise ValueError("模型尚未初始化")
            
        criterion = nn.GaussianNLLLoss()  # 使用高斯负对数似然损失
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                mean, var = self.model(batch_x)
                loss = criterion(mean, batch_y, var)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        进行预测
        
        Args:
            x: 输入数据 [batch_size, seq_len, input_size]
            
        Returns:
            包含预测均值和置信区间的字典
        """
        if self.model is None:
            raise ValueError("模型尚未初始化")
            
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).to(self.device)
            mean, var = self.model(x)
            
            # 计算置信区间
            std = torch.sqrt(var)
            confidence_interval = 1.96 * std  # 95% 置信区间
            
            return {
                'mean': mean.cpu().numpy(),
                'lower_bound': (mean - confidence_interval).cpu().numpy(),
                'upper_bound': (mean + confidence_interval).cpu().numpy()
            }
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"LSTM模型已保存到: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        try:
            if os.path.exists(path):
                if self.model is None:
                    self.build_model(input_size=1)  # 默认输入维度为1
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
                logger.info(f"LSTM模型已从 {path} 加载")
            else:
                raise FileNotFoundError(f"未找到模型文件: {path}")
        except Exception as e:
            logger.error(f"加载LSTM模型失败: {str(e)}")
            raise
