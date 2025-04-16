import numpy as np
import pandas as pd
from typing import Dict, Any, Callable
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler

class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, config):
        """
        初始化超参数调优器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
    def optimize_sarima(self, data: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
        """
        优化SARIMA模型超参数
        
        Args:
            data: 时间序列数据
            n_trials: 优化尝试次数
            
        Returns:
            最优超参数
        """
        def objective(trial):
            # 定义超参数搜索空间
            p = trial.suggest_int('p', 0, 3)
            d = trial.suggest_int('d', 0, 2)
            q = trial.suggest_int('q', 0, 3)
            P = trial.suggest_int('P', 0, 2)
            D = trial.suggest_int('D', 0, 1)
            Q = trial.suggest_int('Q', 0, 2)
            s = trial.suggest_categorical('s', [12, 52])
            
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=3)
            errors = []
            
            for train_idx, test_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    
                    # 训练模型
                    model = SARIMAX(
                        train_data,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted_model = model.fit(disp=False)
                    
                    # 预测并计算误差
                    predictions = fitted_model.forecast(steps=len(test_data))
                    mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
                    errors.append(mape)
                except Exception as e:
                    # 如果模型训练失败，返回高误差
                    return 1000
            
            return np.mean(errors)
        
        # 创建优化研究
        study = optuna.create_study(sampler=TPESampler(), direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # 获取最优参数
        best_params = study.best_params
        logger.info(f"SARIMA最优参数: {best_params}")
        
        return {
            'order': (best_params['p'], best_params['d'], best_params['q']),
            'seasonal_order': (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])
        }
    
    def optimize_lstm(self, X_train, y_train, n_trials: int = 30) -> Dict[str, Any]:
        """
        优化LSTM模型超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_trials: 优化尝试次数
            
        Returns:
            最优超参数
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        def objective(trial):
            # 定义超参数搜索空间
            hidden_size = trial.suggest_int('hidden_size', 16, 128)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # 准备数据
            X = torch.FloatTensor(X_train)
            y = torch.FloatTensor(y_train)
            dataset = TensorDataset(X, y)
            
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=3)
            val_losses = []
            
            for train_idx, val_idx in tscv.split(X):
                X_t, y_t = X[train_idx], y[train_idx]
                X_v, y_v = X[val_idx], y[val_idx]
                
                train_dataset = TensorDataset(X_t, y_t)
                val_dataset = TensorDataset(X_v, y_v)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                
                # 构建模型
                input_size = X.shape[2]
                model = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True
                )
                fc = nn.Linear(hidden_size, 1)
                
                # 训练模型
                optimizer = torch.optim.Adam([*model.parameters(), *fc.parameters()], lr=learning_rate)
                criterion = nn.MSELoss()
                
                for epoch in range(10):  # 简化训练轮数
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        output, _ = model(batch_X)
                        output = fc(output[:, -1, :])
                        loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # 评估模型
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        output, _ = model(batch_X)
                        output = fc(output[:, -1, :])
                        val_loss += criterion(output, batch_y).item()
                
                val_losses.append(val_loss / len(val_loader))
            
            return np.mean(val_losses)
        
        # 创建优化研究
        study = optuna.create_study(sampler=TPESampler(), direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # 获取最优参数
        best_params = study.best_params
        logger.info(f"LSTM最优参数: {best_params}")
        
        return best_params