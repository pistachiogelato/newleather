import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import joblib
import os
from datetime import datetime, timedelta

class DataProcessor:
    """数据处理工具类"""
    
    def __init__(self, config):
        """
        初始化数据处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.scalers = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            处理后的DataFrame
        """
        logger.info(f"正在加载数据: {file_path}")
        df = pd.read_csv(file_path)
        
        # 转换日期列
        df['date'] = pd.to_datetime(df['date'])
        
        # 按日期排序
        df = df.sort_values('date')
        
        # 检查数据完整性
        self._check_data_integrity(df)
        
        return df
    
    def _check_data_integrity(self, df: pd.DataFrame):
        """
        检查数据完整性
        
        Args:
            df: 待检查的DataFrame
        """
        # 检查必需列
        required_columns = ['date', 'price', 'production', 'sales', 'inventory', 'quality_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需列: {missing_columns}")
        
        # 检查空值
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"发现空值:\n{null_counts[null_counts > 0]}")
            
        # 检查数值范围
        if (df['price'] <= 0).any():
            logger.warning("发现非正价格")
        if (df['production'] < 0).any():
            logger.warning("发现负生产量")
        if (df['sales'] < 0).any():
            logger.warning("发现负销售量")
        if (df['quality_score'] < 0).any() or (df['quality_score'] > 100).any():
            logger.warning("质量评分超出范围[0,100]")

    def prepare_time_series_data(self, df: pd.DataFrame, 
                               target_col: str,
                               sequence_length: int,
                               prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备时间序列数据
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名
            sequence_length: 输入序列长度
            prediction_horizon: 预测步长
            
        Returns:
            X: 特征数组
            y: 标签数组
        """
        # 标准化数据
        if target_col not in self.scalers:
            self.scalers[target_col] = MinMaxScaler()
            scaled_data = self.scalers[target_col].fit_transform(df[[target_col]])
        else:
            scaled_data = self.scalers[target_col].transform(df[[target_col]])
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length - prediction_horizon + 1):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length:i + sequence_length + prediction_horizon])
        
        # 确保y的形状是 [samples, prediction_horizon] 而不是 [samples, prediction_horizon, 1]
        X = np.array(X)
        y = np.array(y)
        
        # 如果y是三维的，将其转为二维
        if y.ndim == 3:
            y = y.reshape(y.shape[0], y.shape[1])
        
        return X, y
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            增加特征后的DataFrame
        """
        df = df.copy()
        
        # 时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # 移动平均特征
        windows = [7, 14, 30]
        for window in windows:
            df[f'price_ma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'sales_ma_{window}'] = df['sales'].rolling(window=window).mean()
        
        # 价格变化率
        df['price_change'] = df['price'].pct_change()
        
        # 库存周转率 (日销售量/库存)
        df['inventory_turnover'] = df['sales'] / df['inventory']
        
        # 生产利用率 (实际产量/产能)
        df['capacity_utilization'] = df['production'] / self.config.DAILY_PRODUCTION_CAPACITY
        
        # 填充缺失值
        df = df.fillna(method='bfill')
        
        return df
    
    def save_scaler(self, target_col: str, save_dir: str):
        """
        保存标准化器
        
        Args:
            target_col: 目标列名
            save_dir: 保存目录
        """
        if target_col in self.scalers:
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(self.scalers[target_col], 
                       os.path.join(save_dir, f'{target_col}_scaler.joblib'))
    
    def load_scaler(self, target_col: str, save_dir: str):
        """
        加载标准化器
        
        Args:
            target_col: 目标列名
            save_dir: 保存目录
        """
        scaler_path = os.path.join(save_dir, f'{target_col}_scaler.joblib')
        if os.path.exists(scaler_path):
            self.scalers[target_col] = joblib.load(scaler_path)
    
    def inverse_transform(self, scaled_data: np.ndarray, target_col: str) -> np.ndarray:
        """
        反向转换标准化的数据
        
        Args:
            scaled_data: 标准化后的数据
            target_col: 目标列名
            
        Returns:
            原始尺度的数据
        """
        if target_col not in self.scalers:
            raise ValueError(f"未找到{target_col}的标准化器")
        
        # 确保输入数据形状正确
        if scaled_data.ndim == 1:
            scaled_data = scaled_data.reshape(-1, 1)
        
        return self.scalers[target_col].inverse_transform(scaled_data)
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        计算数据统计信息
        
        Args:
            df: 输入DataFrame
            
        Returns:
            统计信息字典
        """
        stats = {
            'price_stats': {
                'mean': df['price'].mean(),
                'std': df['price'].std(),
                'min': df['price'].min(),
                'max': df['price'].max()
            },
            'inventory_stats': {
                'mean': df['inventory'].mean(),
                'turnover_rate': (df['sales'] / df['inventory']).mean()
            },
            'production_stats': {
                'daily_avg': df['production'].mean(),
                'capacity_utilization': (df['production'] / self.config.DAILY_PRODUCTION_CAPACITY).mean()
            },
            'quality_stats': {
                'mean': df['quality_score'].mean(),
                'std': df['quality_score'].std()
            }
        }
        
        return stats

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用高级方法填充缺失值
        
        Args:
            df: 输入DataFrame
        
        Returns:
            填充后的DataFrame
        """
        df_copy = df.copy()
        
        # 时间序列相关列使用插值法
        time_cols = ['price', 'production', 'sales', 'inventory']
        for col in time_cols:
            if df_copy[col].isnull().any():
                # 先尝试线性插值
                df_copy[col] = df_copy[col].interpolate(method='linear')
                # 对于头尾的缺失值，使用最近的有效值
                df_copy[col] = df_copy[col].fillna(method='bfill').fillna(method='ffill')
        
        # 分类特征使用众数填充
        cat_cols = ['year', 'month', 'day_of_week']
        for col in cat_cols:
            if col in df_copy.columns and df_copy[col].isnull().any():
                mode_val = df_copy[col].mode()[0]
                df_copy[col] = df_copy[col].fillna(mode_val)
        
        # 质量评分使用KNN填充
        if 'quality_score' in df_copy.columns and df_copy['quality_score'].isnull().any():
            try:
                from sklearn.impute import KNNImputer
                # 选择相关特征进行KNN填充
                features = ['price', 'production', 'sales', 'inventory']
                imputer = KNNImputer(n_neighbors=5)
                df_copy['quality_score'] = imputer.fit_transform(df_copy[features + ['quality_score']])[:, -1]
            except ImportError:
                logger.warning("sklearn未安装，使用均值填充质量评分")
                df_copy['quality_score'] = df_copy['quality_score'].fillna(df_copy['quality_score'].mean())
        
        return df_copy

    def add_external_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加外部经济指标
        
        Args:
            df: 输入DataFrame
        
        Returns:
            添加外部指标后的DataFrame
        """
        df_copy = df.copy()
        
        # 模拟经济指标数据
        # 实际应用中应从外部API获取真实数据
        date_range = pd.date_range(start=df_copy['date'].min(), end=df_copy['date'].max())
        
        # 生成模拟的GDP增长率 (季度数据)
        gdp_dates = pd.date_range(start=df_copy['date'].min(), end=df_copy['date'].max(), freq='Q')
        gdp_growth = pd.Series(
            np.random.normal(0.02, 0.005, len(gdp_dates)), 
            index=gdp_dates
        )
        
        # 生成模拟的通货膨胀率 (月度数据)
        inflation_dates = pd.date_range(start=df_copy['date'].min(), end=df_copy['date'].max(), freq='M')
        inflation = pd.Series(
            np.random.normal(0.03, 0.01, len(inflation_dates)), 
            index=inflation_dates
        )
        
        # 将季度GDP数据映射到每日
        df_copy['gdp_growth'] = df_copy['date'].map(
            lambda x: gdp_growth[gdp_growth.index.asof(x)]
        )
        
        # 将月度通胀数据映射到每日
        df_copy['inflation'] = df_copy['date'].map(
            lambda x: inflation[inflation.index.asof(x)]
        )
        
        # 添加季节性指标
        df_copy['is_quarter_end'] = df_copy['date'].dt.is_quarter_end.astype(int)
        df_copy['is_month_end'] = df_copy['date'].dt.is_month_end.astype(int)
        
        return df_copy

if __name__ == '__main__':
    # 测试数据处理
    from config import config
    
    processor = DataProcessor(config['default'])
    df = processor.load_data('data/leather_market_data.csv')
    df = processor.add_features(df)
    stats = processor.calculate_statistics(df)
    
    print("数据统计信息:")
    print(stats) 
