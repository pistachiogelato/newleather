import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from loguru import logger

def generate_sample_data():
    """生成样本数据"""
    try:
        logger.info("开始生成样本数据...")
        
        # 生成日期范围（2016-2021年的工作日）
        start_date = datetime(2016, 1, 1)
        end_date = datetime(2021, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 生成基础价格序列
        n_days = len(dates)
        base_price = 100.0  # 基准价格（元/平方米）
        
        # 添加趋势、季节性和随机波动
        trend = np.linspace(0, 0.5, n_days)  # 上升趋势
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # 年度季节性
        noise = np.random.normal(0, 0.05, n_days)  # 随机波动
        
        prices = base_price * (1 + trend + seasonal + noise)
        prices = np.maximum(prices, base_price * 0.5)  # 确保价格不会太低
        
        # 生成生产数据
        daily_capacity = 1000  # 日产能（平方米）
        production = np.random.normal(daily_capacity, daily_capacity * 0.1, n_days)
        production = np.maximum(production, 0)  # 确保生产量非负
        
        # 生成销售数据
        sales = production * np.random.normal(0.95, 0.1, n_days)
        sales = np.maximum(sales, 0)  # 确保销售量非负
        
        # 计算库存
        inventory = np.zeros(n_days)
        inventory[0] = production[0] - sales[0]
        for i in range(1, n_days):
            inventory[i] = inventory[i-1] + production[i] - sales[i]
        inventory = np.maximum(inventory, 0)  # 确保库存非负
        
        # 生成质量评分（60-100分）
        quality_scores = np.random.normal(90, 5, n_days)
        quality_scores = np.clip(quality_scores, 60, 100)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'production': production,
            'sales': sales,
            'inventory': inventory,
            'quality_score': quality_scores
        })
        
        # 确保data目录存在
        os.makedirs('data', exist_ok=True)
        
        # 保存数据
        output_path = 'data/leather_market_data.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"数据已保存到: {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"生成样本数据失败: {str(e)}")
        raise

if __name__ == '__main__':
    generate_sample_data() 