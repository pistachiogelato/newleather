from flask import Blueprint, jsonify, request
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from models.predictor import HybridPredictor
from utils.data_processor import DataProcessor

api_bp = Blueprint('api', __name__)

# 全局变量
predictor = None
data_processor = None

def init_models(config):
    """
    初始化模型和数据处理器
    
    Args:
        config: 配置对象
    """
    global predictor, data_processor
    try:
        predictor = HybridPredictor(config)
        data_processor = DataProcessor(config)
        
        # 加载模型
        try:
            # 修改为使用MODEL_SAVE_DIR
            predictor.load_models(config.MODEL_SAVE_DIR)
            logger.info("模型加载成功")
        except FileNotFoundError:
            logger.warning("未找到预训练模型，需要先训练模型")
    except Exception as e:
        logger.error(f"初始化模型失败: {str(e)}")
        raise

@api_bp.route('/predictions/price', methods=['GET'])
def predict_price():
    """价格预测接口"""
    try:
        # 加载数据
        df = data_processor.load_data('data/leather_market_data.csv')
        
        # 获取预测天数
        horizon = request.args.get('horizon', default=30, type=int)
        
        # 进行预测
        result = predictor.predict_price(df, horizon)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"价格预测失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/predictions/demand', methods=['GET'])
def predict_demand():
    """需求预测接口"""
    try:
        # 加载数据
        df = data_processor.load_data('data/leather_market_data.csv')
        
        # 获取预测天数
        horizon = request.args.get('horizon', default=30, type=int)
        
        # 进行预测
        result = predictor.predict_demand(df, horizon)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"需求预测失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/predictions/inventory', methods=['GET'])
def predict_inventory():
    """库存预测接口"""
    try:
        # 加载数据
        df = data_processor.load_data('data/leather_market_data.csv')
        
        # 获取预测天数
        horizon = request.args.get('horizon', default=30, type=int)
        
        # 进行预测
        result = predictor.predict_inventory(df, horizon)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"库存预测失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/supply-chain', methods=['GET'])
def supply_chain_analysis():
    """供应链分析接口"""
    try:
        df = data_processor.load_data('data/leather_market_data.csv')
        df = data_processor.add_features(df)
        stats = data_processor.calculate_statistics(df)
        
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        logger.error(f"供应链分析失败: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@api_bp.route('/market-overview', methods=['GET'])
def market_overview():
    """市场概况接口"""
    try:
        df = data_processor.load_data('data/leather_market_data.csv')
        recent_data = df.iloc[-30:]
        
        overview = {
            'price_trend': {
                'current': float(df['price'].iloc[-1]),
                'change_30d': float(df['price'].iloc[-1] - df['price'].iloc[-30]),
                'avg_30d': float(recent_data['price'].mean())
            },
            'market_status': {
                'daily_volume': float(recent_data['sales'].mean()),
                'inventory_level': float(df['inventory'].iloc[-1])
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': overview
        })
    except Exception as e:
        logger.error(f"市场概况分析失败: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500 
