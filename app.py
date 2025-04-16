from flask import Flask, jsonify, request
from flask_cors import CORS
from flasgger import Swagger
from utils.logger import setup_logger
from config import config
from routes.api import api_bp, init_models
import logger

def create_app(config_name='default'):
    """
    工厂函数：创建Flask应用实例
    
    Args:
        config_name: 配置名称，默认为'default'
        
    Returns:
        Flask应用实例
    """
    try:
        # 创建应用实例
        app = Flask(__name__)
        
        # 加载配置
        current_config = config[config_name]
        app.config.from_object(current_config)
        
        # 设置日志
        logger = setup_logger(current_config)
        logger.info("应用初始化开始")
        
        # 初始化CORS
        CORS(app)
        
        # 初始化Swagger
        Swagger(app)
        
        # 初始化模型
        init_models(current_config)
        
        # 注册蓝图
        app.register_blueprint(api_bp, url_prefix=current_config.API_PREFIX)
        
        @app.route('/health')
        def health_check():
            """健康检查接口"""
            return {'status': 'healthy'}, 200
        
        # 添加根路径处理
        @app.route('/')
        def index():
            """根路径处理函数，重定向到API文档"""
            return """
            <html>
                <head>
                    <title>皮革工厂AI系统</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                        h1 { color: #333; }
                        ul { list-style-type: none; padding: 0; }
                        li { margin-bottom: 10px; }
                        a { color: #0066cc; text-decoration: none; }
                        a:hover { text-decoration: underline; }
                    </style>
                </head>
                <body>
                    <h1>皮革工厂AI系统</h1>
                    <p>欢迎使用皮革工厂AI预测系统。请使用以下链接访问系统功能：</p>
                    <ul>
                        <li><a href="/apidocs">API文档</a> - Swagger API文档</li>
                        <li><a href="/api/v1/predictions/price">价格预测API</a> - 获取价格预测</li>
                        <li><a href="/api/v1/predictions/demand">需求预测API</a> - 获取需求预测</li>
                        <li><a href="/api/v1/predictions/inventory">库存预测API</a> - 获取库存预测</li>
                        <li><a href="/health">健康检查</a> - 系统健康状态</li>
                    </ul>
                </body>
            </html>
            """
        
        # 添加全局错误处理
        @app.errorhandler(404)
        def not_found(e):
            """处理404错误"""
            if request.path.startswith(app.config['API_PREFIX']):
                # API路由返回JSON
                return jsonify({
                    'error': '请求的资源不存在',
                    'status_code': 404
                }), 404
            # 非API路由返回HTML
            return f"""
            <html>
                <head>
                    <title>页面未找到 - 皮革工厂AI系统</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                        h1 {{ color: #d9534f; }}
                        a {{ color: #0066cc; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <h1>404 - 页面未找到</h1>
                    <p>您请求的页面 "{request.path}" 不存在。</p>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """, 404
        
        @app.errorhandler(500)
        def server_error(e):
            """处理500错误"""
            logger.error(f"服务器错误: {str(e)}")
            if request.path.startswith(app.config['API_PREFIX']):
                # API路由返回JSON
                return jsonify({
                    'error': '服务器内部错误',
                    'status_code': 500
                }), 500
            # 非API路由返回HTML
            return f"""
            <html>
                <head>
                    <title>服务器错误 - 皮革工厂AI系统</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                        h1 {{ color: #d9534f; }}
                        a {{ color: #0066cc; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <h1>500 - 服务器错误</h1>
                    <p>处理您的请求时发生错误。</p>
                    <p><a href="/">返回首页</a></p>
                </body>
            </html>
            """, 500
        
        logger.info("应用初始化完成")
        return app
        
    except Exception as e:
        logger.error(f"应用初始化失败: {str(e)}")
        raise
