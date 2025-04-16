import click
from app import create_app
from utils.db_init import init_db
from data.generate_data import generate_sample_data
from models.trainer import train_models
from visualization.dashboard import Dashboard
from config import config
from utils.logger import setup_logger
import os

# 获取配置对象
current_config = config['default']

# 设置日志
logger = setup_logger(current_config)

@click.group()
def cli():
    """皮革工厂AI系统命令行工具"""
    pass

@cli.command()
def init():
    """初始化数据库和目录结构"""
    click.echo('正在初始化系统...')
    try:
        # 确保所有必要的目录存在
        for dir_path in [
            current_config.DATA_DIR,
            current_config.MODEL_SAVE_DIR,
            current_config.LOG_DIR
        ]:
            os.makedirs(dir_path, exist_ok=True)
            click.echo(f'创建目录: {dir_path}')
        
        # 初始化数据库
        init_db(current_config)
        click.echo('系统初始化完成！')
    except Exception as e:
        click.echo(f'初始化失败: {str(e)}')
        raise

@cli.command()
def generate():
    """生成模拟数据"""
    click.echo('正在生成模拟数据...')
    try:
        generate_sample_data()
        click.echo('数据生成完成！')
    except Exception as e:
        click.echo(f'数据生成失败: {str(e)}')
        raise

@cli.command()
def train():
    """训练模型"""
    click.echo('正在训练模型...')
    try:
        # 确保模型保存目录存在
        os.makedirs(current_config.MODEL_SAVE_DIR, exist_ok=True)
        click.echo(f'确保模型保存目录存在: {current_config.MODEL_SAVE_DIR}')
        
        train_models(current_config)
        click.echo('模型训练完成！')
    except Exception as e:
        click.echo(f'模型训练失败: {str(e)}')
        raise

@cli.command()
def serve():
    """启动Web服务"""
    click.echo('正在启动Web服务...')
    try:
        app = create_app('default')
        app.run(
            host=current_config.HOST,
            port=current_config.PORT,
            debug=current_config.DEBUG
        )
    except Exception as e:
        click.echo(f'启动Web服务失败: {str(e)}')
        raise

@cli.command()
def dashboard():
    """启动仪表盘"""
    click.echo('正在启动仪表盘...')
    try:
        dashboard = Dashboard(current_config)
        dashboard.run(debug=current_config.DEBUG)
    except Exception as e:
        click.echo(f'启动仪表盘失败: {str(e)}')
        raise

if __name__ == '__main__':
    cli() 
