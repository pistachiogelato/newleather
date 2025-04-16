import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from loguru import logger
import os

from utils.data_processor import DataProcessor
from models.predictor import HybridPredictor

class Dashboard:
    """皮革工厂仪表盘"""
    
    def __init__(self, config):
        """
        初始化仪表盘
        
        Args:
            config: 配置对象
        """
        try:
            self.config = config
            self.data_processor = DataProcessor(config)
            self.predictor = HybridPredictor(config)
            
            # 确保模型目录存在
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            
            # 加载预训练模型
            try:
                self.predictor.load_models(config.MODEL_SAVE_DIR)
            except FileNotFoundError:
                logger.warning("未找到预训练模型，部分功能可能不可用")
            
            # 初始化Dash应用
            external_stylesheets = [
                dbc.themes.BOOTSTRAP,
                'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
            ]

            self.app = dash.Dash(
                __name__,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True
            )
            
            # 添加自定义CSS
            self.app.index_string = '''
            <!DOCTYPE html>
            <html>
                <head>
                    {%metas%}
                    <title>皮革工厂智能决策系统</title>
                    {%favicon%}
                    {%css%}
                    <style>
                        .card {
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            transition: all 0.3s ease;
                        }
                        .card:hover {
                            transform: translateY(-5px);
                            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                        }
                        .card-header {
                            background-color: #f8f9fa;
                            border-bottom: none;
                            font-weight: bold;
                        }
                        .btn-primary {
                            background-color: #007bff;
                            border: none;
                            transition: all 0.3s ease;
                        }
                        .btn-primary:hover {
                            background-color: #0056b3;
                            transform: translateY(-2px);
                        }
                    </style>
                </head>
                <body>
                    {%app_entry%}
                    <footer>
                        {%config%}
                        {%scripts%}
                        {%renderer%}
                    </footer>
                </body>
            </html>
            '''
            
            # 设置布局
            self.app.layout = self.create_layout()
            
            # 注册回调
            self.register_callbacks()
            
        except Exception as e:
            logger.error(f"初始化仪表盘失败: {str(e)}")
            raise
    
    def create_layout(self):
        """创建仪表盘布局"""
        return dbc.Container([
            # 标题行
            dbc.Row([
                dbc.Col(html.H1("皮革工厂智能决策系统", className="text-center my-4"))
            ]),
            
            # 主要指标行
            dbc.Row([
                # 价格指标卡片
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("当前价格"),
                        dbc.CardBody(
                            html.H3(id="current-price", className="text-center")
                        )
                    ], className="mb-4"),
                    width=4
                ),
                
                # 库存指标卡片
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("当前库存"),
                        dbc.CardBody(
                            html.H3(id="current-inventory", className="text-center")
                        )
                    ], className="mb-4"),
                    width=4
                ),
                
                # 质量指标卡片
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("平均质量评分"),
                        dbc.CardBody(
                            html.H3(id="quality-score", className="text-center")
                        )
                    ], className="mb-4"),
                    width=4
                )
            ]),
            
            # 价格预测部分
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("价格预测"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("预测周期"),
                                    dcc.Dropdown(
                                        id="prediction-period",
                                        options=[
                                            {"label": "30天", "value": 30},
                                            {"label": "90天", "value": 90}
                                        ],
                                        value=30
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Button(
                                        "更新预测",
                                        id="update-prediction",
                                        className="btn btn-primary mt-4"
                                    )
                                ], width=4)
                            ]),
                            dcc.Graph(id="price-prediction-graph")
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            # 供应链分析部分
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("供应链分析"),
                        dbc.CardBody([
                            dbc.Row([
                                # 库存周转率图表
                                dbc.Col(
                                    dcc.Graph(id="inventory-turnover-graph"),
                                    width=6
                                ),
                                # 生产利用率图表
                                dbc.Col(
                                    dcc.Graph(id="production-utilization-graph"),
                                    width=6
                                )
                            ])
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            # 预警信息部分
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("预警信息"),
                        dbc.CardBody(
                            html.Div(id="warnings-display")
                        )
                    ], className="mb-4")
                ])
            ]),
            
            # 隐藏的数据存储
            dcc.Store(id="market-data"),
            dcc.Interval(
                id="data-update-interval",
                interval=300000,  # 5分钟更新一次
                n_intervals=0
            )
        ], fluid=True)
    
    def register_callbacks(self):
        """注册回调函数"""
        
        @self.app.callback(
            Output("market-data", "data"),
            Input("data-update-interval", "n_intervals")
        )
        def update_market_data(n):
            """更新市场数据"""
            try:
                df = self.data_processor.load_data('data/leather_market_data.csv')
                df = self.data_processor.add_features(df)
                return df.to_json(date_format='iso', orient='split')
            except Exception as e:
                logger.error(f"更新市场数据失败: {str(e)}")
                return None
        
        @self.app.callback(
            [Output("current-price", "children"),
             Output("current-inventory", "children"),
             Output("quality-score", "children")],
            Input("market-data", "data")
        )
        def update_metrics(json_data):
            """更新主要指标"""
            if not json_data:
                return "N/A", "N/A", "N/A"
            
            df = pd.read_json(json_data, orient='split')
            return (
                f"¥{df['price'].iloc[-1]:.2f}",
                f"{df['inventory'].iloc[-1]:.0f}",
                f"{df['quality_score'].iloc[-1]:.1f}"
            )
        
        @self.app.callback(
            Output("price-prediction-graph", "figure"),
            [Input("update-prediction", "n_clicks"),
             Input("prediction-period", "value")],
            State("market-data", "data")
        )
        def update_price_prediction(n_clicks, horizon, json_data):
            """更新价格预测图表"""
            if not json_data:
                return go.Figure()
            
            df = pd.read_json(json_data, orient='split')
            result = self.predictor.predict_price(df, horizon or 30)
            
            return self.create_prediction_graph(
                dates=result['historical_dates'],
                values=result['historical_values'],
                pred_dates=result['dates'],
                pred_values=result['predictions'],
                lower_bound=result['lower_bound'],
                upper_bound=result['upper_bound'],
                title="皮革价格预测"
            )
        
        @self.app.callback(
            [Output("inventory-turnover-graph", "figure"),
             Output("production-utilization-graph", "figure")],
            Input("market-data", "data")
        )
        def update_supply_chain_graphs(json_data):
            """更新供应链图表"""
            if not json_data:
                return go.Figure(), go.Figure()
            
            df = pd.read_json(json_data, orient='split')
            
            # 库存周转率图表
            turnover_fig = px.line(
                df.iloc[-90:],
                x='date',
                y='inventory_turnover',
                title="库存周转率趋势"
            )
            
            # 生产利用率图表
            utilization_fig = px.line(
                df.iloc[-90:],
                x='date',
                y='capacity_utilization',
                title="生产线利用率趋势"
            )
            
            return turnover_fig, utilization_fig
        
        @self.app.callback(
            Output("warnings-display", "children"),
            Input("market-data", "data")
        )
        def update_warnings(json_data):
            """更新预警信息"""
            if not json_data:
                return html.Div("无数据")
            
            df = pd.read_json(json_data, orient='split')
            warnings = []
            
            # 检查库存水平
            if df['inventory'].iloc[-1] < df['inventory'].mean() * 0.5:
                warnings.append(
                    dbc.Alert("库存水平过低，建议及时补货", color="danger")
                )
            elif df['inventory'].iloc[-1] > df['inventory'].mean() * 1.5:
                warnings.append(
                    dbc.Alert("库存水平过高，建议控制采购", color="warning")
                )
            
            # 检查价格异常
            recent_price_change = (
                df['price'].iloc[-1] - df['price'].iloc[-7]
            ) / df['price'].iloc[-7]
            
            if abs(recent_price_change) > 0.1:
                warnings.append(
                    dbc.Alert(
                        f"价格波动异常: {recent_price_change:.1%}",
                        color="warning"
                    )
                )
            
            return html.Div(warnings) if warnings else html.Div("暂无预警信息")
    
    def create_prediction_graph(self, dates, values, pred_dates, pred_values, 
                              lower_bound=None, upper_bound=None, title="预测图表"):
        """创建预测图表"""
        fig = go.Figure()
        
        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name="历史数据",
            line=dict(color='blue')
        ))
        
        # 添加预测数据
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            name="预测值",
            line=dict(color='red', dash='dash')
        ))
        
        # 如果有置信区间，添加置信区间
        if lower_bound is not None and upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name="95%置信区间"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="数值",
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def run(self, debug=False, port=8050):
        """
        启动仪表盘服务器
        
        Args:
            debug: 是否开启调试模式
            port: 服务端口号
        """
        try:
            logger.info(f"正在启动仪表盘服务器，端口: {port}")
            self.app.run(debug=debug, port=port, host='0.0.0.0')
        except Exception as e:
            logger.error(f"启动仪表盘失败: {str(e)}")
            raise 
