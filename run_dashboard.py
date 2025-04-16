from visualization.dashboard import Dashboard
from config import config

if __name__ == '__main__':
    dashboard = Dashboard(config['default'])
    dashboard.run_server(debug=True) 