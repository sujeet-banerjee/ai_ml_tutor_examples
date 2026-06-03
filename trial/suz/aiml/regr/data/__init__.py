import pandas as pd
from pathlib import Path

# 1. Get the directory where this __init__.py file lives
current_dir = Path(__file__).parent

# 2. Join that directory with your CSV filename
csv_path_house_price = current_dir / 'regression_continuous_house_price.csv'
csv_path_house_price_val = current_dir / 'regression_continuous_house_price_val.csv'
csv_path_support_tickets = current_dir / 'regression_discrete_Support_Tickets.csv'
csv_path_support_tickets_val = current_dir / 'regression_discrete_Support_Tickets_val.csv'

# 3. Load the data using the absolute path
data_csv_house_price = pd.read_csv(csv_path_house_price)
data_csv_house_price_val = pd.read_csv(csv_path_house_price_val)
data_csv_support_tickets = pd.read_csv( csv_path_support_tickets )
data_csv_support_tickets_val = pd.read_csv(csv_path_support_tickets_val)
print(data_csv_house_price)

