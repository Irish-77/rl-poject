import os
import csv
import pandas as pd
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_file = os.path.join(log_dir, f'run_{self.current_time}.csv')
        self.data = []
        
    def log(self, episode, metrics):
        """
        Log metrics for an episode
        metrics: dictionary of metric_name: value pairs
        """
        metrics['episode'] = episode
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.data.append(metrics)
        
        # Write to CSV after each episode
        if len(self.data) == 1:  # First entry, create CSV with headers
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                writer.writeheader()
                writer.writerow(self.data[0])
        else:  # Append to existing CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                writer.writerow(self.data[-1])
    
    def get_data(self):
        """Return all logged data as a pandas DataFrame"""
        return pd.DataFrame(self.data)