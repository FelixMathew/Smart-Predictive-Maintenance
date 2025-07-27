import os
import csv
import time
import random
from datetime import datetime

# Ensure data folder exists
os.makedirs('../data', exist_ok=True)

file_path = '../data/sensor_data.csv'

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'temperature', 'vibration', 'pressure'])

    for _ in range(500):
        timestamp = datetime.now()
        temp = round(random.uniform(50, 120), 2)
        vib = round(random.uniform(0.1, 10.0), 2)
        press = round(random.uniform(80, 160), 2)

        writer.writerow([timestamp, temp, vib, press])
        print(f"{timestamp} | Temp: {temp}, Vib: {vib}, Press: {press}")
        time.sleep(0.2)
