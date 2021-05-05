import os
import pandas as pd
from datetime import datetime

today = datetime.today()
current_day = today.strftime("%B %d, %Y")
current_time = today.strftime("%H:%M:%S")

file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "r")
recognition_of_license_plate = file.read()

city = pd.DataFrame([[current_day, current_time, recognition_of_license_plate]], columns=['Vehicle Entrace Date', 'Vehicle Entrance Time', 'Vehicle License Plate'])
city.to_csv(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Record\Entrance_Record.csv')

os.system(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Record\Entrance_Record.csv')
