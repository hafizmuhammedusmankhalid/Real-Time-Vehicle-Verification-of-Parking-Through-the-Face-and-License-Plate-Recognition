import os
from playsound import playsound

file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "r")
recognition_of_license_plate = file.read()

os.chdir(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Play_Sound\\')

if not recognition_of_license_plate:
    playsound('Entrance_Not_Allowed_Play_Sound.mp3')
else:
    playsound('Entrance_Allowed_Play_Sound.mp3')
