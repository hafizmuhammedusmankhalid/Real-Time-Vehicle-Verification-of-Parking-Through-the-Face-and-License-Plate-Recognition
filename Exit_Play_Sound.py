import os
from playsound import playsound

file0 = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_License_Plate\Checking_Authorization_License_Plate.txt", "r")
Checking_Authorization_License_Plate = file0.read()

print(Checking_Authorization_License_Plate)

file1 = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Checking_Authorization_Face.txt", "r")
Checking_Authorization_Face = file1.read()

print(Checking_Authorization_Face)

file2 = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Decoding_of_QR_Code\Checking_Authorization_of_QR_Code.txt", "r")
Checking_Authorization_of_QR_Code = file2.read()           

print(Checking_Authorization_of_QR_Code)

os.chdir(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Play_Sound\\')

if (Checking_Authorization_License_Plate == "1" and Checking_Authorization_Face == "1" and Checking_Authorization_of_QR_Code == "1") or (Checking_Authorization_License_Plate == "1" and Checking_Authorization_of_QR_Code == "1"):
    playsound('Exit_Allowed_Play_Sound.mp3')
else:
    playsound('Exit_Not_Allowed_Play_Sound.mp3')
