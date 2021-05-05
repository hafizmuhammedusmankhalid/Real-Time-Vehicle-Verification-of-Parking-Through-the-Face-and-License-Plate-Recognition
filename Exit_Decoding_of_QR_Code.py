import sys
import cv2
import time
import numpy as np
from pyzbar.pyzbar import decode

#cap = cv2.VideoCapture("http://192.168.18.20:8080/video")

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)
check = 0 
    
with open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt") as f:
    myDataList = f.read().splitlines()

while True:
 
    success, img = cap.read()
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        print(myData)
 
        if myData in myDataList:
            myOutput = 'Authorized'
            myColor = (0,255,0)
            file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Decoding_of_QR_Code\Checking_Authorization_of_QR_Code.txt", "a")
            file. truncate(0)
            file.write("1")
            check = 1
            file.close()
        
        else:
            myOutput = 'Un-Authorized'
            myColor = (0, 0, 255)
            myOutput = 'Authorized'
            myColor = (0,255,0)
            file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Decoding_of_QR_Code\Checking_Authorization_of_QR_Code.txt", "a")
            file.truncate(0)
            file.write("0")
            check = 2
            file.close()

            
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,myColor,5)
        pts2 = barcode.rect
        cv2.putText(img,myOutput,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,myColor,2)
 
    cv2.imshow('Decoding of Quick Response Code', img)
    cv2.waitKey(1)
    
    if check == 1 or check == 2:
        time.sleep(10)
        cv2.destroyAllWindows()
        sys.exit()