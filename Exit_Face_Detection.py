import os 
import cv2
from mtcnn_cv2 import MTCNN

detector = MTCNN()

os.chdir(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Video_Frames\\')

count = 0

for images in os.listdir():
    image = cv2.cvtColor(cv2.imread(images), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    
    if len(result) > 0:
        count = count + 1
        if count < 25:
            continue
        keypoints = result[0]['keypoints']
        bounding_box = result[0]['box']
        roi = cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 3)
        cv2.imwrite(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Test_Face.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Face_Detected\Test_Face.jpg")
        break
        
    else:
        continue