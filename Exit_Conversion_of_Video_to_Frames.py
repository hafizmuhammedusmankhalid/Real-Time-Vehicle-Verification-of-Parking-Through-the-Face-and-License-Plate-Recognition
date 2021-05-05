import cv2
vidcap = cv2.VideoCapture(r"Exit_Video\Exit_Video.mp4")
success,image = vidcap.read()
width = 1000
height = 700
dim = (width, height)
count = 0
while success:
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("Exit_Video_Frames/Exit_Video_Frame%d.jpg" % count, resized)     # save frame as JPEG file      
    success,image = vidcap.read()
    count += 1
