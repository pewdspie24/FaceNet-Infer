import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int,box.tolist()))
                frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
                if(not isinstance(points_list, list)):
                    points_list= points_list.tolist()
                for usr in points_list:
                    for points in usr:
                        frame = cv2.circle(frame, (int(points[0]), int(points[1])), radius=0, color=(0,0,255), thickness=10)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()