from PIL import Image
import numpy as np
import cv2
import math
from facenet_pytorch import MTCNN
import torch

def euclidean_dist(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def align(img, landmarks):
    left_eye_x, left_eye_y = landmarks[0][0], landmarks[0][1]
    right_eye_x, right_eye_y = landmarks[1][0], landmarks[1][1]
    left_eye = (left_eye_x, left_eye_y)
    right_eye = (right_eye_x, right_eye_y)

    #lấy điểm mắt mới cần quay tới
    if left_eye_y > right_eye_y:
        optimal_eye = (right_eye_x, left_eye_y)
        rot = -1 #ngược chiều kim đồng hồ
    else: 
        optimal_eye = (left_eye_x, right_eye_y)
        rot = 1 #cùng chiều kim đồng hồ

    # cv2.circle(img, optimal_eye, radius=0, color=(0,0,255), thickness=3)
    # cv2.circle(img, (left_eye_x, left_eye_y), radius=0, color=(0,0,255), thickness=3)
    # cv2.circle(img, (right_eye_x, right_eye_y), radius=0, color=(0,0,255), thickness=3)
    # image_center = tuple(np.array(img.shape[1::-1]) / 2)
    # rot_mat = cv2.getRotationMatrix2D(image_center, 10, 1.0)
    # img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    # cv2.resize(img, (160,160))

    #tinh khoang cach euclide
    a = euclidean_dist(left_eye, optimal_eye)
    b = euclidean_dist(right_eye, optimal_eye)
    c = euclidean_dist(left_eye, right_eye)

    print(a,b,c)
    print(left_eye)
    print(right_eye)
    print(optimal_eye)

    # Dùng định lý cosine
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle_a = np.arccos(cos_a)

    # chuyen don vi do
    angle = (angle_a*180)/math.pi
    if rot == -1:
        angle = 90 - angle
    
    print(angle)

    # rotated_img = np.array(Image.fromarray(img).rotate(rot*angle))
    
    # xoay anh bang warpaffine
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rot*angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    while True:
        cv2.imshow("SMT", rotated_img)
        if cv2.waitKey(1)&0xFF == 27:
            break

if __name__ == "__main__":
    img = cv2.imread('data/test_images/Quang/2021-07-03-14-38-5828.jpg')
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device)
    boxes, _, landmarks = mtcnn.detect(img, landmarks=True)
    align(img, landmarks.squeeze())
