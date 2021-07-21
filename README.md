## **Quick Start**
This is a FaceNet and MTCNN Inference from [this](https://github.com/timesler/facenet-pytorch).
For more information, please read my series: https://viblo.asia/p/nhan-dien-khuon-mat-voi-mang-mtcnn-va-facenet-phan-2-bJzKmrVXZ9N
1. Install:
    
    ```bash
    # Clone Repo:
    git clone https://github.com/pewdspie24/FaceNet-Infer.git
    
    # Install with Pip
    pip install -r requirements.txt

    ```
1. Detection & Capturing:
    ```bash
    # Face Detection:
    python face_detect.py
    
    # Face Capturing (Remember to input your name FIRST in console):
    python face_capture.py

    ```
1. Create FaceList and Recognition:
    ```bash
    # Update FaceList:
    python update_faces.py
    
    # Face Recognition:
    python face_recognition.py

    ```
