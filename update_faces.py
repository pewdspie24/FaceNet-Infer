import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np

IMG_PATH = './data/test_images'
DATA_PATH = './data'

device = 'cpu'
print(device)
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)
    
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        # print(usr)
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            # print('smt')
            embeds.append(model(trans(img).to(device).unsqueeze(0))) #1 anh, kich thuoc [1,512]
    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
    embeddings.append(embedding) # 1 cai list n cai [1,512]
    # print(embedding)
    names.append(usr)
    
embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)
print(embeddings.shape)
if device == 'cpu':
    torch.save(embeddings, DATA_PATH+"/faceslistCPU.pth")
else:
    torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)