from torchvision import transforms as trans
import PIL.Image as Image
from mtcnn_file import MTCNN
import torch
import cv2
import os
import glob
from tqdm import tqdm

img_root_dir = "../dataset/original_images/img_align_celeba"
save_dir = "../dataset/celeb"
mtcnn = MTCNN()

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])

ind = 0
embed_map = {}

print("start")

for file in tqdm(glob.glob(f"{img_root_dir}/*")):
    name = file.split("\\")[-1]
    if name.endswith("jpg") or name.endswith("png"):
        try:
            p = file
            img = cv2.imread(p)[:, :, ::-1]
            faces = mtcnn.align_multi(Image.fromarray(img),min_face_size = 64, crop_size = (128,128) )
            if len(faces) == 0:
                continue
            for face in faces:
                scaled_img = face.resize((64, 64), Image.ANTIALIAS)
                new_path = "%08d.jpg"%ind
                ind+=1
                scaled_img.save(os.path.join(save_dir, new_path))
        except Exception as e:
            continue


