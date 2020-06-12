from torchvision import transforms as trans
import PIL.Image as Image
from mtcnn_file import MTCNN
import torch
import cv2
import os
import glob
from tqdm import tqdm
import pickle
from PIL import ImageDraw
from matplotlib import pyplot as plt


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):


    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='white')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline='blue')

    return img_copy

img_root_dir = "../inputs/original_images"
save_dir = "../inputs/processed"

# img_root_dir = "../dataset/original_images/img_align_celeba"
# save_dir = "../dataset/celeb"


mtcnn = MTCNN()

ind = 0
embed_map = {}



print("start")
d = {}
for file in tqdm(glob.glob(f"{img_root_dir}/*")):
# for file in glob.glob(f"{img_root_dir}/*"):

    name = file.split("\\")[-1]
    if name.endswith("jpg") or name.endswith("png"):
        try:
            p = file
            img = cv2.imread(p)[:, :, ::-1] 
            faces,boxes = mtcnn.align_multi(Image.fromarray(img),min_face_size = 64, crop_size = (128,128), )
            if len(faces) == 0:
                continue
            for i,face in enumerate(faces):
                scaled_img = face.resize((64, 64), Image.ANTIALIAS)
                new_path = "%08d.jpg"%ind
                ind+=1
                new_path = p.split('\\')[-1]
                scaled_img.save(os.path.join(save_dir, new_path))
                d[new_path] = boxes[i]
        except :
            continue
a = open(f"{save_dir}/boxes.pkl", "wb")
pickle.dump(d, a)
a.close()
