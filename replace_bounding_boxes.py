import cv2
from PIL import Image
import glob
import pickle
import numpy as np


data_path = "./inputs/swapped/"
a = open("./inputs/processed/boxes.pkl", "rb")
boxes = pickle.load(a)

for file in glob.glob(data_path + "*_m.*g"):
    image_name = file.split("\\")[-1].split("_m")[0]
    box = boxes[image_name + "_t.jpg"]
    box = list(map(int, box))
    w = box[3] - box[1]
    h = box[2] - box[0]
    img = cv2.imread(file)
    img = Image.fromarray(img)
    img = img.resize((h, w), Image.ANTIALIAS)
    img_t = cv2.imread(f"./inputs/original_images/{image_name}_t.jpg")

    img_t_c = img_t[ box[1]:box[3],box[0]:box[2], :]

    img = Image.blend(img, Image.fromarray(img_t_c), 0.25)

    img_t[ box[1]:box[3],box[0]:box[2], :] = img

    cv2.imwrite(f"{data_path}{image_name}_s.jpg", img_t)