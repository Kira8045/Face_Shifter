from models.multiscaleDiscriminator import *
from torch.utils.data import DataLoader
from preprocessing.model import Backbone
from utils.Dataset import Dataset_Sample as Dataset
import torch.nn.functional as F
import torch.optim as optim
from models.aei import *
import torchvision
import torch
import time
import cv2
import os
import visdom
from PIL import Image

vis = visdom.Visdom(server = "127.0.0.1", env = "Face_Swap", port = 8097)

batch_size = 4
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 2000
show_step = 10
save_epoch = 1
start_batch = 0
model_save_path = "./model_weights/"
optim_level = '01'

device = torch.device("cuda")

G = AEI_Net(512).to(device)
D = MultiscaleDiscriminator(input_nc = 3,ndf =64,   n_layers= 6, norm_layer= torch.nn.InstanceNorm2d).to(device)
G.train()
D.train()


arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load("./model_weights/model_ir_se50.pth"))

dataset =Dataset("./inputs/processed")

dataloader= DataLoader(dataset, batch_size = batch_size, num_workers = 0)

MSE = torch.nn.MSELoss()    
L1= torch.nn.L1Loss()

def hinge_loss(X, positive = True):
    if positive:
        return torch.relu(1 -X).mean()
    return torch.relu(X).mean()

def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow = X.shape[0]) *0.5 + 0.5
    return X

def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)

    return torch.cat((Xs, Xt, Y), dim =1).numpy()

def make_image_demo(Xs, Xt):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)

    return torch.cat((Xs, Xt), dim =1).numpy()


if os.path.exists("./model_weights/G_recent.pth") and os.path.exists("./model_weights/D_recent.pth"):
    G.load_state_dict(torch.load(f"{model_save_path}G_recent.pth", map_location = device), strict = False)
    D.load_state_dict(torch.load(f"{model_save_path}D_recent.pth", map_location = device), strict = False)

    print("loaded existing models")

for iteration, data in enumerate(dataloader):
    start_time = time.time()
    Xs, Xt, same_person, image_path = data
    Xs = Xs.to(device)
    Xt = Xt.to(device)
    with torch.no_grad():
        embed, Xs_feats = arcface(F.interpolate(Xs, [112, 112], mode = "bilinear", align_corners= True))
    same_person = same_person.to(device)

    Y, Xt_attr = G(Xt, embed)

    Di = D(Y)
    L_adv = 0
    for di in Di:
        L_adv += hinge_loss(di[0], True)
    
    Y_aligned = Y
    ZY, Y_feats = arcface(F.interpolate( Y_aligned, [112, 112], mode = "bilinear", align_corners= True ))
    L_id = (1 - torch.cosine_similarity(embed, ZY, dim = 1) ).mean()

    Y_attr = G.get_attr(Y)
    L_attr = 0
    for i in range(len(Xt_attr)):
        L_attr = torch.mean( torch.pow( Xt_attr[i] - Y_attr[i], 2 ).reshape(batch_size, -1), dim = 1 ).mean()
    L_attr /=2

    L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    # weights of losses to give importance to required features of an image.
    #weight of identity loss was increased to decrease time to train the model to produce identifiable images
    l_adv = 1
    l_attr = 10
    l_id = 17
    l_rec = 10

    loss_G = l_adv * L_adv + l_attr * L_attr + l_id * L_id + l_rec * L_rec


    fake_D = D(Y.detach())
    loss_fake = 0
    for di in fake_D:
        loss_fake += hinge_loss(di[0], False)
    
    true_D = D(Xs)
    loss_true = 0
    for di in true_D:
        loss_true += hinge_loss(di[0], True)
    
    loss_D = 0.5 * ( loss_fake.mean() + loss_true.mean() )

    batch_time = time.time() - start_time

    for i,y in enumerate(Y.detach().cpu().numpy()):
        y = y*0.5 + 0.5
        y = cv2.convertScaleAbs(y, alpha=(255.0))
        cv2.imwrite(f'./inputs/swapped/{image_path[i]}_m.jpg', y.transpose([1,2,0]))

    image = make_image(Xs, Xt, Y)
    vis.image(image[::-1, :, :], opts = {'title': 'result'}, win = "result")
    image = cv2.convertScaleAbs(image, alpha=(255.0))
    cv2.imwrite('./outputs/sample_vis_demo_image.jpg', image.transpose([1,2,0]))


    print(f'lossD: {loss_D.item()}     loss_G: {loss_G.item()}   batch_time: {batch_time}')
    print(f'loss_attr: {L_attr.item()}  loss_id: {L_id.item()}  loss_rec: {L_rec.item()}')
