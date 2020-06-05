from torchsummary import summary
from models.aei import AEI_Net
import torch

device= torch.device('cuda')
net= AEI_Net(c_id = 512).to(device)
print(net)