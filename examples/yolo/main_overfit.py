from loss import *
from dataset import *
from model import *
from model_resnetyolo import *
from viz import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

C=20
B=2
S=7
batch_size = 4
learning_rate = 1e-5


transform = A.Compose([
    A.Resize(448, 448),
    A.Normalize(), # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ToTensorV2()
], bbox_params=A.BboxParams(format='albumentations')) # format=albumentations is normalized pascal_voc.


train_dataset=VOCDataset(root='data', train=False, transform=transform)
train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# net = Yolov1(split_size=S, num_boxes=B, num_classes=C).cuda()
net=YOLOv1ResNet(S=S, B=B,C=C).cuda()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = YoloLoss(S=S, B=B, C=C)


def train():
    net.train()
    train_loss = 0
    for i, (img, (boxes, labels, Iobj), index) in enumerate(train_loader):
        img, boxes, labels, Iobj = img.cuda(), boxes.cuda(), labels.cuda(), Iobj.cuda()
        out = net(img)
        loss = loss_fn(out, (boxes, labels, Iobj))
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
    return train_loss, i

net.train()
loop=tqdm(range(100))
for epoch in loop:
    mean_loss,i=train()
    loop.set_postfix({'Mean loss':mean_loss / (i + 1)})
torch.save(net.state_dict(),'model.pt')


# net.load_state_dict(torch.load('model.pt'))
# net.eval()
# with torch.no_grad():
#     for i, (img, (boxes, labels, Iobj), index) in enumerate(dataloader):
#         img, boxes, labels, Iobj = img.cuda(), boxes.cuda(), labels.cuda(), Iobj.cuda()
#         out = net(img)
#         break
