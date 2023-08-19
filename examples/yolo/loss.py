import warnings
warnings.filterwarnings("ignore",category=UserWarning)
import torch
import torch.nn as nn
from utils import intersection_over_union
# torch.autograd.set_detect_anomaly(True)

class YoloLoss(nn.Module):
    def __init__(self, S, B, C):
        super(YoloLoss, self).__init__()
        self.mse=nn.MSELoss(reduction='sum')
        self.S=S
        self.B=B
        self.C=C
        self.lambda_coord=5
        self.lambda_noobj=0.5
    def forward(self, predictions, target):
        # extract boxes and labels from predictions
        predictions = predictions.reshape(-1, self.C + 5 * self.B)  # (N, S*S*(C+5B))->(N*S*S, C+5B)
        labels = predictions[..., :20]  # (N*S*S, C)
        boxes = predictions[..., 20:].reshape(-1, self.B, 5)  # (N*S*S, B, 5)

        # boxes_tgt: (N, S*S, 5). labels_tgt: (N, S*S, C). Iobj: (N, S*S)
        boxes_tgt, labels_tgt, Iobj=target
        N=boxes_tgt.shape[0]

        # flatten out batch dimension. select boxes that exists
        Iobj = Iobj.flatten()  # (N*S*S)
        boxes_tgt=boxes_tgt.reshape(-1,5) # (N*S*S, 5)
        labels_tgt=labels_tgt.reshape(-1,self.C) # (N*S*S, C)

        # select cells that have bounding boxes. Nobj: number of boxes that exists
        boxes_tgt_obj=boxes_tgt[Iobj] # (Nobj, 5), could be empty
        Nobj = boxes_tgt_obj.shape[0]
        loss=0
        if Nobj !=0: # if boxes exists:
            # find responsible boxes
            boxes_obj=boxes[Iobj] # (Nobj, B, 5)
            with torch.no_grad():
                ious=intersection_over_union(boxes_obj, boxes_tgt_obj,self.S) # (Nobj, B)
            best_box=torch.argmax(ious, dim=-1) # (Nobj,)
            responsible=boxes_obj[torch.arange(Nobj),best_box,:] # (Nobj, 5)
            # calc loss
            wh=responsible[...,3:]  # (Nobj, 2). select width and height
            loss+=self.lambda_coord*(
                self.mse(responsible[..., 1:3], boxes_tgt_obj[..., 1:3])  # x,y
                + self.mse(torch.sign(wh) * torch.sqrt(torch.abs(wh)+1e-8), torch.sqrt(boxes_tgt_obj[...,3:])) # w,h
            ) # coord loss
            loss += self.mse(responsible[..., 0], boxes_tgt_obj[..., 0]) # box confidence loss

            ######### box iou loss
            # loss += self.mse(ious[ax0, best_box],boxes_tgt_obj[..., 0])  # box iou loss
            #############

            loss+=self.mse(labels[Iobj], labels_tgt[Iobj])# class confidence loss

        # select cells that have no bounding boxes.
        boxes_tgt_noobj = boxes_tgt[~Iobj]  # (Nnoobj, 5), could be empty
        if boxes_tgt_noobj.shape[0]!=0:
            boxes_noobj=boxes[~Iobj] # (Nnoobj, B, 5)
            loss_noobj=self.lambda_noobj*self.mse(boxes_noobj[...,0], boxes_tgt_noobj[...,0,None])
            loss+=self.lambda_noobj*loss_noobj

        return loss/N



# if __name__ == '__main__':
#     from tqdm import tqdm
#     from torch.utils.data import DataLoader
#     from dataset import VOCDataset
#     import albumentations as A
#     from albumentations.pytorch import ToTensorV2
#     torch.manual_seed(0)
#
#     batch_size=32
#     S=7
#     B=1
#     C=20
#
#     transform = A.Compose([
#         A.Resize(448, 448),
#         # A.HorizontalFlip(),
#         A.Normalize(),
#         ToTensorV2()
#     ], bbox_params=A.BboxParams(format='albumentations'))  # format=albumentations is normalized pascal_voc.
#     ds = VOCDataset(root='data', train=False, transform=transform)
#     dl=DataLoader(ds, batch_size=batch_size, shuffle=False,num_workers=12)
#     L = YoloLoss(S=S, B=B, C=C)
#     for data,target,index in tqdm(dl):
#         predictions=torch.randn(data.shape[0],S*S*(C+5*B), requires_grad=True)
#         loss=L(predictions,target)
#         loss.backward()


