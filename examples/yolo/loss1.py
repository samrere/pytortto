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

        # reshape target
        Iobj = Iobj.flatten()  # (N*S*S)
        boxes_tgt=boxes_tgt.reshape(-1,5) # (N*S*S, 5)
        labels_tgt=labels_tgt.reshape(-1,self.C) # (N*S*S, C)
        NSS=Iobj.shape[0]

        with torch.no_grad():
            ious = intersection_over_union(boxes, boxes_tgt)  # (N*S*S, B)
        best_box = torch.argmax(ious, dim=-1)  # (N*S*S,)
        mask=torch.zeros_like(ious)# (N*S*S, B)
        mask[torch.arange(NSS), best_box]=1
        mask=mask[...,None] #NSS,B,1

        responsible=boxes*Iobj[:,None,None]*mask # NSS,B,5
        boxes_tgt_obj=boxes_tgt[:,None,:] * mask #NSS,1,5 * NSS,B,1 -> NSS,B,5

        # calc loss
        loss=0
        wh = responsible[..., 3:]  # (NSS, B, 2). select width and height
        loss += self.lambda_coord * (
                self.mse(responsible[..., 1:3], boxes_tgt_obj[..., 1:3])  # x,y
                + self.mse(torch.sign(wh) * torch.sqrt(torch.abs(wh) + 1e-8), torch.sqrt(boxes_tgt_obj[..., 3:]))  # w,h
        )  # coord loss
        loss += self.mse(responsible[..., 0], boxes_tgt_obj[..., 0])  # box confidence loss
        loss += self.mse(labels*Iobj[:,None], labels_tgt*Iobj[:,None])  # class confidence loss

        Inoobj=~Iobj
        classes_tgt_noobj = (boxes_tgt[:,0]*Inoobj)[:,None] #NSS,1
        classes_noobj = boxes[...,0] * Inoobj[:,None] # NSS, B

        loss_noobj = self.lambda_noobj * self.mse(classes_noobj, classes_tgt_noobj)
        loss += self.lambda_noobj * loss_noobj

        return loss/N

