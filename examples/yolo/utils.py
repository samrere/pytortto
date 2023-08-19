from collections import Counter
import torch

def intersection_over_union(boxes, boxes_tgt,S):
    """
    # Cbox: box confidence
    :param boxes: (N, B, 5), 5 parameters are: (Cbox, x, y, w, h)
    :param boxes_tgt: (N, 5), 5 parameters are: (1., x, y, w, h)
    :return: ious of all boxes: (N, B)
    """

    boxes_tgt=boxes_tgt[:,None,:] # (N, 1, 5)
    area = boxes[..., 3] * boxes[..., 4] # (N,B)
    area_tgt = boxes_tgt[..., 3] * boxes_tgt[..., 4] # (N,1)

    norm_xy=boxes[...,1:3]/S #/S if relative to box
    norm_xy_tgt=boxes_tgt[...,1:3]/S #/S if relative to box
    lt=torch.max(norm_xy-boxes[...,3:]/2, norm_xy_tgt-boxes_tgt[...,3:]/2) # left-top corner (N, B, 2)
    rb=torch.min(norm_xy+boxes[...,3:]/2, norm_xy_tgt+boxes_tgt[...,3:]/2) # right-bottom corner (N, B, 2)

    wh=(rb - lt).clamp(min=0)
    inter=wh[...,0]*wh[...,1] # (N, B)

    union=area+area_tgt-inter

    return inter/union

# def non_max_suppression(bboxes, iou_threshold, prob_threshold):
#     # predictions = [[c,p,x,y,w,h,j],[...],[...]]
#     bboxes=[box for box in bboxes if box[1]>prob_threshold]
#     bboxes_after_nms=[]
#     bboxes=sorted(bboxes, key=lambda x:x[1], reverse=True)
#     while bboxes:
#         chosen_box=bboxes.pop(0)
#         bboxes=[box for box in bboxes if box[0]!=chosen_box[0] or
#                 intersection_over_union(torch.tensor(chosen_box[1:6])[None],torch.tensor(box[1:6])[None])<iou_threshold]
#         bboxes_after_nms.append(chosen_box)
#     return bboxes_after_nms

def non_max_suppression(bboxes, iou_threshold,S):
    # predictions = tensor([[p,x,y,w,h,j],[...],[...]])
    ind_after_nms=[]
    bboxes_indices=bboxes[:,0].argsort().tolist()
    while bboxes_indices:
        ind=bboxes_indices.pop()
        chosen_box=bboxes[ind:(ind+1)]
        bboxes_indices=[i for i in bboxes_indices if
                        intersection_over_union(chosen_box,bboxes[i:(i+1)], S)<iou_threshold]
        ind_after_nms.append(ind)
    return ind_after_nms



# def mean_average_precision(
#         pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20
# ):
#     """
#     Calculates mean average precision
#
#     Parameters:
#         pred_boxes (list): list of lists containing all bboxes with each bboxes
#         specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
#         true_boxes (list): Similar as pred_boxes except all the correct ones
#         iou_threshold (float): threshold where predicted bboxes is correct
#         num_classes (int): number of classes
#
#     Returns:
#         float: mAP value across all classes given a specific IoU threshold
#     """
#
#     # list storing all AP for respective classes
#     average_precisions = []
#
#     # used for numerical stability later on
#     epsilon = 1e-6
#
#     for c in range(num_classes):
#         detections = []
#         ground_truths = []
#
#         # Go through all predictions and targets,
#         # and only add the ones that belong to the
#         # current class c
#         for detection in pred_boxes:
#             if detection[1] == c:
#                 detections.append(detection)
#
#         for true_box in true_boxes:
#             if true_box[1] == c:
#                 ground_truths.append(true_box)
#
#         # find the amount of bboxes for each training example
#         # Counter here finds how many ground truth bboxes we get
#         # for each training example, so let's say img 0 has 3,
#         # img 1 has 5 then we will obtain a dictionary with:
#         # amount_bboxes = {0:3, 1:5}
#         amount_bboxes = Counter([gt[0] for gt in ground_truths])
#
#         # We then go through each key, val in this dictionary
#         # and convert to the following (w.r.t same example):
#         # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
#         for key, val in amount_bboxes.items():
#             amount_bboxes[key] = torch.zeros(val)
#
#         # sort by box probabilities which is index 2
#         detections.sort(key=lambda x: x[2], reverse=True)
#         TP = torch.zeros((len(detections)))
#         FP = torch.zeros((len(detections)))
#         total_true_bboxes = len(ground_truths)
#
#         # If none exists for this class then we can safely skip
#         if total_true_bboxes == 0:
#             continue
#
#         for detection_idx, detection in enumerate(detections):
#             # Only take out the ground_truths that have the same
#             # training idx as detection
#             ground_truth_img = [
#                 bbox for bbox in ground_truths if bbox[0] == detection[0]
#             ]
#
#             num_gts = len(ground_truth_img)
#             best_iou = 0
#
#             for idx, gt in enumerate(ground_truth_img):
#                 iou = intersection_over_union(
#                     torch.tensor(detection[3:]),
#                     torch.tensor(gt[3:]),
#                 )
#
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_gt_idx = idx
#
#             if best_iou > iou_threshold:
#                 # only detect ground truth detection once
#                 if amount_bboxes[detection[0]][best_gt_idx] == 0:
#                     # true positive and add this bounding box to seen
#                     TP[detection_idx] = 1
#                     amount_bboxes[detection[0]][best_gt_idx] = 1
#                 else:
#                     FP[detection_idx] = 1
#
#             # if IOU is lower than the detection is a false positive
#             else:
#                 FP[detection_idx] = 1
#
#         TP_cumsum = torch.cumsum(TP, dim=0)
#         FP_cumsum = torch.cumsum(FP, dim=0)
#         recalls = TP_cumsum / (total_true_bboxes + epsilon)
#         precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
#         precisions = torch.cat((torch.tensor([1]), precisions))
#         recalls = torch.cat((torch.tensor([0]), recalls))
#         # torch.trapz for numerical integration
#         average_precisions.append(torch.trapz(precisions, recalls))
#
#     return sum(average_precisions) / len(average_precisions)







