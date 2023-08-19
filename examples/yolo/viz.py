import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import name2id
from utils import *

id2name = dict([(value, key) for key, value in name2id.items()])

BOX_COLOR = (255, 0, 0)  # Red
BOX_COLOR_TRUE = (1, 150, 32) # green
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=1):
    """Visualizes a single bounding box on the image"""
    p,x_min, y_min, w, h= bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    text=f'{class_name} {p:.3f}'
    ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0)
    cv2.rectangle(img, (x_min, y_min - int(1.5 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=text,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
        thickness=1
    )
    return img

def visualize_bbox_true(img, bbox, class_name, color=BOX_COLOR_TRUE, thickness=1):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max= bbox
    x_min, x_max, y_min, y_max=int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0)
    cv2.rectangle(img, (x_max-text_width, y_max), (x_max, y_max+ int(1.5 * text_height)), BOX_COLOR_TRUE, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_max- text_width, y_max + int(1.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
        thickness=1
    )
    return img


def visualize(img_id, dataset, bboxes,S,plot_truth=True):
    file = dataset.dataset[img_id]
    img = np.array(file[0])
    box_w, box_h = int(file[1]['annotation']['size']['width']), int(file[1]['annotation']['size']['height'])
    for c,bbnd in bboxes:
        class_name = id2name[c]
        for bbox in bbnd:
            p=bbox[0]
            w=box_w*bbox[3] # add /S if w if relative to cell instead of relative to the image
            h=box_h*bbox[4] # add /S if h if relative to cell instead of relative to the image
            xmin = box_w/S * (bbox[1])-w/2
            ymin = box_h/S * (bbox[2])-h/2
            img = visualize_bbox(img, [p.item(),xmin.item(),ymin.item(),w.item(),h.item()], class_name)
    if plot_truth:
        for obj in file[1]['annotation']['object']:
            class_name=obj['name']
            xmin=obj['bndbox']['xmin']
            xmax=obj['bndbox']['xmax']
            ymin = obj['bndbox']['ymin']
            ymax = obj['bndbox']['ymax']
            img = visualize_bbox_true(img, [xmin, xmax, ymin, ymax], class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

