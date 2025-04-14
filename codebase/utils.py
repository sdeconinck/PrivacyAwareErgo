import torch
import torch.nn as nn
import torchvision
from ultralytics.utils.plotting import Annotator
from copy import deepcopy
import numpy as np
import cv2
import re
import os

connect_skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]
import torch
from torch.utils.data import DataLoader
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from torch.utils.data import dataloader, distributed
from ultralytics.utils import RANK, colorstr
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS

def build_dataloader(dataset, batch, workers, sampler = None, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    #sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def convert_confidence_to_visibility(confidence, low_threshold=0.5, high_threshold=0.7):
    if confidence < low_threshold:
        return 0  # Not Visible
    elif low_threshold <= confidence < high_threshold:
        return 1  # Visible but not clear
    else:
        return 2  # Clearly Visible


def visualize_keypoints_predicted(images, predictions):
    images_w_keypoints = torch.zeros_like(images)
    for i in range(images.shape[0]):
        if not predictions[i].keypoints.has_visible:
            images_w_keypoints[i] = images[i]
            continue
        confs = predictions[i].keypoints.conf.cpu().apply_(lambda x: convert_confidence_to_visibility(x))
        images_w_keypoints[i] = torchvision.utils.draw_keypoints(images[i],
                                                                 predictions[i].keypoints.xy,
                                                                 connectivity=connect_skeleton, radius=10,
                                                                 colors="red", visibility=confs)
    return images_w_keypoints


def visualize_keypoints_gt(data):
    normal_w_keypoints = torch.zeros_like(data['img'])
    for i in range(data['img'].shape[0]):
        keypoints = data['keypoints'][data['batch_idx'] == i]
        if len(keypoints) == 0:
            normal_w_keypoints[i] = data['img'][i]
            continue
        normal_w_keypoints[i] = torchvision.utils.draw_keypoints(data['img'][i],
                                                                 keypoints[:, :, :2] * 640,
                                                                 connectivity=connect_skeleton, radius=10,
                                                                 colors="red", visibility=keypoints[:, :, 2])
    return normal_w_keypoints


def annotate_img(img, kpts, bboxs, ids=None):
    if ids is None:
        ids = range(len(kpts))
    else:
        assert len(ids) == len(kpts), f'# ids ({len(ids)}) != # kpts ({len(kpts)})'
    annotator = Annotator(deepcopy(img))

    for id, skel, bbox in zip(ids, kpts, bboxs):
        annotator.box_label(bbox, f'{id}')
        annotator.kpts(skel[:, :2], shape=img.shape[:2], radius=5, kpt_line=True)
    img_annotated = annotator.result()
    return img_annotated
    
def extract_kpts(results, treshold=0.01):
    all_kpts, all_bbox = [], []
    for result in results:

        if result.keypoints.xy is None or result.boxes.xyxy is None or result.keypoints.conf is None:       
            continue
        for kpts, conf, bbox in zip(result.keypoints.xy, result.keypoints.conf, result.boxes.xyxy):
            # print(conf)
            # Each 'result' corresponds to a detected person
            kpts_with_conf = np.concatenate([kpts.cpu().numpy(), conf.cpu().numpy()[:, None]], axis=1)
            # if column 1 and column 2 are 0 then column 3 is 0
            kpts_with_conf[:, 2] = kpts_with_conf[:, 2] * (kpts_with_conf[:, 0] != 0) * (kpts_with_conf[:, 1] != 0)
            # Add column to np array. 1 if confidence is above treshold, 0 otherwise
            kpts_with_conf_treshold = np.concatenate([kpts_with_conf, (kpts_with_conf[:, 2] > treshold)[:, None]],
                                                     axis=1)

            all_kpts.append(kpts_with_conf_treshold)
            all_bbox.append(bbox.cpu().numpy())
            #print(kpts_with_conf_treshold)
    return np.array(all_kpts), np.array(all_bbox)

def convert_yolo_to_bbox(bbox, img_width, img_height):
    """
    Convert YOLO format (normalized) bounding box to absolute pixel coordinates.
    YOLO format: x_center, y_center, width, height (all normalized)
    """
    x_center_abs = int(bbox['x_center'] * img_width)
    y_center_abs = int(bbox['y_center'] * img_height)
    width_abs = int(bbox['width'] * img_width)
    height_abs = int(bbox['height'] * img_height)

    # Calculate the top-left and bottom-right corners of the bounding box
    x_min = int(x_center_abs - width_abs / 2)
    y_min = int(y_center_abs - height_abs / 2)
    x_max = int(x_center_abs + width_abs / 2)
    y_max = int(y_center_abs + height_abs / 2)

    return (x_min, y_min, x_max, y_max)


def read_yolo_file(yolo_file_path, return_yolo=False, img_width=1224, img_height=1024):
    with open(yolo_file_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = line.strip().split()
        # convert numbers to float
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)
        # extract class_id integer from string
        class_id = int(re.findall(r'\d+', class_id)[0])
        bbox = None
        if return_yolo:
            bbox = [x_center, y_center, width, height]
        else:
            bbox = convert_yolo_to_bbox({'x_center': x_center,
                                     'y_center': y_center,
                                     'width': width,
                                     'height': height}, img_width=img_width, img_height= img_height)
        bboxes.append({
            'class_id': class_id,
            'bbox': bbox
        })

    return bboxes

def convert_opencv_to_torch(img, transform=None, transform_np=True):
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    if transform_np and transform:
        img = transform(image=img)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
    if not transform_np and transform:
        img = transform(img)
    return img

def convert_torch_to_opencv(img):
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2: (x1, y1, x2, y2) format
        - x1, y1: Coordinates of the top-left corner of the box
        - x2, y2: Coordinates of the bottom-right corner of the box

    Returns:
    iou: float
        Intersection over Union value between the two boxes
    """

    # Unpack the coordinates of the boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Compute the area of the intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    # Compute the area of both bounding boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou
