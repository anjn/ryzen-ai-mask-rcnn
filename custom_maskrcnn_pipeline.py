import os
import sys
import time
import collections
import queue
import threading
import argparse
from typing import Dict, List, Optional, Tuple, Union
import pickle

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from custom_maskrcnn_model import CodeTimer, CustomMaskRCNN
from coco_utils import COCOEvalDataset, coco_collate_fn, get_coco_category_mapping, get_coco_label_mapping


def preprocess_func(img_org, frame_id, device):
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

    # Scale (fit to width)
    org_size = img_rgb.shape[0:2]
    scale = 960 / org_size[1]
    new_size = (960, int(org_size[0] * scale))
    img_rgb = cv2.resize(img_rgb, new_size)

    # Letterbox
    src_size = img_rgb.shape[0:2]
    dst_size = [540, 960]
    assert src_size[1] == dst_size[1]
    letterbox = [range((d-s)//2, s + (d-s)//2) for s,d in zip(src_size, dst_size)]

    # Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = np.zeros((1, *dst_size, 3), dtype=np.float32)
    img[0, letterbox[0], :, :] = (img_rgb / 255 - mean) / std

    # NHWC to NCHW
    img = img.transpose((0, 3, 1, 2))

    # Conver to torch tensor
    img = torch.from_numpy(img).to(device)

    return ((img_org, frame_id), (scale, letterbox), img)


def backbone_func(img_info, prepro_info, img, model):
    with torch.no_grad():
        features, objectness, pred_bbox_deltas = model.backbone(img)

    return (img_info, prepro_info, img, features, objectness, pred_bbox_deltas)


class NoOutputException(Exception):
    pass


def sort_frame_func(img_info, prepro_info, img, features, objectness, pred_bbox_deltas):
    if not hasattr(sort_frame_func, 'init'):
        sort_frame_func.buffer = []
        sort_frame_func.next_frame = 0
        sort_frame_func.init = True

    sort_frame_func.buffer.append((img_info, prepro_info, img, features, objectness, pred_bbox_deltas))

    for i, item in enumerate(sort_frame_func.buffer):
        if item[1] == sort_frame_func.next_frame:
            sort_frame_func.next_frame += 1
            del sort_frame_func.buffer[i]
            return item
    
    raise NoOutputException()


def other_func(img_info, prepro_info, img, features, objectness, pred_bbox_deltas, model):
    with torch.no_grad():
        model.box_proposal.image_sizes = img.shape[-2:]
        model.box_proposal.grid_sizes = [feature_map.shape[-2:] for feature_map in objectness]
        proposals, box_features = model.box_proposal(img, features, objectness, pred_bbox_deltas)
        class_logits, box_regression = model.box_predictor(box_features)
        boxes, scores, labels = model.box_postprocess(proposals, class_logits, box_regression, [img.shape[2:4]])
        mask_features = model.mask_proposal(features, boxes, [img.shape[2:4]])
        mask_logits = model.mask_predictor(mask_features)
        mask_probs = model.mask_postprocess(mask_logits, labels)

    # Create result
    detections: List[Dict[str, torch.Tensor]] = []
    num_images = len(boxes)
    for i in range(num_images):
        detections.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
                "masks": mask_probs[i],
            }
        )

    detections = model.model.transform.postprocess(detections, [img.shape[2:4]], [img_info[0].shape[0:2]])

    return (img_info, prepro_info, detections)


def visualize_func(img_info, prepro_info, detections, args, label_mapping):
    img_org, frame_id = img_info
    scale, letterbox = prepro_info

    # Get prediction components
    boxes = detections[0]['boxes'].cpu().numpy()
    labels = detections[0]['labels'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()
    masks = detections[0]['masks'].cpu().numpy()
    
    # Generate random colors for visualization
    colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
    
    # Draw each detection
    for box, label, score, mask, color in zip(boxes, labels, scores, masks, colors):
        if score < args.visualize_threshold:
            continue
            
        # Convert color to BGR (for OpenCV)
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_org, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Get category name
        category_name = label_mapping.get(label.item(), "Unknown")
        
        # Draw label and score
        label_text = f'{category_name}: {score:.2f}'
        cv2.putText(img_org, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        # Apply mask
        mask_np = mask[0] > 0.5
        mask_color = (color[:3] * 255 * 0.5).astype(np.uint8)
        mask_overlay = np.zeros_like(img_org)
        for c in range(3):
            mask_overlay[:, :, c] = mask_color[c]
        mask_overlay = cv2.bitwise_and(mask_overlay, mask_overlay, mask=mask_np.astype(np.uint8))
        img_org = cv2.addWeighted(img_org, 1, mask_overlay, 0.5, 0)

    return (img_org, )


def run_task(task_func, input_queue, output_queue, args=()):
    while True:
        try:
            inputs = input_queue.get(timeout=3) # 3 sec
            outputs = task_func(*inputs, *args)
            output_queue.put(outputs)
        except NoOutputException:
            continue
        except queue.Empty:
            print("finish task", task_func)
            break


def start_threads(task_func, input_queue, output_queue, args=(), num_threads=1):
    threads = []
    for _ in range(num_threads):
        thr = threading.Thread(target=run_task, args=(task_func, input_queue, output_queue, args))
        thr.start()
        threads.append(thr)
    return threads


# https://stackoverflow.com/a/54539292
class FPS:
    def __init__(self, average_of=60):
        self.frametimestamps = collections.deque(maxlen=average_of)

    def __call__(self):
        self.frametimestamps.append(time.time())
        return self.get()
    
    def get(self):
        if len(self.frametimestamps) > 1:
            elapsed_time = self.frametimestamps[-1] - self.frametimestamps[0]
            if elapsed_time > 0:
                return (len(self.frametimestamps) - 1) / elapsed_time
            else:
                return 0.0
        else:
            return 0.0


def main(args):
    # Device configuration
    if args.device:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.ann_file is not None:
        # Initialize COCO dataset
        coco_gt = COCO(args.ann_file)
        # Get category mapping
        label_mapping = get_coco_label_mapping(coco_gt)
        # Save
        with open("sample/label_mapping.pickle", mode="wb") as f:
            pickle.dump(label_mapping, f)
    else:
        # Load
        with open("sample/label_mapping.pickle", mode="rb") as f:
            label_mapping = pickle.load(f)

    # Load model and weights
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    )
    model = CustomMaskRCNN(model, args)
    
    model = model.to(device)
    model.eval()

    img_org = cv2.imread("sample/dance.jpg")

    with CodeTimer("preprocess", True):
        outputs = preprocess_func(img_org, 0, device)

    with CodeTimer("backbone", True):
        outputs = backbone_func(*outputs, model)

    with CodeTimer("other", True):
        outputs = other_func(*outputs, model)

    with CodeTimer("visualize", True):
        outputs = visualize_func(*outputs, args, label_mapping)

    cv2.imwrite("output.jpg", outputs[0])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model on COCO dataset')
    parser.add_argument('--ann_file', type=str, default=None,
                      help='Path to COCO annotation file')
    parser.add_argument('--visualize_threshold', type=float, default=0.3,
                      help='Score threshold for visualization (default: 0.3)')
    parser.add_argument('--device', type=str, default='cpu',
                      help='')
    parser.add_argument('--onnx_backbone', type=str, default=None,
                      help='')
    #parser.add_argument('--onnx_box_proposal', type=str, default=None,
    #                  help='')
    #parser.add_argument('--onnx_box_predictor', type=str, default=None,
    #                  help='')
    #parser.add_argument('--onnx_mask_predictor', type=str, default=None,
    #                  help='')
    parser.add_argument('--onnx_ep', type=str, default='cpu',
                      help='')
    args = parser.parse_args()

    args.save_io = False
    args.save_io_dir = "."
    args.export = False
    args.export_dir = "."
    args.onnx_box_proposal = None
    args.onnx_box_predictor = None
    args.onnx_mask_predictor = None

    main(args)
