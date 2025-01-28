import time
from typing import Dict, List, Optional, Tuple, Union

import argparse
import torch
from torch import nn, Tensor

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

# https://github.com/pytorch/vision/blob/867521ec82c78160b16eec1c3a02d4cef93723ff/torchvision/models/detection/rpn.py#L88
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import maskrcnn_inference

class MaskRCNNPreProcess(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, _ = self.model.transform(images)

        return images, original_image_sizes

class MaskRCNNBackboneRPN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        features = self.model.backbone(images)
        objectness, pred_bbox_deltas = self.model.rpn.head(list(features.values()))

        return features, objectness, pred_bbox_deltas

class MaskRCNNBoxProposal(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, features, objectness, pred_bbox_deltas, image_shapes):
        anchors = self.model.rpn.anchor_generator(images, list(features.values()))

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, _ = self.model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        box_features = self.model.roi_heads.box_roi_pool(features, proposals, image_shapes)

        return proposals, box_features

class MaskRCNNMaskProposal(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, boxes, image_shapes):
        mask_features = self.model.roi_heads.mask_roi_pool(features, boxes, image_shapes)

        return mask_features

class MaskRCNNBoxPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, box_features):
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

        return class_logits, box_regression

class MaskRCNNMaskPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mask_features):
        mask_features = self.model.roi_heads.mask_head(mask_features)
        mask_logits = self.model.roi_heads.mask_predictor(mask_features)

        return mask_logits

class MaskRCNNBoxPostProcess(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, proposals, class_logits, box_regression, image_shapes):
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

        return boxes, scores, labels

class MaskRCNNMaskPostProcess(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mask_logits, labels):
        mask_probs = maskrcnn_inference(mask_logits, labels)

        return mask_probs

class CodeTimer:
    def __init__(self, name):
        self.name = name
        self.cuda = torch.cuda.is_available()
        self.reset()
    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.start = time.time()
    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.elapsed += time.time() - self.start
    def reset(self):
        self.elapsed = 0
    def print(self):
        print(f"{self.name} : {self.elapsed:.3f} sec")


class CustomMaskRCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.preprocess = MaskRCNNPreProcess(model)
        self.backbone_rpn = MaskRCNNBackboneRPN(model)
        self.box_proposal = MaskRCNNBoxProposal(model)
        self.box_predictor = MaskRCNNBoxPredictor(model)
        self.box_postprocess = MaskRCNNBoxPostProcess(model)
        self.mask_proposal = MaskRCNNMaskProposal(model)
        self.mask_predictor = MaskRCNNMaskPredictor(model)
        self.mask_postprocess = MaskRCNNMaskPostProcess(model)

        self.timer_preprocess = CodeTimer("preprocess")
        self.timer_backbone_rpn = CodeTimer("backbone_rpn")
        self.timer_box_proposal = CodeTimer("box_proposal")
        self.timer_box_predictor = CodeTimer("box_predictor")
        self.timer_box_postprocess = CodeTimer("box_postprocess")
        self.timer_mask_proposal = CodeTimer("mask_proposal")
        self.timer_mask_predictor = CodeTimer("mask_predictor")
        self.timer_mask_postprocess = CodeTimer("mask_postprocess")

        self.reset()

    def reset(self):
        self.timer_preprocess.reset()
        self.timer_backbone_rpn.reset()
        self.timer_box_proposal.reset()
        self.timer_box_predictor.reset()
        self.timer_box_postprocess.reset()
        self.timer_mask_proposal.reset()
        self.timer_mask_predictor.reset()
        self.timer_mask_postprocess.reset()

    def print(self):
        print("\nTime:")
        self.timer_preprocess.print()
        self.timer_backbone_rpn.print()
        self.timer_box_proposal.print()
        self.timer_box_predictor.print()
        self.timer_box_postprocess.print()
        self.timer_mask_proposal.print()
        self.timer_mask_predictor.print()
        self.timer_mask_postprocess.print()
    
    def forward(self, images):
        # Preprocess
        with self.timer_preprocess:
            images, original_image_sizes = self.preprocess(images)

        # Backbone + RPN
        with self.timer_backbone_rpn:
            features, objectness, pred_bbox_deltas = self.backbone_rpn(images.tensors)

        # Box proposals
        with self.timer_box_proposal:
            proposals, box_features = self.box_proposal(images, features, objectness, pred_bbox_deltas, images.image_sizes)

        # Box prediction
        with self.timer_box_predictor:
            class_logits, box_regression = self.box_predictor(box_features)

        # Box postprocess
        with self.timer_box_postprocess:
            boxes, scores, labels = self.box_postprocess(proposals, class_logits, box_regression, images.image_sizes)

        # Mask proposals
        with self.timer_mask_proposal:
            mask_features = self.mask_proposal(features, boxes, images.image_sizes)

        # Mask prediction
        with self.timer_mask_predictor:
            mask_logits = self.mask_predictor(mask_features)

        # Mask postprocess
        with self.timer_mask_postprocess:
            mask_probs = self.mask_postprocess(mask_logits, labels)
        
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

        detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        return detections


def main(args):
    # Device configuration
    if args.device:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = maskrcnn_resnet50_fpn_v2()
    model = CustomMaskRCNN(model)

    model = model.to(device)
    model.eval()

    input = [torch.rand((3, 800, 1056)).to(device) for _ in range(1)]

    # Warm up
    with torch.no_grad():
        model(input)

    model.reset()

    with torch.no_grad():
        model(input)

    model.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model on COCO dataset')
    parser.add_argument('--device', type=str, default=None,
                      help='')
    args = parser.parse_args()
    main(args)
