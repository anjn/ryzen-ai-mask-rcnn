import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import argparse
import numpy as np
import torch
from torch import nn, Tensor
import onnxruntime

from torchvision_mod.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

# https://github.com/pytorch/vision/blob/867521ec82c78160b16eec1c3a02d4cef93723ff/torchvision/models/detection/rpn.py#L88
from torchvision_mod.models.detection.rpn import concat_box_prediction_layers
from torchvision_mod.models.detection.roi_heads import maskrcnn_inference

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
    def __init__(self, model, import_onnx=True, quantize=True):
        super().__init__()
        self.model = model

        if import_onnx:
            self.session = onnxruntime.InferenceSession(f"model/maskrcnn_backbone_rpn{'_quant' if quantize else ''}.onnx")

    def forward(self, images):
        if not hasattr(self, 'session'):
            features = self.model.backbone(images)
            objectness, pred_bbox_deltas = self.model.rpn.head(list(features.values()))

        else:
            output_names = [
                'feature_0', 'feature_1', 'feature_2', 'feature_3',
                'cls_logits_0', 'cls_logits_1', 'cls_logits_2', 'cls_logits_3', 'cls_logits_4', 
                'bbox_pred_0', 'bbox_pred_1', 'bbox_pred_2', 'bbox_pred_3', 'bbox_pred_4', 
            ]

            outputs = self.session.run(output_names, {'input': images.cpu().numpy()})

            device = images.device
            outputs = [torch.from_numpy(o).to(device) for o in outputs]

            objectness = [outputs[i] for i in range(4, 9)]
            pred_bbox_deltas = [outputs[i] for i in range(9, 14)]

            features = {}
            for i in range(4):
                features[f"{i}"] = outputs[i]
            #features["pool"] = torch.empty((*features["0"].shape[0:2], *objectness[4].shape[2:4])).to(device)

        return features, objectness, pred_bbox_deltas

class MaskRCNNBoxProposal(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_tensors, features, objectness, pred_bbox_deltas):
        anchors = self.forward_anchor_generator(image_tensors, self.grid_sizes) 
        anchors = [anchor.to(image_tensors.device) for anchor in anchors]

        image_sizes = [self.image_sizes for _ in range(image_tensors.shape[0])]

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, _ = self.model.rpn.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)

        box_features = self.model.roi_heads.box_roi_pool(features, proposals, image_sizes)

        return proposals, box_features

    def forward_anchor_generator(self, image_tensors, grid_sizes) -> List[Tensor]:
        strides = [
            [
                torch.empty((), dtype=torch.int64).fill_(self.image_sizes[0] // g[0]),
                torch.empty((), dtype=torch.int64).fill_(self.image_sizes[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        self.model.rpn.anchor_generator.cell_anchors = [cell_anchor for cell_anchor in self.model.rpn.anchor_generator.cell_anchors]
        anchors_over_all_feature_maps = self.model.rpn.anchor_generator.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(image_tensors.shape[0]):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors

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
    def __init__(self, model, save_io=False, save_io_dir="model_io"):
        super().__init__()
        self.model = model
        self.save_io = save_io
        self.save_io_dir = save_io_dir

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
    
    def forward(self, images, img_infos, export=False):#, export_dir=Path("export_onnx")):
        ids = [info['id'] for info in img_infos]

        if self.save_io:
            assert len(images) == 1
            #print(f"{img_infos=}")
            for id, image in zip(ids, images):
                Path(f"{self.save_io_dir}/{id}").mkdir(parents=True, exist_ok=True)
                input = image.view(1, *images[0].shape).to('cpu').detach().numpy().copy()
                #print(f"{input.shape=}")
                np.save(f"{self.save_io_dir}/{id}/input.npy", input)

        # Preprocess
        with self.timer_preprocess:
            images, original_image_sizes = self.preprocess(images)

        if self.save_io:
            np.save(f"{self.save_io_dir}/{ids[0]}/input_backbone.npy", images.tensors.cpu().numpy())

        # Backbone + RPN
        with self.timer_backbone_rpn:
            features, objectness, pred_bbox_deltas = self.backbone_rpn(images.tensors)

        #for feat in list(features.values()):
        #    print(f"{feat.shape = }")
        #for obj in objectness:
        #    print(f"{obj.shape = }")
        #for box in pred_bbox_deltas:
        #    print(f"{box.shape = }")

        # Box proposals
        with self.timer_box_proposal:
            self.box_proposal.image_sizes = images.tensors.shape[-2:]
            #self.box_proposal.grid_sizes = [feature_map.shape[-2:] for feature_map in list(features.values())]
            self.box_proposal.grid_sizes = [feature_map.shape[-2:] for feature_map in objectness]

            proposals, box_features = self.box_proposal(images.tensors, features, objectness, pred_bbox_deltas)

            #print(f"{len(proposals)=}")
            #print(f"{box_features.shape=}")

            if export:
                self.box_proposal = self.box_proposal.to('cpu')

                torch.onnx.export(
                        self.box_proposal,
                        (images.tensors, features, objectness, pred_bbox_deltas),
                        str(export_dir / "maskrcnn_box_proposal.onnx"),
                        export_params=True,
                        opset_version=13,  # Recommended opset
                        input_names=['input', 'features', 'objectness', 'pred_bbox_deltas'],
                        output_names=['proposals', 'box_features'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'},
                                      'objectness': {0: 'batch_size'}, 'pred_bbox_deltas': {0: 'batch_size'},
                                      'proposals': {0: 'proposal_size'}, 'box_features': {0: 'proposal_size'}},
                        )

        # Box prediction
        with self.timer_box_predictor:
            class_logits, box_regression = self.box_predictor(box_features)

            #print(f"{box_features.shape=}")
            #print(f"{class_logits.shape=}")
            #print(f"{box_regression.shape=}")

            if export:
                torch.onnx.export(
                        self.box_predictor,
                        box_features,
                        str(export_dir / "maskrcnn_box_predictor.onnx"),
                        export_params=True,
                        opset_version=13,  # Recommended opset
                        input_names=['input'],
                        output_names=['class_logits', 'box_regression'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}, 'box_regression': {0: 'batch_size'}},
                        )

        # Box postprocess
        with self.timer_box_postprocess:
            boxes, scores, labels = self.box_postprocess(proposals, class_logits, box_regression, images.image_sizes)

        # Mask proposals
        with self.timer_mask_proposal:
            mask_features = self.mask_proposal(features, boxes, images.image_sizes)

            #print(f"{features.keys()=}")
            #print(f"{len(boxes)=}")
            #print(f"{mask_features.shape=}")

            if export:
                torch.onnx.export(
                        self.mask_proposal,
                        (features, boxes, images.image_sizes),
                        str(export_dir / "maskrcnn_mask_proposal.onnx"),
                        export_params=True,
                        opset_version=13,  # Recommended opset
                        input_names=['features', 'boxes', 'image_sizes'],
                        output_names=['mask_features'],
                        dynamic_axes={'features': {0: 'batch_size'}, 'image_sizes': {0: 'batch_size'}, 'boxes': {0: 'proposal_size'}, 'mask_features': {0: 'proposal_size'}},
                        )

        # Mask prediction
        with self.timer_mask_predictor:
            mask_logits = self.mask_predictor(mask_features)

            if export:
                torch.onnx.export(
                        self.mask_predictor,
                        mask_features,
                        str(export_dir / "maskrcnn_mask_predictor.onnx"),
                        export_params=True,
                        opset_version=13,  # Recommended opset
                        input_names=['input'],
                        output_names=['mask_logits'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'mask_logits': {0: 'batch_size'}},
                        )

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
        #print(f"{boxes=}")
        #print(f"{labels=}")
        #print(f"{scores=}")

        #print(f"{boxes[0].shape=}")
        #print(f"{labels[0].shape=}")
        #print(f"{scores[0].shape=}")
        #print(f"{mask_probs[0].shape=}")

        if self.save_io:
            from copy import deepcopy
            save_detections = self.model.transform.postprocess(deepcopy(detections), images.image_sizes, images.image_sizes)
            for id, pred in zip(ids, save_detections):
                #print(f"{pred['masks'].shape=}")
                np.save(f"{self.save_io_dir}/{id}/boxes.npy", pred["boxes"].cpu().numpy())
                np.save(f"{self.save_io_dir}/{id}/scores.npy", pred["scores"].cpu().numpy())
                np.save(f"{self.save_io_dir}/{id}/labels.npy", pred["labels"].cpu().numpy())
                np.save(f"{self.save_io_dir}/{id}/masks.npy", pred["masks"].cpu().numpy())

        if "org_height" in img_infos[0]:
            org_sizes = [(info["org_height"], info["org_width"]) for info in img_infos]
            detections = self.model.transform.postprocess(detections, images.image_sizes, org_sizes)
            for info, org in zip(img_infos, org_sizes):
                info["height"] = org[0]
                info["width"] = org[1]
        else:
            detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        return detections
    

def main(args):
    # Device configuration
    if args.device:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    )
    model = CustomMaskRCNN(model)

    model = model.to(device)
    model.eval()

    #input = [torch.rand((3, 800, 1056)).to(device) for _ in range(1)]
    input = torch.from_numpy(np.load("model_io/397133/input.npy"))
    img_infos = [{"id": 397133, "height": input.shape[2], "width": input.shape[3]} for _ in range(input.shape[0])]

    with torch.no_grad():
        pred = model(input, img_infos)[0]

    model.print()

    golden = {
        "boxes" : np.load("model_io/397133/boxes.npy"),
        "labels": np.load("model_io/397133/labels.npy"),
        "scores": np.load("model_io/397133/scores.npy"),
        "masks" : np.load("model_io/397133/masks.npy"),
    }

    print(f"{golden['boxes'].shape=}")
    print(f"{golden['labels'].shape=}")
    print(f"{golden['scores'].shape=}")
    print(f"{golden['masks'].shape=}")
    print(f"{pred['boxes'].shape=}")
    print(f"{pred['labels'].shape=}")
    print(f"{pred['scores'].shape=}")
    print(f"{pred['masks'].shape=}")
    
    print(f"{np.all(np.abs(pred['labels'].cpu().numpy() - golden['labels']) < 1e-4) = }")
    print(f"{np.all(np.abs(pred['boxes' ].cpu().numpy() - golden['boxes' ]) < 1e-3) = }")
    print(f"{np.all(np.abs(pred['scores'].cpu().numpy() - golden['scores']) < 1e-4) = }")
    print(f"{np.all(np.abs(pred['masks' ].cpu().numpy() - golden['masks' ]) < 1e-3) = }")

    #torch.onnx.export(
    #    model,
    #    (input), 
    #    "maskrcnn_custom.onnx",
    #    opset_version = 17,
    #    input_names=['input'],
    #    output_names = ['boxes', 'labels', 'scores', 'masks'],
    #    dynamic_axes={
    #        'input':  {0: 'batch_size', 2: 'height', 3: 'width'},
    #        'boxes':  {0: 'batch_size', 1: 'num'},
    #        'labels': {0: 'batch_size', 1: 'num'},
    #        'scores': {0: 'batch_size', 1: 'num'},
    #        'masks':  {0: 'batch_size', 1: 'num'},
    #    },
    #)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model on COCO dataset')
    parser.add_argument('--device', type=str, default=None,
                      help='')
    args = parser.parse_args()
    main(args)
