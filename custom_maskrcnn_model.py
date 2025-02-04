import os
from re import X
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import argparse
import numpy as np
import torch
from torch import nn, Tensor
import onnxruntime

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import maskrcnn_inference


def create_session(onnx_ep, model_path, model_name, num_of_dpu_runners=4, enable_analyzer=False):
    print(f"Load {model_path} with {onnx_ep} EP")

    if onnx_ep == "cpu":
        return onnxruntime.InferenceSession(model_path)

    elif onnx_ep == 'vai':
        cache_dir = os.path.join(os.getcwd(),  r'cache')

        return onnxruntime.InferenceSession(
            # 量子化済み ONNX モデルを指定
            model_path,
            # NPU を使用して推論を実行するように指示
            providers = ['VitisAIExecutionProvider'],
            # NPU 実行に関するオプション
            provider_options = [{
                'config_file': f"{os.environ['VAIP_CONFIG_HOME']}/vaip_config.json",
                'num_of_dpu_runners': num_of_dpu_runners,
                'cacheDir': cache_dir,
                'cacheKey': model_name,
                'ai_analyzer_visualization': enable_analyzer,
                'ai_analyzer_profiling': enable_analyzer,
            }]
        )

    else:
        raise ValueError(f"Invalid onnxruntime execution provider : {onnx_ep}")


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


class MaskRCNNBackbone(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model

        if args.onnx_backbone is not None:
            self.session = create_session(args.onnx_ep, args.onnx_backbone, "maskrcnn_backbone")

    def forward(self, images):
        if not hasattr(self, 'session'):
            features = self.model.backbone(images)
            objectness, pred_bbox_deltas = self.model.rpn.head(list(features.values()))

        else:
            output_names = [
                'features_0',
                'features_1',
                'features_2',
                'features_3',
                'objectness_0',
                'objectness_1',
                'objectness_2',
                'objectness_3',
                'objectness_4', 
                'pred_bbox_deltas_0',
                'pred_bbox_deltas_1',
                'pred_bbox_deltas_2',
                'pred_bbox_deltas_3',
                'pred_bbox_deltas_4', 
            ]
            outputs = self.session.run(output_names, {'input': images.cpu().numpy()})

            device = images.device
            outputs = [torch.from_numpy(o).to(device) for o in outputs]

            objectness = [outputs[i] for i in range(4, 9)]
            pred_bbox_deltas = [outputs[i] for i in range(9, 14)]

            features = {}
            for i in range(4):
                features[f"{i}"] = outputs[i]

        return features, objectness, pred_bbox_deltas


class MaskRCNNBoxProposal(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model

        if args.onnx_box_proposal is not None:
            print(f"Load {args.onnx_box_proposal}")
            self.session = onnxruntime.InferenceSession(args.onnx_box_proposal)

    def forward(self, image_tensors, features, objectness, pred_bbox_deltas):
        if not hasattr(self, 'session'):
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

        else:
            inputs = {
                'features_0' : features["0"].cpu().numpy(),
                'features_1' : features["1"].cpu().numpy(),
                'features_2' : features["2"].cpu().numpy(),
                'features_3' : features["3"].cpu().numpy(),
                'objectness_0' : objectness[0].cpu().numpy(),
                'objectness_1' : objectness[1].cpu().numpy(),
                'objectness_2' : objectness[2].cpu().numpy(),
                'objectness_3' : objectness[3].cpu().numpy(),
                'objectness_4' : objectness[4].cpu().numpy(),
                'pred_bbox_deltas_0' : pred_bbox_deltas[0].cpu().numpy(),
                'pred_bbox_deltas_1' : pred_bbox_deltas[1].cpu().numpy(),
                'pred_bbox_deltas_2' : pred_bbox_deltas[2].cpu().numpy(),
                'pred_bbox_deltas_3' : pred_bbox_deltas[3].cpu().numpy(),
                'pred_bbox_deltas_4' : pred_bbox_deltas[4].cpu().numpy(),
            }
            output_names = [
                'proposals', 'box_features'
            ]

            outputs = self.session.run(output_names, inputs)

            device = image_tensors.device
            outputs = [torch.from_numpy(o).to(device) for o in outputs]

            proposals = [outputs[0]]
            box_features = outputs[1]

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


class MaskRCNNBoxPredictor(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model

        if args.onnx_box_predictor is not None:
            self.session = create_session(args.onnx_ep, args.onnx_box_predictor, "maskrcnn_box_predictor")

    def forward(self, box_features):
        if not hasattr(self, 'session'):
            box_features = self.model.roi_heads.box_head(box_features)
            class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

        else:
            inputs = { 'box_features' : box_features.cpu().numpy(), }
            output_names = [ 'class_logits', 'box_regression' ]
            outputs = self.session.run(output_names, inputs)

            device = box_features.device
            outputs = [torch.from_numpy(o).to(device) for o in outputs]

            class_logits = outputs[0]
            box_regression = outputs[1]

        return class_logits, box_regression


class MaskRCNNMaskProposal(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, boxes, image_shapes):
        mask_features = self.model.roi_heads.mask_roi_pool(features, boxes, image_shapes)

        return mask_features


class MaskRCNNMaskPredictor(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model

        if args.onnx_mask_predictor is not None:
            self.session = create_session(args.onnx_ep, args.onnx_mask_predictor, "maskrcnn_mask_predictor")

    def forward(self, mask_features):
        if not hasattr(self, 'session'):
            mask_features = self.model.roi_heads.mask_head(mask_features)
            mask_logits = self.model.roi_heads.mask_predictor(mask_features)

        else:
            inputs = { 'mask_features' : mask_features.cpu().numpy(), }
            output_names = [ 'mask_logits', ]
            outputs = self.session.run(output_names, inputs)

            device = mask_features.device
            outputs = [torch.from_numpy(o).to(device) for o in outputs]

            mask_logits = outputs[0]

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


# https://stackoverflow.com/a/52749808
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
    def print(self, test_num):
        print(f"{self.name} : {self.elapsed/test_num:.3f} sec")


class CustomMaskRCNN(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.save_io = args.save_io
        self.save_io_dir = args.save_io_dir
        self.export = args.export
        self.export_dir = Path(args.export_dir)

        self.preprocess = MaskRCNNPreProcess(model)
        self.backbone = MaskRCNNBackbone(model, args)
        self.box_proposal = MaskRCNNBoxProposal(model, args)
        self.box_predictor = MaskRCNNBoxPredictor(model, args)
        self.box_postprocess = MaskRCNNBoxPostProcess(model)
        self.mask_proposal = MaskRCNNMaskProposal(model)
        self.mask_predictor = MaskRCNNMaskPredictor(model, args)
        self.mask_postprocess = MaskRCNNMaskPostProcess(model)

        self.timer_preprocess = CodeTimer("preprocess")
        self.timer_backbone = CodeTimer("backbone")
        self.timer_box_proposal = CodeTimer("box_proposal")
        self.timer_box_predictor = CodeTimer("box_predictor")
        self.timer_box_postprocess = CodeTimer("box_postprocess")
        self.timer_mask_proposal = CodeTimer("mask_proposal")
        self.timer_mask_predictor = CodeTimer("mask_predictor")
        self.timer_mask_postprocess = CodeTimer("mask_postprocess")

        self.reset()

    def reset(self):
        self.timer_preprocess.reset()
        self.timer_backbone.reset()
        self.timer_box_proposal.reset()
        self.timer_box_predictor.reset()
        self.timer_box_postprocess.reset()
        self.timer_mask_proposal.reset()
        self.timer_mask_predictor.reset()
        self.timer_mask_postprocess.reset()

    def print(self):
        print("\nTime:")
        self.timer_preprocess.print()
        self.timer_backbone.print()
        self.timer_box_proposal.print()
        self.timer_box_predictor.print()
        self.timer_box_postprocess.print()
        self.timer_mask_proposal.print()
        self.timer_mask_predictor.print()
        self.timer_mask_postprocess.print()
    
    def forward(self, images, img_infos):
        ids = [info['id'] for info in img_infos]

        if self.export:
            os.makedirs(str(self.export_dir), exist_ok=True)

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

        # Backbone + FPN
        with self.timer_backbone:
            features, objectness, pred_bbox_deltas = self.backbone(images.tensors)

        if self.export:
            #for feat in list(features.values()):
            #    print(f"{feat.shape = }")
            #for obj in objectness:
            #    print(f"{obj.shape = }")
            #for box in pred_bbox_deltas:
            #    print(f"{box.shape = }")

            torch.onnx.export(
                    self.backbone,
                    (images.tensors),
                    str(self.export_dir / "maskrcnn_backbone.onnx"),
                    export_params=True,
                    opset_version=13,
                    input_names=[
                        'input',
                    ],
                    output_names=[
                        'features_0',
                        'features_1',
                        'features_2',
                        'features_3',
                        'features_4',
                        'objectness_0',
                        'objectness_1',
                        'objectness_2',
                        'objectness_3',
                        'objectness_4',
                        'pred_bbox_deltas_0',
                        'pred_bbox_deltas_1',
                        'pred_bbox_deltas_2',
                        'pred_bbox_deltas_3',
                        'pred_bbox_deltas_4',
                    ],
                    dynamic_axes={})

        # Box proposals
        with self.timer_box_proposal:
            self.box_proposal.image_sizes = images.tensors.shape[-2:]
            self.box_proposal.grid_sizes = [feature_map.shape[-2:] for feature_map in objectness]
            proposals, box_features = self.box_proposal(images.tensors, features, objectness, pred_bbox_deltas)

        if self.export:
            #print(f"{len(proposals)=}")
            #print(f"{proposals[0].shape=}")
            #print(f"{box_features.shape=}")

            torch.onnx.export(
                    self.box_proposal,
                    (images.tensors, features, objectness, pred_bbox_deltas),
                    str(self.export_dir / "maskrcnn_box_proposal.onnx"),
                    export_params=True,
                    opset_version=13,
                    input_names=[
                        'input',
                        'features_0',
                        'features_1',
                        'features_2',
                        'features_3',
                        'objectness_0',
                        'objectness_1',
                        'objectness_2',
                        'objectness_3',
                        'objectness_4',
                        'pred_bbox_deltas_0',
                        'pred_bbox_deltas_1',
                        'pred_bbox_deltas_2',
                        'pred_bbox_deltas_3',
                        'pred_bbox_deltas_4',
                    ],
                    output_names=['proposals', 'box_features'],
                    dynamic_axes={})

        # Box prediction
        with self.timer_box_predictor:
            class_logits, box_regression = self.box_predictor(box_features)

        if self.save_io:
            np.save(f"{self.save_io_dir}/{ids[0]}/box_features.npy", box_features.cpu().numpy())

        if self.export:
            #print("Export box prediction")
            #print(f"{class_logits.shape=}")
            #print(f"{box_regression.shape=}")

            torch.onnx.export(
                    self.box_predictor,
                    box_features,
                    str(self.export_dir / "maskrcnn_box_predictor.onnx"),
                    export_params=True,
                    opset_version=13,
                    input_names=['box_features'],
                    output_names=['class_logits', 'box_regression'],
                    dynamic_axes={
                        #'box_features': {0: 'proposal_size'},
                        #'class_logits': {0: 'proposal_size'},
                        #'box_regression': {0: 'proposal_size'},
                    })

        # Box postprocess
        with self.timer_box_postprocess:
            boxes, scores, labels = self.box_postprocess(proposals, class_logits, box_regression, images.image_sizes)

        if self.export:
            #print(f"{images.image_sizes=}")
            #print(f"{boxes[0].shape=}")
            #print(f"{scores[0].shape=}")
            #print(f"{labels[0].shape=}")

            torch.onnx.export(
                    self.box_postprocess,
                    (proposals, class_logits, box_regression, images.image_sizes),
                    str(self.export_dir / "maskrcnn_box_postprocess.onnx"),
                    export_params=True,
                    opset_version=13,
                    input_names=['proposals', 'class_logits', 'box_regression', 'image_sizes'],
                    output_names=['boxes', 'scores', 'labels'],
                    dynamic_axes={
                        #'proposals': {0: 'proposal_size'},
                        #'class_logits': {0: 'proposal_size'},
                        #'box_features': {0: 'proposal_size'},
                        'boxes': {0: 'detection_size'},
                        'scores': {0: 'detection_size'},
                        'labels': {0: 'detection_size'},
                    })

        # Mask proposals
        with self.timer_mask_proposal:
            mask_features = self.mask_proposal(features, boxes, images.image_sizes)

        if self.export:
            #print(f"{features.keys()=}")
            #print(f"{len(boxes)=}")
            #print(f"{mask_features.shape=}")

            torch.onnx.export(
                    self.mask_proposal,
                    (features, boxes, images.image_sizes),
                    str(self.export_dir / "maskrcnn_mask_proposal.onnx"),
                    export_params=True,
                    opset_version=13,
                    input_names=['features', 'boxes', 'image_sizes'],
                    output_names=['mask_features'],
                    dynamic_axes={
                        #'features': {0: 'batch_size'},
                        #'image_sizes': {0: 'batch_size'},
                        'boxes': {0: 'detection_size'},
                        'mask_features': {0: 'detection_size'},
                    })

        # Mask prediction
        with self.timer_mask_predictor:
            mask_logits = self.mask_predictor(mask_features)

        if self.save_io:
            np.save(f"{self.save_io_dir}/{ids[0]}/mask_features.npy", mask_features.cpu().numpy())

        if self.export:
            torch.onnx.export(
                    self.mask_predictor,
                    mask_features,
                    str(self.export_dir / "maskrcnn_mask_predictor.onnx"),
                    export_params=True,
                    opset_version=13,
                    input_names=['mask_features'],
                    output_names=['mask_logits'],
                    dynamic_axes={
                        'mask_features': {0: 'detection_size'},
                        'mask_logits': {0: 'detection_size'},
                    })

        # Mask postprocess
        with self.timer_mask_postprocess:
            mask_probs = self.mask_postprocess(mask_logits, labels)

        if self.export:
            torch.onnx.export(
                    self.mask_postprocess,
                    (mask_logits, labels),
                    str(self.export_dir / "maskrcnn_mask_postprocess.onnx"),
                    export_params=True,
                    opset_version=13,
                    input_names=['mask_logits', 'labels'],
                    output_names=['mask_probs'],
                    dynamic_axes={
                        'mask_logits': {0: 'detection_size'},
                        'mask_probs': {0: 'detection_size'},
                    })
        
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

    # TorchVisionの学習済みモデルを読み込む
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    )

    # 改造版のモデルを作る
    model = CustomMaskRCNN(model, args)

    # 入力データを準備
    if args.input == "random":
        input = torch.rand((1, 3, 800, 1056)).to(device)
        input_id = 0
    else:
        input = torch.from_numpy(np.load(f"model_io/{args.input}/input.npy"))
        input_id = int(args.input)

    img_infos = [{"id": input_id, "height": input.shape[2], "width": input.shape[3]} for _ in range(input.shape[0])]

    # 推論を実行
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Warm up
        if args.warm_up:
            pred = model(input, img_infos)[0]
            model.reset()

        for _ in range(args.test_num):
            pred = model(input, img_infos)[0]

    # 測定した実行時間を出力
    model.print(args.test_num)

    # ランダム入力でない場合は期待値との比較を行う
    if args.input != "random":
        golden = {
            "boxes" : np.load(f"model_io/{args.input}/boxes.npy"),
            "labels": np.load(f"model_io/{args.input}/labels.npy"),
            "scores": np.load(f"model_io/{args.input}/scores.npy"),
            "masks" : np.load(f"model_io/{args.input}/masks.npy"),
        }
    
        print(f"\n{golden['boxes'].shape=}")
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model on COCO dataset')
    parser.add_argument('--device', type=str, default=None,
                      help='')
    parser.add_argument('--input', type=str, default='random',
                      help='')
    parser.add_argument('--export', action='store_true',
                      help='')
    parser.add_argument('--export_dir', type=str, default='model',
                      help='')
    parser.add_argument('--onnx_backbone', type=str, default=None,
                      help='')
    parser.add_argument('--onnx_box_proposal', type=str, default=None,
                      help='')
    parser.add_argument('--onnx_box_predictor', type=str, default=None,
                      help='')
    parser.add_argument('--onnx_mask_predictor', type=str, default=None,
                      help='')
    parser.add_argument('--onnx_ep', type=str, default='cpu',
                      help='')
    parser.add_argument('--warm_up', action='store_true',
                      help='')
    parser.add_argument('--test_num', type=int, default=1,
                      help='')
    args = parser.parse_args()

    args.save_io = False
    args.save_io_dir = None

    assert not (args.export and (args.onnx_backbone or args.onnx_box_predictor or args.onnx_mask_predictor))

    main(args)
