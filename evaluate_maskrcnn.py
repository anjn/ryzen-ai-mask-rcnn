import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
import argparse
import cv2
import matplotlib.pyplot as plt
import os
import time
import concurrent.futures
import queue
from queue import Queue
from threading import Thread
import threading

class COCOEvalDataset(Dataset):
    """
    Custom Dataset for COCO evaluation with batch processing support
    """
    def __init__(self, coco_gt, img_dir, transforms=None):
        self.coco = coco_gt
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = self.coco.getImgIds()
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = img_info['file_name']
        
        # Load image
        image = torchvision.io.read_image(f"{self.img_dir}/{image_path}")
        if image.shape[0] == 1:  # Handle grayscale images
            image = image.repeat(3, 1, 1)
            
        # Apply transforms if provided
        if self.transforms is not None:
            image = self.transforms(image)
            
        return {
            'image': image,
            'img_id': img_id,
            'img_info': img_info
        }

def collate_fn(batch):
    """
    Custom collate function for the DataLoader
    """
    return {
        'images': [item['image'] for item in batch],
        'img_ids': [item['img_id'] for item in batch],
        'img_infos': [item['img_info'] for item in batch]
    }

def get_coco_category_mapping(coco_gt):
    """
    Get mapping between model category indices and COCO dataset category IDs
    Args:
        coco_gt: COCO ground truth object
    Returns:
        Dictionary mapping model indices to COCO category IDs
    """
    # COCO classes that the pre-trained model was trained on (in order)
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Get COCO category information
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    coco_name_to_id = {cat['name']: cat['id'] for cat in cats}
    
    # Create mapping from model index to COCO category ID
    model_to_coco_id = {}
    for idx, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES):
        if name != 'N/A' and name != '__background__':
            if name in coco_name_to_id:
                model_to_coco_id[idx] = coco_name_to_id[name]
    
    return model_to_coco_id

def visualize_prediction(image, prediction, category_mapping, coco_gt, output_path, args):
    """
    Visualize prediction results on the image
    Args:
        image: Input image tensor
        prediction: Model prediction dictionary
        category_mapping: Mapping from model category indices to COCO category IDs
        coco_gt: COCO ground truth object
        output_path: Path to save visualization
        args: Command line arguments
    """
    # Convert tensor to numpy array
    image_np = image.cpu().permute(1, 2, 0).numpy()
    
    # Create visualization image
    vis_image = image_np.copy()
    
    # Get prediction components
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    masks = prediction['masks'].cpu().numpy()
    
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
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Get category name
        coco_category_id = category_mapping.get(label.item(), -1)
        if coco_category_id != -1:
            category_name = coco_gt.loadCats([coco_category_id])[0]['name']
        else:
            category_name = f"Unknown (Label {label})"
        
        # Draw label and score
        label_text = f'{category_name}: {score:.2f}'
        cv2.putText(vis_image, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        # Apply mask
        mask_np = mask[0] > 0.5
        mask_color = (color[:3] * 255 * 0.5).astype(np.uint8)
        mask_overlay = np.zeros_like(vis_image)
        for c in range(3):
            mask_overlay[:, :, c] = mask_color[c]
        mask_overlay = cv2.bitwise_and(mask_overlay, mask_overlay, mask=mask_np.astype(np.uint8))
        vis_image = cv2.addWeighted(vis_image, 1, mask_overlay, 0.5, 0)
    
    # Save visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in format [x1, y1, x2, y2]
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
    else:
        return 0.0
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def mask_to_rle(binary_mask, height, width):
    """
    Convert a binary mask to RLE format - Optimized version
    Args:
        binary_mask: A binary mask (0 or 1 values)
        height: Original image height
        width: Original image width
    Returns:
        RLE encoded mask in COCO format
    """
    # Fortranオーダーで1次元配列に変換（列優先）
    mask_array = binary_mask.ravel(order='F')
    
    # 最初が1の場合、0の長さは0から開始
    counts = []
    if mask_array[0] == 1:
        counts.append(0)
    
    # 連続する同じ値の長さを計算
    current_value = mask_array[0]
    current_count = 1
    
    for value in mask_array[1:]:
        if value == current_value:
            current_count += 1
        else:
            counts.append(current_count)
            current_value = value
            current_count = 1
    
    # 最後の連続部分を追加
    counts.append(current_count)
    
    # 最後が1で終わる場合、0の長さ0を追加
    if current_value == 1:
        counts.append(0)
    
    # 合計が画像のピクセル数と一致することを確認
    assert sum(counts) == height * width, f"RLE counts sum ({sum(counts)}) does not match image size ({height * width})"
    
    return {'counts': counts, 'size': [height, width]}

class MaskProcessor:
    """
    マスク処理を並列化するためのクラス
    """
    def __init__(self, max_workers=None):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.queue = Queue()
        self.results = {}
        self.processing = True
        self.worker_thread = Thread(target=self._process_queue)
        self.worker_thread.start()
        self.lock = threading.Lock()

    def _process_queue(self):
        while self.processing or not self.queue.empty():
            try:
                task = self.queue.get(timeout=3.0)
                if task is None:
                    break
                
                idx, binary_mask, height, width = task
                future = self.executor.submit(mask_to_rle, binary_mask, height, width)
                with self.lock:
                    self.results[idx] = future
                
            except queue.Empty:
                continue

    def submit_mask(self, idx, binary_mask, height, width):
        self.queue.put((idx, binary_mask, height, width))

    def get_result(self, idx):
        with self.lock:
            future = self.results.get(idx)
            if future is None:
                return None
            return future.result()

    def shutdown(self):
        self.processing = False
        self.queue.put(None)
        self.worker_thread.join()
        self.executor.shutdown()

def process_pending_results(pending_results, mask_processor):
    """
    Process pending results and get their RLE masks
    """
    processed_results = []
    new_pending_results = []
    for item in pending_results:
        mask_id = item['mask_id']
        result = item['result']
        
        # Get RLE mask from processor
        rle = mask_processor.get_result(mask_id)
        if rle is not None:
            result['segmentation'] = rle
            processed_results.append(result)
        else:
            new_pending_results.append(item)
    
    return processed_results, new_pending_results

def evaluate_model(model, coco_gt, device, args, category_mapping):
    """
    Evaluate model on COCO dataset with batch processing and parallel mask processing
    """
    results = []
    total_predictions = 0
    filtered_predictions = 0

    print("\nEvaluation Settings:")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Max samples: {args.max_samples}")
    print(f"Batch size: {args.batch_size}")

    # Create dataset and dataloader
    preprocess_transforms = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
    dataset = COCOEvalDataset(coco_gt, args.img_dir, transforms=preprocess_transforms)
    
    if args.max_samples is not None:
        dataset.img_ids = dataset.img_ids[:args.max_samples]
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    if args.visualize:
        os.makedirs(args.visualization_dir, exist_ok=True)

    # Initialize mask processor with number of workers
    mask_processor = MaskProcessor(max_workers=args.num_workers * 2)
    mask_count = 0
    pending_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            images = [img.to(device) for img in batch['images']]
            predictions = model(images)
            
            for img_idx, (pred, img_id, img_info) in enumerate(zip(predictions, batch['img_ids'], batch['img_infos'])):
                if args.visualize:
                    output_path = os.path.join(args.visualization_dir, f'result_{img_id}.png')
                    visualize_prediction(batch['images'][img_idx], pred, category_mapping, coco_gt, output_path, args)
                
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                masks = pred['masks'].cpu().numpy()
                
                total_predictions += len(boxes)
                
                # 並列処理のためのマスク変換とリザルト作成
                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    if score < args.score_threshold:
                        continue
                    
                    coco_category_id = category_mapping.get(label.item(), -1)
                    if coco_category_id == -1:
                        continue
                    
                    filtered_predictions += 1
                    binary_mask = (mask[0] > 0.5).astype(np.uint8)
                    
                    # マスク処理をキューに追加
                    mask_processor.submit_mask(mask_count, binary_mask, img_info['height'], img_info['width'])
                    
                    # 結果を保存
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    
                    pending_results.append({
                        'mask_id': mask_count,
                        'result': {
                            'image_id': img_id,
                            'category_id': coco_category_id,
                            'bbox': [x1, y1, width, height],
                            'score': float(score.item())
                        }
                    })
                    mask_count += 1

            # 一定数のマスクが処理されるのを待つ
            if len(pending_results) >= 1000:
                processed_results, pending_results = process_pending_results(pending_results, mask_processor)
                results.extend(processed_results)

    # 残りの結果を処理
    while len(pending_results) > 0:
        processed_results, pending_results = process_pending_results(pending_results, mask_processor)
        results.extend(processed_results)
        time.sleep(0.1)
    mask_processor.shutdown()

    print(f"\nPrediction Statistics:")
    print(f"Total predictions across all images: {total_predictions}")
    print(f"Predictions after score threshold: {filtered_predictions}")
    print(f"Average predictions per image: {total_predictions / len(dataset):.2f}")
    print(f"Average filtered predictions per image: {filtered_predictions / len(dataset):.2f}")

    with open(args.output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nSaved {len(results)} predictions to {args.output_path}")
    if args.visualize:
        print(f"Saved visualization results to {args.visualization_dir}")
    
    if len(results) == 0:
        print("WARNING: No predictions passed the score threshold!")
        return 0.0, 0.0
    
    # 評価
    coco_dt = coco_gt.loadRes(args.output_path)
    
    # バウンディングボックスの評価
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    if args.max_samples is not None:
        coco_eval.params.imgIds = dataset.img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    box_map = coco_eval.stats[0]
    
    # セグメンテーションの評価
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    if args.max_samples is not None:
        coco_eval.params.imgIds = dataset.img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mask_map = coco_eval.stats[0]

    return box_map, mask_map

def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize COCO dataset
    coco_gt = COCO(args.ann_file)
    
    # Print category information
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    print("\nCOCO Categories (first 5):")
    for cat in cats[:5]:
        print(f"ID: {cat['id']}, Name: {cat['name']}")

    # Get category mapping
    category_mapping = get_coco_category_mapping(coco_gt)
    print("\nCategory Mapping (first 5):")
    items = list(category_mapping.items())[:5]
    for model_idx, coco_id in items:
        cat = coco_gt.loadCats([coco_id])[0]
        print(f"Model index {model_idx} -> COCO ID {coco_id} ({cat['name']})")

    # Load model and weights
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if not args.weights_path else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        box_score_thresh=args.score_threshold,
        rpn_post_nms_top_n_test=1000,  # Number of proposals after NMS
        box_detections_per_img=100,     # Maximum detections per image
        rpn_pre_nms_top_n_test=1000,    # Number of proposals before NMS
        rpn_fg_iou_thresh=0.7,          # IoU threshold for foreground
        rpn_bg_iou_thresh=0.3,          # IoU threshold for background
        box_nms_thresh=0.5,             # NMS threshold for predictions
        box_fg_iou_thresh=0.5,          # Box head foreground IoU threshold
        box_bg_iou_thresh=0.5           # Box head background IoU threshold
    )
    
    # Print model configuration
    print("\nModel Configuration:")
    print(f"Score threshold: {args.score_threshold}")
    print(f"RPN post NMS top N: 1000")
    print(f"Box detections per image: 100")
    print(f"Box NMS threshold: 0.5")
    print(f"Box foreground IoU threshold: 0.5")
    
    if args.weights_path:
        print(f"Loading weights from {args.weights_path}")
        model.load_state_dict(torch.load(args.weights_path))
    
    model = model.to(device)
    model.eval()

    # Evaluate model
    print(f"Evaluating model on {args.max_samples if args.max_samples else 'all'} samples...")
    box_map, mask_map = evaluate_model(model, coco_gt, device, args, category_mapping)
    
    print(f"\nResults:")
    print(f"Box mAP: {box_map:.4f}")
    print(f"Mask mAP: {mask_map:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model on COCO dataset')
    parser.add_argument('--ann_file', type=str, 
                      default='C:/Users/juna/Work/dataset/coco/annotations/instances_val2017.json',
                      help='Path to COCO annotation file')
    parser.add_argument('--img_dir', type=str,
                      default='C:/Users/juna/Work/dataset/coco/val2017',
                      help='Directory containing the images')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to process (default: all samples)')
    parser.add_argument('--weights_path', type=str, default=None,
                      help='Path to model weights file (default: use pre-trained weights)')
    parser.add_argument('--score_threshold', type=float, default=0.05,
                      help='Score threshold for predictions (default: 0.05)')
    parser.add_argument('--output_path', type=str, default='maskrcnn_predictions.json',
                      help='Path to save prediction results (default: maskrcnn_predictions.json)')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization of predictions')
    parser.add_argument('--visualization_dir', type=str, default='visualization_results',
                      help='Directory to save visualization results (default: visualization_results)')
    parser.add_argument('--visualize_threshold', type=float, default=0.5,
                      help='Score threshold for visualization (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for evaluation (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker processes for data loading (default: 4)')
    
    args = parser.parse_args()
    main(args)
