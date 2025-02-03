from pathlib import Path
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
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
from collections import OrderedDict

from custom_maskrcnn_model import CustomMaskRCNN
from coco_utils import COCOEvalDataset, coco_collate_fn, get_coco_category_mapping


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
    # Convert to 1D array in Fortran order (column-major)
    mask_array = binary_mask.ravel(order='F')
    
    # If the first value is 1, start with a zero-length run of zeros
    counts = []
    if mask_array[0] == 1:
        counts.append(0)
    
    # Calculate lengths of consecutive identical values
    current_value = mask_array[0]
    current_count = 1
    
    for value in mask_array[1:]:
        if value == current_value:
            current_count += 1
        else:
            counts.append(current_count)
            current_value = value
            current_count = 1
    
    # Add the final run length
    counts.append(current_count)
    
    # If ending with 1, append a zero-length run of zeros
    if current_value == 1:
        counts.append(0)
    
    # Verify total length matches image size
    assert sum(counts) == height * width, f"RLE counts sum ({sum(counts)}) does not match image size ({height * width})"
    
    return {'counts': counts, 'size': [height, width]}

class MaskProcessor:
    """
    Class for parallel mask processing
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

    fix_input_resolution = True
    input_resolution = (800, 1056)

    # Create dataset and dataloader
    preprocess_transforms = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
    dataset = COCOEvalDataset(coco_gt, args.img_dir, transforms=preprocess_transforms,
                              resize=input_resolution if fix_input_resolution else None,
                              )
    
    if args.max_samples is not None:
        dataset.img_ids = dataset.img_ids[:args.max_samples]
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=coco_collate_fn
    )

    if args.visualize:
        os.makedirs(args.visualization_dir, exist_ok=True)

    # Initialize mask processor with number of workers
    mask_processor = MaskProcessor(max_workers=args.num_workers * 2)
    mask_count = 0
    pending_results = []

    #input_transform = GeneralizedRCNNTransform(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #input_transform.training = False

    with torch.no_grad():
        with tqdm(dataloader, desc="Processing batches") as pbar:
            for batch in pbar:
                images = [img.to(device) for img in batch['images']]

                predictions = model(images, batch['img_infos'])

                #if fix_input_resolution:
                #    image_sizes = [(*image.shape[1:3],) for image in images]
                #    org_sizes = [(info["org_height"], info["org_width"]) for info in batch['img_infos']]
                #    print(f"{image_sizes=}")
                #    print(f"{org_sizes=}")
                #    predictions = input_transform.postprocess(predictions, image_sizes, org_sizes)
                #    print(f"{predictions[0]['masks'].shape=}")

                #    for info, org in zip(batch['img_infos'], org_sizes):
                #        info["height"] = org[0]
                #        info["width"] = org[1]
                
                for img_idx, (pred, img_id, img_info) in enumerate(zip(predictions, batch['img_ids'], batch['img_infos'])):
                    if args.visualize:
                        output_path = os.path.join(args.visualization_dir, f'result_{img_id}.png')
                        visualize_prediction(batch['images'][img_idx], pred, category_mapping, coco_gt, output_path, args)
                    
                    boxes = pred['boxes'].cpu().numpy()
                    scores = pred['scores'].cpu().numpy()
                    labels = pred['labels'].cpu().numpy()
                    masks = pred['masks'].cpu().numpy()
                    
                    total_predictions += len(boxes)
                    
                    # Convert masks and create results for parallel processing
                    for box, score, label, mask in zip(boxes, scores, labels, masks):
                        coco_category_id = category_mapping.get(label.item(), -1)
                        if coco_category_id == -1:
                            continue
                        
                        filtered_predictions += 1
                        binary_mask = (mask[0] > 0.5).astype(np.uint8)
                        
                        # Add mask processing to queue
                        mask_processor.submit_mask(mask_count, binary_mask, img_info['height'], img_info['width'])
                        
                        # Save result
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
                
                pbar.set_postfix(OrderedDict(pending=len(pending_results)))

                # Wait for a batch of masks to be processed
                if len(pending_results) >= 1000:
                    processed_results, pending_results = process_pending_results(pending_results, mask_processor)
                    results.extend(processed_results)

    # Process remaining results
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
    
    # Evaluation
    coco_dt = coco_gt.loadRes(args.output_path)
    
    # Evaluate bounding boxes
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    if args.max_samples is not None:
        coco_eval.params.imgIds = dataset.img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    box_map = coco_eval.stats[0]
    
    # Evaluate segmentation
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
    if args.device:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize COCO dataset
    coco_gt = COCO(args.ann_file)
    
    # Print category information
    #cats = coco_gt.loadCats(coco_gt.getCatIds())
    #print("\nCOCO Categories (first 5):")
    #for cat in cats[:5]:
    #    print(f"ID: {cat['id']}, Name: {cat['name']}")

    # Get category mapping
    category_mapping = get_coco_category_mapping(coco_gt)
    #print("\nCategory Mapping (first 5):")
    #items = list(category_mapping.items())[:5]
    #for model_idx, coco_id in items:
    #    cat = coco_gt.loadCats([coco_id])[0]
    #    print(f"Model index {model_idx} -> COCO ID {coco_id} ({cat['name']})")

    # Load model and weights
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=args.score_threshold,
        rpn_post_nms_top_n_test=1000,   # Number of proposals after NMS
        box_detections_per_img=100,     # Maximum detections per image
        rpn_pre_nms_top_n_test=1000,    # Number of proposals before NMS
        rpn_fg_iou_thresh=0.7,          # IoU threshold for foreground
        rpn_bg_iou_thresh=0.3,          # IoU threshold for background
        box_nms_thresh=0.5,             # NMS threshold for predictions
        box_fg_iou_thresh=0.5,          # Box head foreground IoU threshold
        box_bg_iou_thresh=0.5           # Box head background IoU threshold
    )

    model = CustomMaskRCNN(model, save_io=args.save_io, save_io_dir=args.save_io_dir)
    
    # Print model configuration
    print("\nModel Configuration:")
    print(f"Score threshold: {args.score_threshold}")
    print(f"RPN post NMS top N: 1000")
    print(f"Box detections per image: 100")
    print(f"Box NMS threshold: 0.5")
    print(f"Box foreground IoU threshold: 0.5")
    
    model = model.to(device)
    model.eval()

    if isinstance(model, CustomMaskRCNN):
        # Warm up
        with torch.no_grad():
            model([torch.rand((3, 800, 1056)).to(device) for _ in range(1)], [{"id": 0, "height": 800, "width": 1056}])
        model.reset()

    # Evaluate model
    print(f"Evaluating model on {args.max_samples if args.max_samples else 'all'} samples...")
    box_map, mask_map = evaluate_model(model, coco_gt, device, args, category_mapping)
    
    print(f"\nResults:")
    print(f"Box mAP: {box_map:.4f}")
    print(f"Mask mAP: {mask_map:.4f}")

    if isinstance(model, CustomMaskRCNN):
        model.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model on COCO dataset')
    parser.add_argument('--ann_file', type=str, 
                      default='C:/Users/juna/Work/dataset/coco/annotations/instances_val2017.json',
                      help='Path to COCO annotation file')
    parser.add_argument('--img_dir', type=str,
                      default='C:/Users/juna/Work/dataset/coco/val2017',
                      help='Directory containing the images')
    parser.add_argument('--max_samples', type=int, default=1,
                      help='Maximum number of samples to process (default: all samples)')
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
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for evaluation (default: 4)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of worker processes for data loading (default: 4)')
    parser.add_argument('--device', type=str, default=None,
                      help='')
    parser.add_argument('--save_io', action='store_true',
                      help='')
    parser.add_argument('--save_io_dir', type=str, default='model_io',
                      help='')
    
    args = parser.parse_args()
    main(args)
