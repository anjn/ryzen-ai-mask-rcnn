import json
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
import argparse
import cv2
import matplotlib.pyplot as plt
import os

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

def preprocess_image(image, device):
    """
    Preprocess image for model input
    Args:
        image: Input image tensor
        device: Device to move tensor to
    Returns:
        Preprocessed image tensor
    """
    # Convert to float32
    image = image.to(torch.float32)
    
    # Normalize pixel values
    image = image / 255.0
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    image = (image - mean) / std
    
    return image

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
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
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
        if score < args.score_threshold:
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

def evaluate_model(model, coco_gt, device, args, category_mapping):
    """
    Evaluate model on COCO dataset
    Args:
        model: Model to evaluate
        coco_gt: COCO ground truth object
        device: Device to run evaluation on
        args: Command line arguments containing max_samples and img_dir
        category_mapping: Mapping from model category indices to COCO category IDs
    Returns:
        box_map: Mean Average Precision for bounding boxes
        mask_map: Mean Average Precision for masks
    """
    # Initialize list to store detections
    results = []
    total_predictions = 0
    filtered_predictions = 0

    print("\nEvaluation Settings:")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Max samples: {args.max_samples}")

    # Get image IDs
    img_ids = coco_gt.getImgIds()
    if args.max_samples is not None:
        img_ids = img_ids[:args.max_samples]

    # Create visualization directory if needed
    if args.visualize:
        os.makedirs(args.visualization_dir, exist_ok=True)
    
    # Process each image
    with torch.no_grad():
        for img_id in tqdm(img_ids, desc="Processing images"):
            # Load image
            img_info = coco_gt.loadImgs(img_id)[0]
            image_path = img_info['file_name']
            
            # Load and preprocess image
            image = torchvision.io.read_image(f"{args.img_dir}/{image_path}")
            if image.shape[0] == 1:  # Handle grayscale images
                image = image.repeat(3, 1, 1)
            
            # Move to device first, then preprocess
            image = image.to(device)
            image = preprocess_image(image, device)
            
            # Forward pass
            predictions = model([image])[0]
            
            # Visualize and save results if enabled
            if args.visualize:
                output_path = os.path.join(args.visualization_dir, f'result_{img_id}.png')
                visualize_prediction(image, predictions, category_mapping, coco_gt, output_path, args)
            
            # Convert predictions to COCO format
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            masks = predictions['masks'].cpu().numpy()
            
            total_predictions += len(boxes)
            
            # Convert masks to RLE format
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                if score < args.score_threshold:  # Skip low confidence predictions
                    continue
                
                # Map model category index to COCO category ID
                coco_category_id = category_mapping.get(label.item(), -1)
                if coco_category_id == -1:  # Skip if no valid mapping
                    continue
                
                filtered_predictions += 1
                # Ensure mask is binary
                binary_mask = (mask[0] > 0.5).astype(np.uint8)
                rle = mask_to_rle(binary_mask, img_info['height'], img_info['width'])
                
                # Convert box from [x1, y1, x2, y2] to COCO format [x, y, width, height]
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1
                result = {
                    'image_id': img_id,
                    'category_id': coco_category_id,  # Use mapped COCO category ID
                    'bbox': [x1, y1, width, height],  # COCO format
                    'score': float(score.item()),  # Ensure float
                    'segmentation': rle
                }
                results.append(result)
    
    # Print prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"Total predictions across all images: {total_predictions}")
    print(f"Predictions after score threshold: {filtered_predictions}")
    print(f"Average predictions per image: {total_predictions / len(img_ids):.2f}")
    print(f"Average filtered predictions per image: {filtered_predictions / len(img_ids):.2f}")

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nSaved {len(results)} predictions to {args.output_path}")
    if args.visualize:
        print(f"Saved visualization results to {args.visualization_dir}")
    
    if len(results) == 0:
        print("WARNING: No predictions passed the score threshold!")
        return 0.0, 0.0
    
    # Evaluate bounding boxes
    coco_dt = coco_gt.loadRes(args.output_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    if args.max_samples is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    box_map = coco_eval.stats[0]  # mAP@IoU=0.50:0.95
    
    # Evaluate segmentation
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    if args.max_samples is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mask_map = coco_eval.stats[0]  # mAP@IoU=0.50:0.95

    return box_map, mask_map

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
    Convert a binary mask to RLE format
    Args:
        binary_mask: A binary mask (0 or 1 values)
        height: Original image height
        width: Original image width
    Returns:
        RLE encoded mask
    """
    rle = {'counts': [], 'size': [height, width]}
    counts = rle.get('counts')
    
    last_elem = 0
    running_length = 0
    
    for elem in binary_mask.ravel(order='F'):
        if elem == last_elem:
            running_length += 1
        else:
            counts.append(running_length)
            running_length = 1
            last_elem = elem
    counts.append(running_length)
    
    return rle

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
        rpn_pre_nms_top_n_test=2000,    # Number of proposals before NMS
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
    parser.add_argument('--score_threshold', type=float, default=0.5,
                      help='Score threshold for predictions (default: 0.5)')
    parser.add_argument('--output_path', type=str, default='maskrcnn_predictions.json',
                      help='Path to save prediction results (default: maskrcnn_predictions.json)')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization of predictions')
    parser.add_argument('--visualization_dir', type=str, default='visualization_results',
                      help='Directory to save visualization results (default: visualization_results)')
    
    args = parser.parse_args()
    main(args)
