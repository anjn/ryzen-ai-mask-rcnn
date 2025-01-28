import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
import json
import argparse

def evaluate_model(model, coco_gt, device, args):
    """
    Evaluate model on COCO dataset
    Args:
        model: Model to evaluate
        coco_gt: COCO ground truth object
        device: Device to run evaluation on
        args: Command line arguments containing max_samples and img_dir
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
            
            # Convert to float and apply proper normalization
            image = image.to(torch.float32)
            image = torchvision.transforms.functional.normalize(
                image / 255.0,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            image = image.to(device)
            
            # Forward pass
            predictions = model([image])[0]
            
            # Convert predictions to COCO format
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            masks = predictions['masks'].cpu().numpy()
            
            # Print debug information for the first image
            if img_id == img_ids[0]:
                print(f"\nFirst image predictions:")
                print(f"Number of predictions: {len(boxes)}")
                print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
                print(f"Unique labels: {np.unique(labels)}")
                print(f"Mask shape: {masks.shape}")
                print(f"Mask value range: {masks.min():.3f} - {masks.max():.3f}")
            
            total_predictions += len(boxes)
            
            # Convert masks to RLE format
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                if score < args.score_threshold:  # Skip low confidence predictions
                    continue
                
                filtered_predictions += 1
                # Ensure mask is binary
                binary_mask = (mask[0] > 0.5).astype(np.uint8)
                rle = mask_to_rle(binary_mask, img_info['height'], img_info['width'])
                result = {
                    'image_id': img_id,
                    'category_id': label.item(),
                    'bbox': box.tolist(),
                    'score': score.item(),
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
    
    if len(results) == 0:
        print("WARNING: No predictions passed the score threshold!")
        return 0.0, 0.0
    
    # Evaluate bounding boxes
    coco_dt = coco_gt.loadRes(args.output_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    box_map = coco_eval.stats[0]  # mAP@IoU=0.50:0.95
    
    # Evaluate segmentation
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mask_map = coco_eval.stats[0]  # mAP@IoU=0.50:0.95

    return box_map, mask_map

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

    # Load model and weights
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if not args.weights_path else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        box_score_thresh=args.score_threshold,
        rpn_post_nms_top_n_test=100,  # Increase number of proposals
        box_detections_per_img=100     # Increase number of detections
    )
    
    if args.weights_path:
        print(f"Loading weights from {args.weights_path}")
        model.load_state_dict(torch.load(args.weights_path))
    
    model = model.to(device)
    model.eval()

    # Initialize COCO dataset
    coco_gt = COCO(args.ann_file)

    # Evaluate model
    print(f"Evaluating model on {args.max_samples if args.max_samples else 'all'} samples...")
    box_map, mask_map = evaluate_model(model, coco_gt, device, args)
    
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
    
    args = parser.parse_args()
    main(args)
