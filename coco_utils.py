from torch.utils.data import Dataset, DataLoader
import torchvision
from pycocotools.coco import COCO

class COCOEvalDataset(Dataset):
    """
    Custom Dataset for COCO evaluation with batch processing support
    """
    def __init__(self, coco_gt, img_dir, transforms=None, resize=None):
        self.coco = coco_gt
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = self.coco.getImgIds()
        self.resize = resize
        if self.resize is not None:
            self.resize_transform = torchvision.transforms.Resize(self.resize)
        
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

        if self.resize is not None:
            image = self.resize_transform(image)
            img_info['org_height'] = img_info['height']
            img_info['org_width'] = img_info['width']
            img_info['height'] = self.resize[0]
            img_info['width'] = self.resize[1]
            
        return {
            'image': image,
            'img_id': img_id,
            'img_info': img_info
        }

def coco_collate_fn(batch):
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

