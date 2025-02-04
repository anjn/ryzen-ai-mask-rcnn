import collections
import pickle
import sys
import time
import queue
import cv2
import argparse
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

from custom_maskrcnn_model import CodeTimer, CustomMaskRCNN
from custom_maskrcnn_pipeline import (
    MaskRCNNPipeline,
    FPS,
)

def main(args):
    control_fps = True
    #control_fps = False
    line_width = 2
    #video_path = 0
    play_loop = True
    save_video = False

    # Start video capture
    print("Start video capture", args.video)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():  
        print(f"Error: Couldn't open the video file! {args.video}")  
        sys.exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{video_fps=}")
    video_fps = 1.0
    print(f"{video_fps=}")

    # Record video
    video_out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter('output.avi', fourcc, 15.0, (1920,  1080))

    # Device configuration
    if args.device:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load label_mapping
    with open("sample/label_mapping.pickle", mode="rb") as f:
        label_mapping = pickle.load(f)

    # Load model and weights
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    )
    model = CustomMaskRCNN(model, args)
    model = model.to(device)
    model.eval()

    # Mask-RCNN pipeline
    pipeline = MaskRCNNPipeline(model, device, args, label_mapping)

    # Variables for fps control
    video_start_time = None
    frame_count = 0

    # Variables for fps measuring
    fps_counter = FPS()

    # Variables for ordering output frames
    input_frame_id = 0
    buffered_frame = collections.deque(maxlen=5)
    out_frame = None

    # Main loop
    while True:
        if video_start_time is None:
            video_start_time = time.time()
        
        if control_fps:
            # Calculate target time for current frame
            target_time = video_start_time + (frame_count / video_fps)
            current_time = time.time()
            
            # Sleep if we're ahead of schedule
            sleep_duration = target_time - current_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        
        for _ in range(1):
            ret, frame = cap.read()
        frame_count += 1

        if not ret:
            if play_loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                video_start_time = None
                continue
            else:
                break

        pipeline.put(frame)
        input_frame_id += 1

        try:
            while True:
                buffered_frame.append(pipeline.get())
                fps_counter() # Increment counter
        except queue.Empty:
            pass

        if len(buffered_frame) > 0:
            out_frame, _ = buffered_frame.popleft()

            # Draw fps
            label = f"Mask R-CNN FPS : {fps_counter.get():.1f}"
            cv2.putText(out_frame, label, (10, 40), 0, 1, (0,  0,0), thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(out_frame, label, (10, 40), 0, 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)

            if video_out is not None:
                video_out.write(out_frame)

        if out_frame is not None:
            cv2.imshow('Video', out_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("finish app")
            break

    cap.release()
    if video_out is not None:
        video_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model on COCO dataset')
    parser.add_argument('--ann_file', type=str, default=None,
                      help='Path to COCO annotation file')
    parser.add_argument('--video', type=str, default=r"C:\Users\juna\OneDrive - Advanced Micro Devices Inc\Embedded-x86\Demo\Ryzen-AI\Videos\video-object-detection-1.mp4",
                      help='')
    parser.add_argument('--visualize_threshold', type=float, default=0.3,
                      help='Score threshold for visualization (default: 0.3)')
    parser.add_argument('--device', type=str, default='cpu',
                      help='')
    parser.add_argument('--input_size', type=int, default=[544, 960], nargs=2,
                    help='Model input size')
    parser.add_argument('--onnx_backbone', type=str, default="./model/maskrcnn_backbone_quant.onnx",
                      help='')
    parser.add_argument('--onnx_ep', type=str, default='vai',
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