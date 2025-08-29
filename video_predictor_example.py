#!/usr/bin/env python3
"""
Video segmentation with SAM 2 using Grounding DINO

This script shows how to use SAM 2 for text-prompted segmentation in videos. It covers:
- Loading and setting up SAM 2 video predictor and Grounding DINO
- Text-prompted object detection with Grounding DINO
- Segmentation with detected bounding boxes
- Propagating masks across video frames
- Multi-object segmentation and tracking
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def show_mask(mask, ax, obj_id=None, random_color=False):
    """Display a mask on the given axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """Display a bounding box on the given axis."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def load_sam2_predictor():
    """Load and setup the SAM 2 video predictor."""
    print("Loading SAM 2 video predictor...")
    
    # Model configuration
    model_cfg = "sam2/configs/sam2/sam2_hiera_b.yaml"
    sam2_checkpoint = "checkpoints/sam2_hiera_b.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Model config: {model_cfg}")
    print(f"Checkpoint: {sam2_checkpoint}")
    
    # Build the predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor


def load_grounding_dino():
    """Load and setup Grounding DINO model."""
    print("Loading Grounding DINO model...")
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
    
    print(f"Grounding DINO model loaded: {model_id}")
    return processor, model


def load_video_frames(video_dir):
    """Load video frames from directory."""
    print(f"Loading video frames from: {video_dir}")
    
    # Get frame names
    frame_names = [
        p for p in os.listdir(video_dir)
        if p.endswith('.jpg') and p[0].isdigit()
    ]
    frame_names.sort()
    
    print(f"Found {len(frame_names)} frames")
    
    # Show first frame
    if frame_names:
        first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
        print(f"First frame size: {first_frame.size}")
        
        plt.figure(figsize=(10, 6))
        plt.imshow(first_frame)
        plt.title("First video frame")
        plt.axis('off')
        plt.show()
    
    return frame_names


def detect_object_with_text(predictor, processor, model, video_dir, frame_names, text_prompt, ann_frame_idx=0):
    """Detect object using Grounding DINO with text prompt and segment with SAM2."""
    print(f"\n=== Text-Prompted Object Detection: '{text_prompt}' ===")
    
    # Initialize inference state
    print("Initializing inference state...")
    inference_state = predictor.init_state(video_path=video_dir)
    
    # Load the annotation frame
    ann_frame = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
    print(f"Annotation frame: {ann_frame_idx}")
    
    # Display the frame
    plt.figure(figsize=(10, 6))
    plt.imshow(ann_frame)
    plt.title(f"Frame {ann_frame_idx} - Detecting: '{text_prompt}'")
    plt.axis('off')
    plt.show()
    
    # Use Grounding DINO to detect object with text prompt
    print(f"Detecting '{text_prompt}' with Grounding DINO...")
    inputs = processor(images=ann_frame, text=text_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=0.25,
        target_sizes=[ann_frame.size[::-1]]  # (H, W)
    )
    
    # Find the best detection
    boxes = results[0]["boxes"]   # (N, 4)
    scores = results[0]["scores"] # (N,)
    labels = results[0]["labels"]
    
    if len(scores) > 0:
        best_idx = scores.argmax().item()
        best_box = boxes[best_idx].tolist()   # [x_min, y_min, x_max, y_max]
        best_score = scores[best_idx].item()
        best_label = labels[best_idx]
        
        print(f"Best detection:")
        print(f"  Box coordinates: {best_box}")
        print(f"  Confidence score: {best_score:.3f}")
        print(f"  Label: {best_label}")
        
        # Add the bounding box to SAM2
        ann_obj_id = predictor.add_new_mask(
            inference_state, 
            ann_frame_idx, 
            box=best_box
        )
        
        # Check if a box was detected by Grounding DINO
        if best_box is not None:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=best_box, # Use the generated box instead of points
            )
        
        print(f"Object ID assigned: {ann_obj_id}")
        
        # Display the frame with the detected bounding box
        plt.figure(figsize=(10, 6))
        plt.imshow(ann_frame)
        show_box(best_box, plt.gca())
        plt.title(f"Frame {ann_frame_idx} - Detected: '{text_prompt}' (Score: {best_score:.3f})")
        plt.axis('off')
        plt.show()
        
        return inference_state, ann_obj_id, best_box, best_score
        
    else:
        print(f"No object '{text_prompt}' detected in the image")
        return None, None, None, None


def propagate_masks(predictor, inference_state):
    """Propagate masks across the video."""
    print("Propagating masks across video...")
    
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            obj_id: predictor.get_mask_from_logits(out_mask_logits[obj_id_idx])
            for obj_id_idx, obj_id in enumerate(out_obj_ids)
        }
    # render the segmentation results every few frames
    vis_frame_stride = 30
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    print(f"Generated masks for {len(video_segments)} frames")
    return video_segments


def segmentation_with_text_prompt(predictor, processor, model, video_dir, frame_names, text_prompt):
    """Demonstrate segmentation with text-prompted bounding box detection."""
    print(f"\n=== Segmentation with Text Prompt: '{text_prompt}' ===")
    
    # Detect object and get bounding box
    inference_state, ann_obj_id, best_box, best_score = detect_object_with_text(
        predictor, processor, model, video_dir, frame_names, text_prompt
    )
    
    if inference_state is None:
        print(f"Failed to detect '{text_prompt}'")
        return None, None, None
    
    # Propagate the prompts to get masks across the video
    video_segments = propagate_masks(predictor, inference_state)
    
    # Display results for annotation frame
    ann_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    plt.figure(figsize=(10, 6))
    plt.imshow(ann_frame)
    show_mask(video_segments[0][ann_obj_id], plt.gca(), obj_id=ann_obj_id)
    show_box(best_box, plt.gca())
    plt.title(f"Frame 0 - Segmentation of '{text_prompt}'")
    plt.axis('off')
    plt.show()
    
    return inference_state, video_segments, ann_obj_id


def multi_object_text_segmentation(predictor, processor, model, video_dir, frame_names):
    """Demonstrate multi-object segmentation with different text prompts."""
    print("\n=== Multi-Object Text-Prompted Segmentation ===")
    
    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_dir)
    
    # Define text prompts for different objects
    text_prompts = ["person", "car", "chair"]
    obj_ids = []
    boxes = []
    
    ann_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    
    # Detect and add each object
    for i, text_prompt in enumerate(text_prompts):
        print(f"\nDetecting object {i+1}: '{text_prompt}'")
        
        inputs = processor(images=ann_frame, text=text_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.25,
            target_sizes=[ann_frame.size[::-1]]
        )
        
        boxes_detected = results[0]["boxes"]
        scores = results[0]["scores"]
        
        if len(scores) > 0:
            best_idx = scores.argmax().item()
            best_box = boxes_detected[best_idx].tolist()
            best_score = scores[best_idx].item()
            
            print(f"  Detected '{text_prompt}' with score: {best_score:.3f}")
            
            # Add to SAM2
            obj_id = predictor.add_new_mask(
                inference_state, 
                0, 
                box=best_box
            )
            
            # Check if a box was detected by Grounding DINO
            if best_box is not None:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=best_box, # Use the generated box instead of points
                )
            
            obj_ids.append(obj_id)
            boxes.append(best_box)
            print(f"  Object ID: {obj_id}")
        else:
            print(f"  No '{text_prompt}' detected")
    
    if not obj_ids:
        print("No objects detected for any prompt")
        return None, None
    
    # Display the frame with all detected objects
    plt.figure(figsize=(15, 6))
    
    # Original frame with boxes
    plt.subplot(1, 2, 1)
    plt.imshow(ann_frame)
    for i, box in enumerate(boxes):
        show_box(box, plt.gca())
    plt.title(f"Frame 0 - All detected objects")
    plt.axis('off')
    
    # Propagate masks
    video_segments = propagate_masks(predictor, inference_state)
    
    # Show segmentation results
    plt.subplot(1, 2, 2)
    plt.imshow(ann_frame)
    for i, obj_id in enumerate(obj_ids):
        if obj_id in video_segments[0]:
            show_mask(video_segments[0][obj_id], plt.gca(), obj_id=obj_id, random_color=True)
    plt.title(f"Frame 0 - Segmentation results")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return inference_state, video_segments


def main():
    """Main function to run the text-prompted video segmentation examples."""
    print("=== SAM 2 + Grounding DINO Video Segmentation Demo ===\n")
    
    # Set video directory
    video_dir = "./custom_video_frames"
    if not os.path.exists(video_dir):
        print(f"Video directory {video_dir} not found.")
        print("Please ensure you have video frames in the correct format:")
        print("- Directory should contain JPEG frames with names like '00000.jpg', '00001.jpg', etc.")
        return
    
    # Load models
    try:
        predictor = load_sam2_predictor()
        processor, model = load_grounding_dino()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure SAM 2 and Grounding DINO are properly installed")
        return
    
    # Load video frames
    try:
        frame_names = load_video_frames(video_dir)
        if not frame_names:
            print("No video frames found!")
            return
    except Exception as e:
        print(f"Error loading video frames: {e}")
        return
    
    # Run text-prompted segmentation examples
    try:
        # Example 1: Single object segmentation with text prompt
        print("\n" + "="*50)
        inference_state, video_segments, ann_obj_id = segmentation_with_text_prompt(
            predictor, processor, model, video_dir, frame_names, "person"
        )
        
        # Example 2: Multi-object segmentation with different text prompts
        print("\n" + "="*50)
        multi_inference_state, multi_segments = multi_object_text_segmentation(
            predictor, processor, model, video_dir, frame_names
        )
        
        print("\n=== All examples completed successfully! ===")
        print("You can now try SAM 2 + Grounding DINO on your own videos!")
        print("Use different text prompts like 'car', 'dog', 'chair', etc.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 