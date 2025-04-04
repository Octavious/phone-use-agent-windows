#!/usr/bin/env python3
"""
OmniParser Runner Script

This script serves as a command-line interface to OmniParser,
allowing it to be called from the phone agent to process screenshots.
"""

import os
import sys
import json
import argparse
import base64
import warnings
import logging
import io
import locale
from pathlib import Path
from transformers import logging as transformers_logging

# Handle Windows encoding issues
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            # Fallback to default locale
            locale.setlocale(locale.LC_ALL, '')
        except locale.Error:
            # If all else fails, just set the encoding
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Uncomment the following line to Force OmniParser to use GPU1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Disable HF warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific Hugging Face warnings
transformers_logging.set_verbosity_error()

# Additionally, suppress logging below WARNING level
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from PIL import Image

# Clear any existing CUDA memory
torch.cuda.empty_cache()

# Add the OmniParser directory to sys.path to import its modules
current_dir = Path(__file__).parent
omniparser_dir = current_dir / "OmniParser"
sys.path.append(str(omniparser_dir))

# Import OmniParser utilities
try:
    from util.utils import (
        check_ocr_box, 
        get_yolo_model, 
        get_caption_model_processor, 
        get_som_labeled_img,
        predict_yolo  # Import predict_yolo from utils.py
    )
except ImportError as e:
    print(f"Error importing OmniParser modules: {e}")
    print(f"Current directory: {current_dir}")
    print(f"OmniParser directory: {omniparser_dir}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="OmniParser Runner for phone screenshots")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    parser.add_argument("--box_threshold", type=float, default=0.05, help="Box confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="IOU threshold for box overlap")
    parser.add_argument("--use_paddleocr", action="store_true", help="Use PaddleOCR for text detection")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for detection")
    return parser.parse_args()

def process_image(image_path, box_threshold, iou_threshold, use_paddleocr, imgsz):
    """Process an image with OmniParser."""
    try:
        # Set up device - use CUDA if available (which will be GPU1 based on CUDA_VISIBLE_DEVICES)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"OmniParser running on device: {device}")
        
        # Load image
        print(f"Loading image from: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: Image file does not exist at {image_path}")
            return None
            
        image_input = Image.open(image_path)
        print(f"Image size: {image_input.size}")
        print(f"Image mode: {image_input.mode}")
        
        # Calculate box overlay ratio for visualization
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        # Load models
        print("Loading YOLO model...")
        yolo_model_path = 'OmniParser/weights/icon_detect/model.pt'
        if not os.path.exists(yolo_model_path):
            print(f"Error: YOLO model file does not exist at {yolo_model_path}")
            return None
            
        yolo_model = get_yolo_model(model_path=yolo_model_path)
        print("YOLO model loaded successfully")
        
        print("Loading Florence caption model...")
        caption_model_path = "OmniParser/weights/icon_caption_florence"
        if not os.path.exists(caption_model_path):
            print(f"Error: Florence caption model does not exist at {caption_model_path}")
            return None
            
        caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path=caption_model_path
        )
        print("Florence caption model loaded successfully")
        
        # Run OCR
        print("Running OCR...")
        try:
            ocr_bbox_rslt, _ = check_ocr_box(
                image_input, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
                use_paddleocr=use_paddleocr
            )
            text, ocr_bbox = ocr_bbox_rslt
            print(f"OCR results - Text count: {len(text)}, Bbox count: {len(ocr_bbox)}")
        except Exception as e:
            print(f"Error during OCR: {str(e)}")
            text, ocr_bbox = [], []
        
        # Handle empty OCR results
        if not text or not ocr_bbox:
            print("No text detected in image, continuing with icon detection only")
            ocr_bbox = []
            text = []
        
        # Run YOLO detection
        print("Running YOLO detection...")
        try:
            yolo_boxes, yolo_conf, yolo_phrases = predict_yolo(
                model=yolo_model,
                image=image_input,
                box_threshold=box_threshold,
                imgsz=imgsz,
                scale_img=True,
                iou_threshold=iou_threshold
            )
            print(f"YOLO detection results - Boxes: {len(yolo_boxes) if yolo_boxes is not None else 0}")
            print(f"YOLO confidence scores: {yolo_conf if yolo_conf is not None else 'None'}")
            print(f"YOLO phrases: {yolo_phrases if yolo_phrases is not None else 'None'}")
        except Exception as e:
            print(f"Error during YOLO detection: {str(e)}")
            yolo_boxes, yolo_conf, yolo_phrases = None, None, None
        
        # Process results
        if yolo_boxes is not None and len(yolo_boxes) > 0:
            print("Processing YOLO detection results...")
            # Convert boxes to the correct format
            yolo_boxes = yolo_boxes / torch.tensor([image_input.size[0], image_input.size[1], image_input.size[0], image_input.size[1]])
            
            # Get labeled image
            labeled_img, label_coordinates, filtered_boxes = get_som_labeled_img(
                image_source=image_input,
                model=yolo_model,
                BOX_TRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                text_scale=0.8 * box_overlay_ratio,
                text_padding=5,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=iou_threshold,
                scale_img=True,
                imgsz=imgsz
            )
            
            print(f"Processing complete - Filtered boxes: {len(filtered_boxes)}")
            
            # Prepare output
            output = {
                'image': labeled_img,
                'label_coordinates': label_coordinates,
                'filtered_boxes': filtered_boxes
            }
            
            return output
        else:
            print("No objects detected in image")
            print("Possible reasons:")
            print("1. Image quality or content")
            print("2. Model weights not loaded correctly")
            print("3. Detection thresholds too high")
            print("4. YOLO detection failed")
            return None
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        # Process the image
        output_data = process_image(
            args.input,
            args.box_threshold,
            args.iou_threshold,
            args.use_paddleocr,
            args.imgsz
        )
        
        # Handle None output
        if output_data is None:
            print("No objects detected in image, saving empty result")
            output_data = {
                'image': None,
                'label_coordinates': [],
                'filtered_boxes': []
            }
        
        # Save output to file
        with open(args.output, 'w') as f:
            json.dump(output_data, f)
        
        print(f"Results saved to {args.output}")
        print(f"Output data: {json.dumps(output_data, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
