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
from pathlib import Path
from transformers import logging as transformers_logging

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
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

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
        image_input = Image.open(image_path)
        
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
        yolo_model = get_yolo_model(model_path='OmniParser/weights/icon_detect/model.pt')
        
        print("Loading Florence caption model...")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path="OmniParser/weights/icon_caption_florence"
        )
        
        # Run OCR
        print("Running OCR...")
        ocr_bbox_rslt, _ = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Process with SOM labeling
        print("Analyzing screen elements...")
        labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_input, 
            yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=caption_model_processor, 
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz
        )
        
        # Format output for compatibility with original format
        formatted_output = {
            'annotated_image': labeled_img,  # This is already base64 encoded
            'elements': parsed_content_list
        }
        
        print(f"Detected {len(parsed_content_list)} elements")
        return formatted_output
        
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        raise

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
        
        # Save output to file
        with open(args.output, 'w') as f:
            json.dump(output_data, f)
        
        print(f"Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
