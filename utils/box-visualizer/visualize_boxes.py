import json
import cv2
import numpy as np
import os
import base64
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Frame, Scrollbar
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageTk

class VisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JSON Box Visualizer")
        self.root.geometry("1200x800")
        
        # Variables
        self.json_path = None
        self.original_image = None
        self.displayed_image = None
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.scroll_x = 0
        self.scroll_y = 0
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        # Top frame for controls
        top_frame = Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # File selection button
        self.select_button = Button(top_frame, text="Select JSON File", command=self.select_file)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # Zoom control
        zoom_frame = Frame(top_frame)
        zoom_frame.pack(side=tk.LEFT, padx=20)
        Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_scale = Scale(zoom_frame, from_=0.1, to=3.0, resolution=0.1, 
                               orient=tk.HORIZONTAL, length=150, command=self.update_zoom)
        self.zoom_scale.set(1.0)
        self.zoom_scale.pack(side=tk.LEFT)
        
        # Rotation control
        rotation_frame = Frame(top_frame)
        rotation_frame.pack(side=tk.LEFT, padx=20)
        Label(rotation_frame, text="Rotation:").pack(side=tk.LEFT)
        self.rotation_scale = Scale(rotation_frame, from_=0, to=360, 
                                   orient=tk.HORIZONTAL, length=150, command=self.update_rotation)
        self.rotation_scale.set(0)
        self.rotation_scale.pack(side=tk.LEFT)
        
        # Reset button
        self.reset_button = Button(top_frame, text="Reset View", command=self.reset_view)
        self.reset_button.pack(side=tk.LEFT, padx=20)
        
        # Create a frame for the canvas and scrollbars
        canvas_frame = Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollbars
        self.scrollbar_y = Scrollbar(canvas_frame)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.scrollbar_x = Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas for image display
        self.canvas = tk.Canvas(canvas_frame, bg="gray", 
                               xscrollcommand=self.scrollbar_x.set,
                               yscrollcommand=self.scrollbar_y.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        self.scrollbar_x.config(command=self.canvas.xview)
        self.scrollbar_y.config(command=self.canvas.yview)
        
        # Bind mouse events for dragging
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        
        # Bind mouse wheel for zooming
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
        # Status label
        self.status_label = Label(self.root, text="Select a JSON file to visualize", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
    def start_drag(self, event):
        self.dragging = True
        self.last_x = event.x
        self.last_y = event.y
        
    def drag(self, event):
        if self.dragging:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.last_x = event.x
            self.last_y = event.y
            
    def stop_drag(self, event):
        self.dragging = False
        
    def on_mousewheel(self, event):
        # Zoom in/out with mouse wheel
        if event.delta > 0:
            self.zoom_factor = min(3.0, self.zoom_factor + 0.1)
        else:
            self.zoom_factor = max(0.1, self.zoom_factor - 0.1)
        
        self.zoom_scale.set(self.zoom_factor)
        self.update_display()
        
    def select_file(self):
        self.json_path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if self.json_path:
            self.status_label.config(text=f"Loaded: {os.path.basename(self.json_path)}")
            self.visualize_boxes(self.json_path)
    
    def visualize_boxes(self, json_path):
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the base64 image from JSON
        image_data = data.get('image', '')
        if not image_data:
            self.status_label.config(text="Error: No image data found in JSON")
            return
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(image_bytes))
            # Convert to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.status_label.config(text=f"Error decoding image: {str(e)}")
            return
        
        # Store original image
        self.original_image = image.copy()
        
        # Create a copy for drawing
        vis_image = image.copy()
        
        # Get image dimensions
        height, width = vis_image.shape[:2]
        
        # Check if filtered_boxes exists in the JSON
        filtered_boxes = data.get('filtered_boxes', [])
        if filtered_boxes:
            self.status_label.config(text=f"Found {len(filtered_boxes)} filtered boxes")
            
            for box in filtered_boxes:
                # Get bbox coordinates (normalized coordinates)
                bbox = box.get('bbox', [])
                if len(bbox) == 4:  # Ensure we have all 4 coordinates
                    # Convert normalized coordinates to pixel coordinates
                    x1, y1, x2, y2 = bbox
                    x1_px = int(x1 * width)
                    y1_px = int(y1 * height)
                    x2_px = int(x2 * width)
                    y2_px = int(y2 * height)
                    
                    # Draw the box
                    cv2.rectangle(vis_image, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
                    
                    # Get the content
                    content = box.get('content', '')
                    
                    if content:
                        # Add text with background for better visibility
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        text = f": {content}"
                        
                        # Get text size
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Create background rectangle
                        padding = 5
                        bg_pts = np.array([
                            [x2_px + 5, y1_px - text_height - padding],
                            [x2_px + 5 + text_width + padding * 2, y1_px - text_height - padding],
                            [x2_px + 5 + text_width + padding * 2, y1_px + padding],
                            [x2_px + 5, y1_px + padding]
                        ], np.int32)
                        
                        # Draw semi-transparent background
                        overlay = vis_image.copy()
                        cv2.fillPoly(overlay, [bg_pts], (255, 255, 255))
                        cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
                        
                        # Draw text
                        cv2.putText(vis_image, text, (x2_px + 5 + padding, y1_px),
                                  font, font_scale, (0, 0, 255), thickness)
        
        # Store the visualization image
        self.displayed_image = vis_image
        
        # Reset scroll position
        self.scroll_x = 0
        self.scroll_y = 0
        
        # Update the display
        self.update_display()
    
    def update_display(self):
        if self.displayed_image is None:
            return
        
        # Apply zoom and rotation
        img = self.displayed_image.copy()
        
        # Apply rotation
        if self.rotation_angle != 0:
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        # Apply zoom
        if self.zoom_factor != 1.0:
            height, width = img.shape[:2]
            new_height = int(height * self.zoom_factor)
            new_width = int(width * self.zoom_factor)
            img = cv2.resize(img, (new_width, new_height))
        
        # Convert to PIL format for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Convert to PhotoImage for Tkinter
        self.photo = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Configure canvas scrolling region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Reset scroll position if zoomed out
        if self.zoom_factor <= 1.0:
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
    
    def update_zoom(self, value):
        self.zoom_factor = float(value)
        self.update_display()
    
    def update_rotation(self, value):
        self.rotation_angle = float(value)
        self.update_display()
    
    def reset_view(self):
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.zoom_scale.set(1.0)
        self.rotation_scale.set(0)
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizationApp(root)
    root.mainloop() 