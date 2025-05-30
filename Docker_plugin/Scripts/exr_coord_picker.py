import cv2
import numpy as np
import OpenEXR
import Imath
import array
import sys
from pathlib import Path
import glob
import json
import os
import re
from collections import OrderedDict

def read_exr_to_numpy(file_path: str) -> np.ndarray:
    """Read EXR file and convert to numpy array."""
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get data window
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read all channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    channel_data = []
    
    for channel in channels:
        data = array.array('f', exr_file.channel(channel, FLOAT)).tolist()
        channel_data.append(np.array(data).reshape(height, width))
    
    # Stack channels
    img = np.dstack(channel_data)
    return img

class CoordinatePicker:
    def __init__(self, image):
        self.image = image.copy()
        self.display = image.copy()
        self.points_positive = []
        self.points_negative = []
        self.box_start = None
        self.box_end = None
        self.drawing = False
        self.current_mode = 'positive'  # 'positive' or 'negative'
        self.history = []  # For undo functionality
        self.drawing_box = False  # Flag to distinguish between point selection and box drawing
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:  # Shift + Left click for box
                self.drawing_box = True
                self.box_start = (x, y)
            else:  # Normal left click for points
                if self.current_mode == 'positive':
                    self.points_positive.append((x, y))
                else:
                    self.points_negative.append((x, y))
                self._save_state()
                self.display = self.image.copy()
                self._draw_all()
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_box:
                # Update box while dragging
                self.display = self.image.copy()
                self._draw_all()
                cv2.rectangle(self.display, self.box_start, (x, y), (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing_box:
                # Finish drawing box
                self.drawing_box = False
                self.box_end = (x, y)
                self._save_state()
                self.display = self.image.copy()
                self._draw_all()

    def _save_state(self):
        """Save current state for undo"""
        self.history.append({
            'points_positive': self.points_positive.copy(),
            'points_negative': self.points_negative.copy(),
            'box_start': self.box_start,
            'box_end': self.box_end
        })

    def undo(self):
        """Undo last action"""
        if self.history:
            state = self.history.pop()
            self.points_positive = state['points_positive']
            self.points_negative = state['points_negative']
            self.box_start = state['box_start']
            self.box_end = state['box_end']
            self.display = self.image.copy()
            self._draw_all()

    def _draw_all(self):
        """Draw all points and box"""
        # Draw positive points in green
        for pt in self.points_positive:
            cv2.circle(self.display, pt, 5, (0, 255, 0), -1)
        # Draw negative points in red
        for pt in self.points_negative:
            cv2.circle(self.display, pt, 5, (0, 0, 255), -1)
        # Draw box if exists
        if self.box_start and self.box_end:
            cv2.rectangle(self.display, self.box_start, self.box_end, (0, 255, 0), 2)

    def get_coordinates(self):
        coords = {
            'points_positive': self.points_positive,
            'points_negative': self.points_negative,
            'box': None
        }
        if self.box_start and self.box_end:
            x1, y1 = self.box_start
            x2, y2 = self.box_end
            coords['box'] = [x1, y1, x2, y2]
        return coords

    def set_coordinates(self, coords):
        self.points_positive = coords.get('points_positive', [])
        self.points_negative = coords.get('points_negative', [])
        box = coords.get('box')
        if box:
            self.box_start = (box[0], box[1])
            self.box_end = (box[2], box[3])
        self.display = self.image.copy()
        self._draw_all()

    def reset(self):
        """Reset all selections"""
        self.points_positive = []
        self.points_negative = []
        self.box_start = None
        self.box_end = None
        self.history = []
        self.display = self.image.copy()

class SequenceViewer:
    def __init__(self, sequence_path):
        self.sequence_path = sequence_path
        self.frame_files = self._get_frame_files()
        if not self.frame_files:
            raise ValueError(f"No EXR files found in: {sequence_path}")
        
        self.current_frame = 0
        self.selections = OrderedDict()  # frame_idx -> CoordinatePicker
        self.image_cache = {}  # Cache for loaded images
        self.cache_size = 3  # Reduced cache size to prevent memory issues
        self.load_selections()
        
    def _get_frame_files(self):
        """Get list of EXR files, handling both numbered files and sequence patterns"""
        if '%' in self.sequence_path:
            # Handle sequence pattern
            files = sorted(glob.glob(self.sequence_path))
        else:
            # Handle directory or single file
            if os.path.isfile(self.sequence_path):
                return [self.sequence_path]
            
            # Get all EXR files in directory
            dir_path = os.path.dirname(self.sequence_path)
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory not found: {dir_path}")
            
            # Find all EXR files and sort them
            files = []
            for f in os.listdir(dir_path):
                if f.endswith('.exr'):
                    # Extract frame number if present
                    match = re.search(r'(\d+)', f)
                    if match:
                        frame_num = int(match.group(1))
                        files.append((frame_num, os.path.join(dir_path, f)))
            
            # Sort by frame number and get file paths
            files = [f[1] for f in sorted(files)]
        
        return files
        
    def load_selections(self):
        """Load saved selections from JSON file if it exists"""
        json_path = os.path.splitext(self.sequence_path)[0] + '_selections.json'
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    saved_data = json.load(f)
                    for frame_idx, coords in saved_data.items():
                        picker = CoordinatePicker(self.get_current_image())
                        picker.set_coordinates(coords)
                        self.selections[int(frame_idx)] = picker
            except Exception as e:
                print(f"Error loading saved selections: {e}")

    def save_selections(self):
        """Save current selections to JSON file"""
        json_path = os.path.splitext(self.sequence_path)[0] + '_selections.json'
        try:
            # Convert selections to serializable format
            serializable_selections = {}
            for frame_idx, picker in self.selections.items():
                if picker.points_positive or picker.points_negative or picker.box_start:  # Only save frames with selections
                    serializable_selections[str(frame_idx)] = picker.get_coordinates()
            
            with open(json_path, 'w') as f:
                json.dump(serializable_selections, f, indent=2)
            print(f"Selections saved to {json_path}")
        except Exception as e:
            print(f"Error saving selections: {e}")

    def get_current_image(self):
        """Load and return the current frame image with caching"""
        if self.current_frame in self.image_cache:
            return self.image_cache[self.current_frame]
        
        # Load the image
        img = read_exr_to_numpy(self.frame_files[self.current_frame])
        
        # Normalize image while preserving quality
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > min_val:
            img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
        
        # Update cache
        self.image_cache[self.current_frame] = img
        
        # Remove oldest cached image if cache is full
        if len(self.image_cache) > self.cache_size:
            oldest_frame = min(self.image_cache.keys())
            del self.image_cache[oldest_frame]
        
        return img

    def get_current_picker(self):
        """Get or create CoordinatePicker for current frame"""
        if self.current_frame not in self.selections:
            img = self.get_current_image()
            picker = CoordinatePicker(img)
            self.selections[self.current_frame] = picker
        return self.selections[self.current_frame]

    def next_frame(self):
        if self.current_frame < len(self.frame_files) - 1:
            self.current_frame += 1
            # Force reload of current frame
            if self.current_frame in self.image_cache:
                del self.image_cache[self.current_frame]
            return True
        return False

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            # Force reload of current frame
            if self.current_frame in self.image_cache:
                del self.image_cache[self.current_frame]
            return True
        return False

    def clear_empty_selections(self):
        """Remove frames with no selections"""
        frames_to_remove = []
        for frame_idx, picker in self.selections.items():
            if not picker.points_positive and not picker.points_negative and not picker.box_start:
                frames_to_remove.append(frame_idx)
        
        for frame_idx in frames_to_remove:
            del self.selections[frame_idx]

def main():
    if len(sys.argv) != 2:
        print("Usage: python exr_coord_picker.py <path_to_exr_sequence>")
        print("Example: python exr_coord_picker.py '/path/to/frames/frame_%04d.exr'")
        print("Example: python exr_coord_picker.py '/path/to/frames/'")
        return

    sequence_path = sys.argv[1]
    try:
        viewer = SequenceViewer(sequence_path)
    except Exception as e:
        print(f"Error initializing sequence viewer: {e}")
        return

    # Create window with a smaller size
    cv2.namedWindow('EXR Sequence Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EXR Sequence Viewer', 800, 600)  # Set initial window size
    
    print("\nInstructions:")
    print("- Press 'P' for positive points mode (green)")
    print("- Press 'N' for negative points mode (red)")
    print("- Left click to add points in current mode")
    print("- Shift + Left click and drag to draw a box")
    print("- Press 'A' for previous frame")
    print("- Press 'D' for next frame")
    print("- Press 'R' to reset current frame selections")
    print("- Press 'Z' to undo last selection")
    print("- Press 'Q' to quit and save selections")

    # Set up mouse callback
    def mouse_callback(event, x, y, flags, param):
        viewer.get_current_picker().mouse_callback(event, x, y, flags, param)

    cv2.setMouseCallback('EXR Sequence Viewer', mouse_callback)

    while True:
        picker = viewer.get_current_picker()
        cv2.imshow('EXR Sequence Viewer', picker.display)
        
        # Update window title with frame info and current mode
        frame_name = os.path.basename(viewer.frame_files[viewer.current_frame])
        mode = "Positive" if picker.current_mode == 'positive' else "Negative"
        cv2.setWindowTitle('EXR Sequence Viewer', 
                         f'Frame {viewer.current_frame + 1}/{len(viewer.frame_files)} - {frame_name} - Mode: {mode}')
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            picker.reset()
        elif key == ord('d'):
            if viewer.next_frame():
                picker = viewer.get_current_picker()
        elif key == ord('a'):
            if viewer.prev_frame():
                picker = viewer.get_current_picker()
        elif key == ord('p'):
            picker.current_mode = 'positive'
        elif key == ord('n'):
            picker.current_mode = 'negative'
        elif key == ord('z'):
            picker.undo()

    cv2.destroyAllWindows()

    # Clear empty selections before saving
    viewer.clear_empty_selections()
    
    # Save final selections
    viewer.save_selections()

    # Print all selections
    print("\nAll Frame Selections:")
    for frame_idx, picker in sorted(viewer.selections.items()):
        frame_name = os.path.basename(viewer.frame_files[frame_idx])
        print(f"\nFrame {frame_idx + 1} ({frame_name}):")
        coords = picker.get_coordinates()
        if coords['points_positive']:
            print("  Positive Points:")
            for pt in coords['points_positive']:
                print(f"    [{pt[0]}, {pt[1]}]")
        if coords['points_negative']:
            print("  Negative Points:")
            for pt in coords['points_negative']:
                print(f"    [{pt[0]}, {pt[1]}]")
        if coords['box']:
            print(f"  Box: {coords['box']}")

if __name__ == "__main__":
    main() 