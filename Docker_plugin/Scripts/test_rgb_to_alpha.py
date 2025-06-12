import cv2
import numpy as np
import OpenEXR
import Imath
import array
import os
from pathlib import Path

def read_exr_to_numpy(file_path: str) -> np.ndarray:
    """Read EXR file and convert to numpy array."""
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get data window
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Get available channels
    available_channels = exr_file.header()['channels'].keys()
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # If we have RGB channels, read them
    if all(channel in available_channels for channel in ['R', 'G', 'B']):
        channels = ['R', 'G', 'B']
        channel_data = []
        for channel in channels:
            data = array.array('f', exr_file.channel(channel, FLOAT)).tolist()
            channel_data.append(np.array(data).reshape(height, width))
        img = np.dstack(channel_data)
    # If we have a single channel, read it
    elif 'A' in available_channels:
        data = array.array('f', exr_file.channel('A', FLOAT)).tolist()
        img = np.array(data).reshape(height, width)
    else:
        # Try to read any available channel
        channel = list(available_channels)[0]
        data = array.array('f', exr_file.channel(channel, FLOAT)).tolist()
        img = np.array(data).reshape(height, width)
    
    return img

def write_single_channel_exr(mask: np.ndarray, output_path: str):
    """Write numpy array to single-channel EXR file."""
    height, width = mask.shape[:2]
    
    # Create header with single channel
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    # Convert mask to float32
    mask_float = mask.astype(np.float32)
    
    # Create output file
    out = OpenEXR.OutputFile(output_path, header)
    
    # Write pixels with single channel
    out.writePixels({
        'A': mask_float.tobytes()
    })
    out.close()

def main():
    # Get input file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python test_rgb_to_alpha.py <input_exr_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Read the input EXR file
    print(f"Reading input file: {input_path}")
    mask = read_exr_to_numpy(input_path)
    
    # If mask is 3 channels, take one channel (they should all be the same)
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = mask[:, :, 0]
    
    # Create output path
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}_single{input_path.suffix}"
    
    # Write the mask as single channel
    print(f"Writing output file: {output_path}")
    write_single_channel_exr(mask, str(output_path))
    print("Done!")

if __name__ == "__main__":
    import sys
    main() 