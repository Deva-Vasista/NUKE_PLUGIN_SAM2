#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from pathlib import Path

def convert_mp4_to_exr(input_path, output_dir=None, start_frame=1):
    """
    Convert MP4 file to EXR frames using ffmpeg.
    
    Args:
        input_path (str): Path to input MP4 file
        output_dir (str, optional): Directory to save EXR frames. If None, uses input file directory
        start_frame (int, optional): Starting frame number for the sequence
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return False
    
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_exr"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct output pattern
    output_pattern = output_dir / f"{input_path.stem}_%04d.exr"
    
    # Construct ffmpeg command with corrected parameters
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-start_number', str(start_frame),
        '-vf', 'format=rgba',
        '-pix_fmt', 'gbrapf32le',
        '-compression_level', '0',
        str(output_pattern)
    ]
    
    try:
        print(f"Converting {input_path} to EXR frames...")
        subprocess.run(cmd, check=True)
        print(f"Conversion complete! EXR frames saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert MP4 files to EXR frames')
    parser.add_argument('input', help='Input MP4 file or directory containing MP4 files')
    parser.add_argument('-o', '--output', help='Output directory for EXR frames')
    parser.add_argument('-s', '--start-frame', type=int, default=1, help='Starting frame number')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        convert_mp4_to_exr(input_path, args.output, args.start_frame)
    elif input_path.is_dir():
        for mp4_file in input_path.glob('*.mp4'):
            convert_mp4_to_exr(mp4_file, args.output, args.start_frame)
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == '__main__':
    main() 