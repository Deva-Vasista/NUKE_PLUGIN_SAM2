from setuptools import setup, find_packages
import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        sys.exit("Python >= 3.8 is required")

def setup_sam2():
    """Setup SAM2 repository and download checkpoints if needed."""
    # Setup paths
    sam2_repo_path = os.path.join("NukeSamurai", "sam2_repo")
    checkpoints_dir = os.path.join(sam2_repo_path, "sam2", "checkpoints")
    
    # Create checkpoints directory
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Install SAM2 repository if it exists
    if os.path.exists(sam2_repo_path):
        print("Installing SAM2 repository in development mode...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", sam2_repo_path])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{sam2_repo_path}[notebooks]"])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install SAM2 repository: {e}")
            print("You may need to install it manually.")
    else:
        print(f"Warning: SAM2 repository not found at {sam2_repo_path}")
        print("Please ensure you have cloned the repository correctly.")

def create_config():
    """Create default configuration file if it doesn't exist."""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    default_config = """
# Default configuration for NukeSamurai API
api_host: "0.0.0.0"
api_port: 8000
log_level: "INFO"
output_dir: "Output"
max_batch_size: 32
cuda_device: 0  # Set to -1 for CPU
"""
    
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        print("Creating default configuration file...")
        config_file.write_text(default_config.strip())

def main():
    # Check Python version
    check_python_version()
    
    # Read requirements
    with open("requirements.txt") as f:
        requirements = [line.strip() for line in f.readlines() 
                       if line.strip() and not line.startswith("#")]
    
    # Setup SAM2
    setup_sam2()
    
    # Create default config
    create_config()
    
    # Setup package
    setup(
        name="nuke_samurai_api",
        version="0.1.0",
        description="NukeSamurai API for EXR image segmentation using SAM2",
        author="NukeSamurai Team",
        packages=find_packages(),
        install_requires=requirements,
        entry_points={
            'console_scripts': [
                'nuke-samurai-server=src.cli:run_server',
                'nuke-samurai-test=src.cli:run_tests',
            ],
        },
        python_requires=">=3.8",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Multimedia :: Graphics",
        ],
        include_package_data=True,
        zip_safe=False,
    )

if __name__ == "__main__":
    main() 