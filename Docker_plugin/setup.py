from setuptools import setup, find_packages
import os
import subprocess

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

# Optional: Download SAM2 checkpoints if they don't exist
checkpoints_dir = os.path.join("NukeSamurai", "sam2_repo", "sam2", "checkpoints")
if not os.path.exists(checkpoints_dir):
    print("Creating checkpoints directory...")
    os.makedirs(checkpoints_dir, exist_ok=True)

# Check if we need to install the SAM2 repo
sam2_repo_path = os.path.join("NukeSamurai", "sam2_repo")
if os.path.exists(sam2_repo_path):
    print("SAM2 repository found. Will install it in development mode.")
    try:
        # This will be executed when setup.py is run with pip install -e .
        subprocess.check_call(["pip", "install", "-e", sam2_repo_path])
        subprocess.check_call(["pip", "install", "-e", f"{sam2_repo_path}[notebooks]"])
    except subprocess.CalledProcessError:
        print("Warning: Failed to install SAM2 repository. You may need to install it manually.")

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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 