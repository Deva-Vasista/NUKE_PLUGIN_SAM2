from setuptools import setup, find_packages

setup(
    name="nuke_samurai_api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-multipart==0.0.6",
        "numpy==1.24.3",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "opencv-python==4.8.1.78",
        "OpenEXR==1.3.9",
        "loguru==0.7.2",
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0"
    ],
) 