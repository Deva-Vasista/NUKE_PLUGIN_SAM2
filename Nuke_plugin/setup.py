import os
import sys
import subprocess
import venv
from pathlib import Path

def setup_venv():
    # Get the directory where this script is located
    current_dir = Path(__file__).parent.absolute()
    venv_dir = current_dir / "venv"
    
    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
    
    # Determine the pip path based on the OS
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip"
    else:
        pip_path = venv_dir / "bin" / "pip"
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([str(pip_path), "install", "-r", str(current_dir / "requirements.txt")])
    
    # Create a .env file with the Python path
    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python"
    else:
        python_path = venv_dir / "bin" / "python"
    
    with open(current_dir / ".env", "w") as f:
        f.write(f"PYTHONPATH={python_path.parent}\n")
    
    print("Setup completed successfully!")
    print(f"Virtual environment created at: {venv_dir}")
    print(f"Python executable: {python_path}")

if __name__ == "__main__":
    setup_venv() 