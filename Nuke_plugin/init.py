import nuke
import os
from pathlib import Path
import sys

### SAM2
# Add plugin paths
nuke.pluginAddPath("./icons", addToSysPath=False)
nuke.pluginAddPath("./scripts", addToSysPath=False)

# Add virtual environment to Python path
current_dir = Path(__file__).parent.absolute()
venv_dir = current_dir / "venv"

if venv_dir.exists():
    if os.name == 'nt':  # Windows
        python_path = venv_dir / "Scripts"
        site_packages = venv_dir / "Lib" / "site-packages"
    else:  # Unix/Linux/Mac
        python_path = venv_dir / "bin"
        site_packages = venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    if python_path.exists():
        nuke.pluginAddPath(str(python_path), addToSysPath=True)
    if site_packages.exists():
        nuke.pluginAddPath(str(site_packages), addToSysPath=True)
else:
    nuke.tprint("Warning: Virtual environment not found. Please run setup.py first.")
###
