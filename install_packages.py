import subprocess
import sys
import torch

def install_packages():
    # Install unsloth from the GitHub repository
    subprocess.run([sys.executable, "-m", "pip", "install", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"], check=True)

    # Check CUDA version and install additional dependencies accordingly
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version >= 8:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "packaging", "ninja", "einops", "flash-attn", "xformers", "trl", "peft", "accelerate", "bitsandbytes"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "xformers", "trl", "peft", "accelerate", "bitsandbytes"], check=True)

if __name__ == "__main__":
    install_packages()
