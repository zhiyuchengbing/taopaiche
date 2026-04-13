import torch
print("Torch version:", torch.__version__)
print("CUDA version inside torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
