import torch
import platform

print("=== SYSTEM CHECK ===")
print("Platform:", platform.platform())

print("\n=== TORCH CHECK ===")
print("Torch version:", torch.__version__)

print("\n=== CUDA CHECK ===")
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
else:
    print("Running on CPU")

print("\n=== TEST TENSOR ===")
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Device:", x.device)

print("\n✅ Everything working!")