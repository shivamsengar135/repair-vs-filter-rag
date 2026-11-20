"""Phase 0: confirm torch sees the GPU and can run a real computation on it."""
import sys
import time

import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__, "| built with CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FAIL: torch cannot see a CUDA device")
print("device:", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"VRAM free/total: {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
print("bf16 supported:", torch.cuda.is_bf16_supported())
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.perf_counter()
y = (x @ x).float().abs().mean().item()
torch.cuda.synchronize()
print(f"8192x8192 bf16 matmul on GPU ok ({time.perf_counter() - t0:.3f}s), mean|.|={y:.3f}")
print("TORCH_GPU_OK")
