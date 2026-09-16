#!/bin/bash
set -e

ENV_NAME="q2j_estimating_information_bounds"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo " Setting up Conda Environment: ${ENV_NAME} "
echo " (Optimized for Blackwell cards) "

source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Create (or update) the environment from the pinned environment.yml:
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "-> Updating conda environment '${ENV_NAME}' from environment.yml..."
  conda env update -n "${ENV_NAME}" -f environment.yml
else
  echo "-> Creating conda environment '${ENV_NAME}' from environment.yml..."
  conda env create -f environment.yml
fi

# 2. Activate the environment
conda activate "${ENV_NAME}"

# 3. Verify installation 
echo "-> Verifying installation..."
python - <<'PYCHECK'
import sys
import torch

print(f"  PyTorch      : {torch.__version__}")
print(f"  Compiled CUDA: {torch.version.cuda}")
print(f"  Arch list    : {torch.cuda.get_arch_list()}")

if not torch.cuda.is_available():
    sys.exit(
        "\n  FAIL: CUDA is not available.\n"
        "  - 'Driver/library version mismatch' or 'CUDA error 804' => reboot;\n"
        "    a driver update landed without reloading the kernel module.\n"
        "  - Otherwise the wheel may not match the installed driver."
    )

cap = torch.cuda.get_device_capability(0)
sm = f"sm_{cap[0]}{cap[1]}"
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  Device       : {name}  ({sm}, {vram:.1f} GB)")
print(f"  bf16 support : {torch.cuda.is_bf16_supported()}")

if sm not in torch.cuda.get_arch_list():
    print(
        f"\n  WARNING: {sm} is not in this build's arch list. If the matmul"
        f"\n  below fails, reinstall torch from a CUDA index that supports"
        f"\n  {sm} (12.8+ for consumer Blackwell)."
    )

# Actually launch a kernel — the only real proof.
try:
    a = torch.randn(4096, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    out = float((a @ b).sum())
    torch.cuda.synchronize()
except Exception as e:
    sys.exit(f"\n  FAIL: kernel launch failed on {sm}: {type(e).__name__}: {e}")

if out != out:  # NaN
    sys.exit("\n  FAIL: matmul returned NaN.")
print(f"  Kernel launch: OK")
PYCHECK

echo " Setup complete! To activate your environment, run:"
echo " conda activate ${ENV_NAME}"
