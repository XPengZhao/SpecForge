# 🚀 Get Started

## 📦 Installation

To install this project, you can simply run the following command.

- **Install from source (recommended)**

```bash
# git clone the source code
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

# create a new virtual environment
uv venv -p 3.11
source .venv/bin/activate

# install specforge
uv pip install -v . --prerelease=allow
```

- **Install from PyPI**

```bash
pip install specforge
```

## Accelerator-specific environments

### NVIDIA CUDA

The standard installation above uses the platform selected by PyTorch. Install
a CUDA build compatible with the host driver, then run every recipe through
the same `specforge train` entry.

### AMD ROCm

For the pinned ROCm environment, install the checked-in requirements before the
package:

```bash
python -m pip install -r requirements-rocm.txt
python -m pip install -e .
```

The file pins a ROCm 7.2 PyTorch stack. Use a wheel index and driver combination
compatible with the host if your ROCm version differs. Online runs require a
ROCm-compatible SGLang capture service; offline feature consumers can start
without target inference. PyTorch exposes ROCm accelerators through its
`torch.cuda` API and uses NCCL for distributed runs.

### Ascend NPU

Install the Ascend driver/firmware and the CANN toolkit first, matching the
`torch_npu` release you intend to use (see its release notes). Source CANN's
environment, then install the checked-in NPU requirements before the package:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -r requirements-npu.txt
python -m pip install -e .
```

`requirements-npu.txt` pins `torch==2.11.0` plus a matching `torch_npu`; adjust
the exact `torch_npu` patch to the release that matches torch 2.11.0. The
checked-in
[`qwen3.5-4b-dflash-online-npu.yaml`](../../examples/configs/qwen3.5-4b-dflash-online-npu.yaml),
[`qwen3.5-4b-domino-online-npu.yaml`](../../examples/configs/qwen3.5-4b-domino-online-npu.yaml),
and
[`deepseek-v4-flash-dspark-offline-npu.yaml`](../../examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml)
recipes use external SGLang server capture with SDPA consumers. `sglang` is a
hard import dependency: the stock wheel satisfies offline-training imports,
but online capture requires an NPU-compatible SGLang/Mooncake service. The
unified launcher detects the NPU device, self-launches the process count
recorded in YAML, and selects HCCL; see the
[training guide](../basic_usage/training.md#cuda-rocm-and-ascend-npu).
