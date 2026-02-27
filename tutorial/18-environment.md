# 5.3 环境配置与依赖安装

> 📍 **问题**：租到GPU后，我应该如何在实例上安装SimpleLLM的运行环境？

---

## 5.3.1 登录到GPU实例

### 📝 连接方式

假设你已经启动了RunPod实例，现在需要连接到它。

```bash
# 方式1：使用RunPod终端（推荐）
# 在RunPod网页点击 "Connect" -> "RunPod Terminal"

# 方式2：使用SSH
ssh root@<your-instance-ip>
```

### 📝 验证GPU

```bash
# 检查GPU是否可用
nvidia-smi
```

如果看到类似以下输出，说明GPU正常：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100 80...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   40C    P0    80W / 700W |     10MiB / 81920MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## 5.3.2 创建工作目录

```bash
# 创建项目目录
mkdir -p /workspace/simple-llm
cd /workspace/simple-llm
```

---

## 5.3.3 克隆SimpleLLM仓库

### 📝 步骤

```bash
# 克隆仓库
git clone https://github.com/anomalyco/simple-llm.git .

# 或者如果已经在本地有仓库，可以跳过这一步
```

---

## 5.3.4 安装Python环境

### 📝 检查Python版本

```bash
# 检查Python版本
python3 --version

# 如果是3.12+，继续
# 如果不是，需要安装Python 3.12+
```

### 📝 使用setup.sh自动安装

SimpleLLM提供了自动安装脚本：

```bash
# 运行安装脚本
# 默认会创建 .venv 虚拟环境并安装依赖
./setup.sh
```

### 📝 手动安装（如果需要）

如果自动安装失败，可以手动安装：

```bash
# 1. 安装uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. 创建虚拟环境
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3. 安装PyTorch
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 4. 安装Flash Attention（重要！）
# 从GitHub下载预编译的wheel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 5. 安装其他依赖
pip install numpy safetensors tokenizers huggingface_hub einops triton tqdm
```

---

## 5.3.5 下载模型

### 📝 问题

**问题**：模型文件很大，需要多长时间？

**答案**：
- gpt-oss-120b模型约200GB+
- 下载时间取决于网络速度
- 建议使用HuggingFace镜像或直接下载

### 📝 下载步骤

```bash
# 方法1：使用HuggingFace CLI（推荐）
# 需要先登录 HuggingFace
huggingface-cli login

# 下载模型
hf download openai/gpt-oss-120b --local-dir ./gpt-oss-120b

# 方法2：如果下载失败，可以尝试
# 从其他镜像源下载
```

### ⚠️ 重要提示

- 模型很大，确保存储空间足够（至少200GB）
- 如果下载中断，可以重新执行，会断点续传

---

## 5.3.6 验证安装

### 📝 问题

**问题**：如何确认环境配置正确？

**答案**：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 验证Python包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import flash_attn; print('Flash Attention: OK')"

# 验证CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 预期输出

```
PyTorch: 2.9.1+cu128
Triton: 3.x.x
Flash Attention: OK
CUDA: True
GPU: NVIDIA H100 80GB
```

---

## 5.3.7 常见问题排查

### 📝 问题1：CUDA版本不匹配

```
Error: CUDA version mismatch
```

**解决方案**：
- 确保使用CUDA 12.8的PyTorch版本
- 检查nvidia-smi显示的CUDA版本

### 📝 问题2：Flash Attention安装失败

```
ERROR: Could not build wheels for flash-attn
```

**解决方案**：
- 确保是CUDA 12.x环境
- 尝试使用预编译的wheel文件
- 或者从源码编译

### 📝 问题3：模型下载失败

```
Error: Connection error
```

**解决方案**：
- 检查网络连接
- 尝试使用镜像
- 或手动下载后上传

---

## 5.3.8 本节小结

### ✅ 关键要点

1. **连接实例**：通过RunPod终端或SSH
2. **克隆仓库**：从GitHub获取代码
3. **安装依赖**：PyTorch + Flash Attention + Triton
4. **下载模型**：确保存储空间足够
5. **验证安装**：检查所有依赖是否正常

---

## 下节预告

> [5.4 运行SimpleLLM推理](./19-running-inference.md) 将指导你运行第一次推理。
