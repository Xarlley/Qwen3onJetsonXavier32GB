# 在 Jetson AGX Xavier 32GB 上部署 Qwen3（vLLM 0.6.3 + GPTQ）

一份**可逐步照做**的复现教程：在 **NVIDIA Jetson AGX Xavier（32GB）** 上，用 **可编辑安装的 vLLM 0.6.3** + **纯 Python Qwen3 移植**，运行 **GPTQ Int4** 的 Instruct 模型，并提供 OpenAI 兼容 HTTP 服务。

> 目标硬件算力有限（Volta，`sm_72`）。本仓库记录的是**能跑通且可用**的路径，不是桌面 4090 / 数据中心 GPU 的性能对标方案。

- **实测量级（单流 chat，暖机后）：** 约 **13–15 tok/s**，首 token（TTFT）可到约 **0.2 s** 量级（与 `max_tokens`、上下文、温度有关）。
- **语言：** 简体中文为主。
- **许可：** [MIT](./LICENSE)。模型权重遵循其原仓库协议（Hugging Face / ModelScope）。

---

## 目录

1. [适用场景与不适用范围](#1-适用场景与不适用范围)
2. [硬件与软件基线](#2-硬件与软件基线)
3. [红线（必读）](#3-红线必读)
4. [总体架构](#4-总体架构)
5. [步骤 A：系统与 CUDA / cuDNN](#5-步骤-a系统与-cuda--cudnn)
6. [步骤 B：从源码构建 PyTorch（sm_72）](#6-步骤-b从源码构建-pytorchsm_72)
7. [步骤 C：XFormers 与可编辑 vLLM](#7-步骤-cxformers-与可编辑-vllm)
8. [步骤 D：Qwen3 纯 Python 移植](#8-步骤-dqwen3-纯-python-移植)
9. [步骤 E：下载 GPTQ 权重](#9-步骤-e下载-gptq-权重)
10. [步骤 F：启动 OpenAI 兼容服务](#10-步骤-f启动-openai-兼容服务)
11. [步骤 G：验证、测速与客户端](#11-步骤-g验证测速与客户端)
12. [常见故障](#12-常见故障)
13. [性能与「可用」门槛](#13-性能与可用门槛)
14. [仓库文件说明](#14-仓库文件说明)
15. [TODO / 已知缺口](#15-todo--已知缺口)

---

## 1. 适用场景与不适用范围

| 适合 | 不适合 / 需另找方案 |
|------|---------------------|
| Jetson AGX Xavier 32GB，JetPack 5.1.x | Orin / 新 JetPack 6 可能路径不同，需自行验证 |
| 本地离线或局域网推理、学习、边缘原型 | 高并发 SaaS、长上下文大批量离线 |
| Qwen3 **4B 级 GPTQ** | 原版 AWQ（Xavier `sm_72` **不够**）、依赖 FlashAttention 的路径 |
| OpenAI 兼容 `/v1/chat/completions` | 指望官方 vLLM ≥0.8 开箱 Qwen3（本板卡停在 0.6.3 栈） |

---

## 2. 硬件与软件基线

以下为本教程验证过的组合（名称请按你的板子核对，勿假设版本可随意互换）：

| 组件 | 版本 / 说明 |
|------|-------------|
| 板卡 | Jetson AGX Xavier **32GB**，GPU 架构 **Volta `sm_72`** |
| 系统 | **JetPack 5.1.7** / **L4T R35.6.5**（或同系 JP5.1.x，需自测） |
| CUDA（用户态） | **12.2**，并依赖 **`/usr/local/cuda/compat`（cuda-compat）** |
| 驱动侧 | JP5 驱动偏 CUDA 11.4 系 → **必须**用 compat，否则易报 driver too old |
| cuDNN | **9**（示例路径 `/opt/cudnn9`） |
| Python | **3.8**（JP5 常见） |
| PyTorch | **2.4.1**，**自源码**编到 `TORCH_CUDA_ARCH_LIST=7.2` |
| Attention | **XFormers**（**不是** `vllm_flash_attn` / FlashAttention；后者运行时要 sm≥80） |
| 推理引擎 | **vLLM 0.6.3.post\***，**editable** 安装在 `~/src/vllm` |
| 量化 | **GPTQ only**；**不要**在本板上走 AWQ（需 sm≥75） |
| 示例权重 | `Qwen3-4B-Instruct-2507-GPTQ-Int4`（HF / ModelScope 同名仓库，作者以镜像站为准） |

板端用户名下文写 `nvidia`（许多镜像默认账户）；若你的账户不同，把路径中的用户名换成自己的，或一律用 `~`。

---

## 3. 红线（必读）

1. **禁止 `sudo` 跑 vLLM / `python -m vllm...`**  
   `sudo` 会丢掉用户环境里的 `LD_LIBRARY_PATH`，**cuda-compat** 失效 → 典型报错类似 *NVIDIA driver too old (11040)*。
2. **`OPENBLAS_NUM_THREADS=1`**  
   部分 Jetson 上自带 OpenBLAS 多线程会导致 **NaN**。
3. **`LD_LIBRARY_PATH` 必须含** `/usr/local/cuda/compat`（以及你的 cuDNN，如 `/opt/cudnn9/lib`）。建议写入 `/etc/environment`，启动脚本再强制 export 一次。
4. **pip 重装 vLLM 必须 `--no-deps`**  
   否则可能把辛苦编好的 `torch` 覆盖成 PyPI 轮子。
5. **改 `.py` 后必须重启服务**（已 import 的模块不会热更新）。改 C++/CUDA 才需要 `build_ext` / 完整重编。
6. **磁盘极紧**（eMMC 常见只剩数 GB）  
   只留一份 GPTQ；及时清理 HF/ModelScope 半截缓存；编译可用 tmpfs（见下文 rambuild）。
7. **隐私与安全**  
   勿把本机密码、私钥、内网拓扑写进公开仓库；SSH 密钥权限自管；示例密码一律自设为强口令。

---

## 4. 总体架构

```
YOUR_PC (SSH / 浏览器 / PowerShell 客户端)
        |  USB 以太网或局域网
        v
Jetson AGX Xavier
  ~/models/<GPTQ 权重>
  ~/src/vllm          ← editable + Qwen3 Python 移植
  python -m vllm.entrypoints.openai.api_server
        → 0.0.0.0:8000
        → /v1/chat/completions  （推荐）
        → /v1/models , /health
```

权重只提供参数；**`head_dim` / Q-K RMSNorm / 注册表** 由 vLLM 的 Python 模型文件决定。官方 vLLM 约从 **0.8.4** 才原生带 Qwen3；在 0.6.3 上需要本教程的移植步骤。

---

## 5. 步骤 A：系统与 CUDA / cuDNN

1. 刷好 **JetPack 5.1.7**（或确认 `cat /etc/nv_tegra_release` 为 R35.6.x 一类）。
2. 安装 / 确认 **CUDA 12.2** 工具链与 **cuda-compat** 包，使目录存在：  
   `/usr/local/cuda/compat`
3. 安装 **cuDNN 9** 到例如 `/opt/cudnn9`（路径以你实际为准）。
4. 编辑 `/etc/environment`（需管理员权限；内容按你的路径调整）：

```bash
CUDA_HOME="/usr/local/cuda"
LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64:/opt/cudnn9/lib"
OPENBLAS_NUM_THREADS="1"
```

5. 重新登录后自检（**不要 sudo**）：

```bash
. /etc/environment
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/opt/cudnn9/lib:${LD_LIBRARY_PATH}"
nvcc --version
python3 -c "import sys; print(sys.version)"
```

6. （可选）风扇 / 功耗：`nvpmodel`、`jetson_clocks`、厂商风扇模式。长压测时注意温度与降频；非必须。

---

## 6. 步骤 B：从源码构建 PyTorch（sm_72）

PyPI 上的 torch 轮子往往**不含** `sm_72`。需要从源码构建并安装到用户目录（示意）：

```bash
export TORCH_CUDA_ARCH_LIST=7.2
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
# 按你习惯 clone pytorch 对应 2.4.1 标签，安装依赖后：
# python3 setup.py develop --user
# 或官方文档推荐的 pip 可编辑安装方式
```

验证：

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.4.1 ... True ... Xavier
```

> **TODO：** 完整 torch 编译参数与依赖包列表因镜像而异，请以你当时的构建日志为准；失败时优先检查 `sm_72`、cuda-compat、内存（建议加大 swap）。

---

## 7. 步骤 C：XFormers 与可编辑 vLLM

### 7.1 XFormers

在 Volta 上 vLLM 会走 **XFormers** 后端。需自行编译/安装与当前 torch 匹配的 xformers（`TORCH_CUDA_ARCH_LIST=7.2`）。  
日志中应出现类似：`Using XFormers backend`。

### 7.2 获取 vLLM 0.6.3 源码并 editable 安装

```bash
mkdir -p ~/src
cd ~/src
# 取得 v0.6.3 / 0.6.3.post* 源码树，目录名建议：
# ~/src/vllm

export TORCH_CUDA_ARCH_LIST=7.2
export MAX_JOBS=4          # 视内存调整；Xavier 上勿过大
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

cd ~/src/vllm
pip3 install --user --no-deps --no-build-isolation -e .
```

**关键：** 始终带 **`--no-deps`**。

### 7.3 可选：tmpfs「rambuild」减轻 eMMC 压力

编译产物可放到内存盘，例如：

```bash
sudo mkdir -p /mnt/rambuild
# 按需 mount tmpfs 到 /mnt/rambuild
cd ~/src/vllm
ln -sfn /mnt/rambuild/vllm_build build
ln -sfn /mnt/rambuild/vllm_deps .deps
```

**注意：** 重启后 tmpfs 清空，符号链接还在但目录空 → 需要重建 `build`/`.deps` 后再编译。可用脚本封装（示例名 `~/build_vllm.sh`）。

仅改 Python **不必**重跑完整编译；改 CUDA/C++ 才需要 `build_ext` 或重装 editable。

---

## 8. 步骤 D：Qwen3 纯 Python 移植

### 8.1 为何要移植？

- 权重 `config.json`：`model_type=qwen3`，`architectures=["Qwen3ForCausalLM"]`，**`head_dim=128`**（4B）。
- 若误用 Qwen2 公式 `hidden_size // num_heads` → `2560/32=80`，**维度错误**。
- Qwen3 还有 **`attention_bias=false`**、**Q/K 上的 RMSNorm**（在 RoPE 前）。
- transformers 4.46 时代**没有**官方 `Qwen3Config` → 要在 vLLM 的 `_CONFIG_REGISTRY` 注册。

对照上游：可读 vLLM **≥0.8.4** 的 `qwen3.py` 作架构参考，但 API 需**对齐你本地的 `qwen2.py`（0.6.3）**（`kv_cache` / `AttentionMetadata` 等），不要生搬 `VllmConfig` 新接口。

### 8.2 需要动到的文件（均在 `~/src/vllm`）

| # | 相对路径 | 操作 |
|---|----------|------|
| 1 | `vllm/transformers_utils/configs/qwen3.py` | **新建** `Qwen3Config`（含 `head_dim`） |
| 2 | `vllm/transformers_utils/configs/__init__.py` | 导出 |
| 3 | `vllm/transformers_utils/config.py` | `_CONFIG_REGISTRY["qwen3"] = Qwen3Config` |
| 4 | `vllm/model_executor/models/qwen3.py` | **新建** 模型（对齐本地 qwen2 API） |
| 5 | `vllm/model_executor/models/registry.py` | `"Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM")` |
| 6 | `vllm/sampling_params.py` | 空 guided 参数时返回 `None`（见下） |

详细设计说明见 [docs/qwen3-backport.md](./docs/qwen3-backport.md)。

### 8.3 Guided decoding / outlines 坑

`GuidedDecodingParams.from_optional(...)` 在**没有任何** json/regex/grammar 时若仍返回空对象，请求可能 lazy-import **outlines** → 拖入旧依赖 → `numpy.bool` 等错误 → HTTP 500。  
**修复：** 全部引导字段为 `None` 时 **返回 `None`**，普通 chat 不再碰 outlines。

可选：在 `sitecustomize.py` 里为 `numpy` 补回已移除的别名（仅缓解其它路径；有 guided 补丁后日常 chat 可不依赖）。

### 8.4 移植自检

```bash
cd ~/src/vllm
python3 - << 'PY'
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.configs.qwen3 import Qwen3Config
assert "qwen3" in _CONFIG_REGISTRY
cls = ModelRegistry._try_load_model_cls("Qwen3ForCausalLM")
print("cls", cls)
c = Qwen3Config(hidden_size=2560, num_attention_heads=32, head_dim=128)
assert c.head_dim == 128
print("VERIFY_OK")
PY
```

---

## 9. 步骤 E：下载 GPTQ 权重

示例模型目录：

```bash
mkdir -p ~/models
# 最终希望存在：
# ~/models/Qwen3-4B-Instruct-2507-GPTQ-Int4/
```

### 9.1 网络

Jetson 常需经 **YOUR_PC 代理** 访问外网，例如 SSH 反向隧道（端口号按你的代理监听修改）：

```bash
# 在 YOUR_PC 上（示例）
ssh -N -R 7897:127.0.0.1:7897 USER@JETSON_IP
```

板端：

```bash
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
```

### 9.2 Hugging Face 失败时 → ModelScope

HF 可能因 XET、TLS、断连失败。可改用 **ModelScope** `snapshot_download` 或镜像站 `wget`，再整理到固定路径 `~/models/Qwen3-4B-Instruct-2507-GPTQ-Int4`。

验货：

```bash
python3 - << 'PY'
import json
p = __import__("os").path.expanduser("~/models/Qwen3-4B-Instruct-2507-GPTQ-Int4/config.json")
c = json.load(open(p))
print(c["model_type"], c["architectures"], c.get("head_dim"),
      c.get("quantization_config", {}).get("quant_method"))
# 期望: qwen3  ['Qwen3ForCausalLM']  128  gptq
PY
df -h ~
```

---

## 10. 步骤 F：启动 OpenAI 兼容服务

1. 复制示例脚本并安装：

```bash
cp scripts/start_vllm_qwen3.sh.example ~/start_vllm_qwen3.sh
chmod +x ~/start_vllm_qwen3.sh
# 按需编辑 MODEL_DIR / PORT / chat-template
```

2. 启动（**禁止 sudo**）：

```bash
nohup ~/start_vllm_qwen3.sh > ~/vllm_qwen3.log 2>&1 &
```

3. 建议参数含义：

| 参数 | 建议 | 说明 |
|------|------|------|
| `--dtype float16` | 是 | |
| `--max-model-len 2048` | 起步值 | 显存紧时勿盲目加大 |
| `--gpu-memory-utilization 0.85` | 左右 | Xavier 共享内存 |
| `--enforce-eager` | 常用 | 减少 CUDA graph 边角问题 |
| `--host 0.0.0.0 --port 8000` | 局域网访问 | |

4. 健康检查：

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health
# 期望 200
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool | head
```

5. 重启：

```bash
pkill -f 'vllm.entrypoints.openai.api_server' || true
# 若端口仍占用：fuser -k 8000/tcp
sleep 2
nohup ~/start_vllm_qwen3.sh > ~/vllm_qwen3.log 2>&1 &
```

日志中应看到 XFormers、模型加载完成、Uvicorn 监听 8000。

---

## 11. 步骤 G：验证、测速与客户端

### 11.1 请用 Chat，不要只用 Completions

Instruct 模型请走 **`/v1/chat/completions`**（带 messages / chat template）。  
`/v1/completions` 常不套模板，**质量差**，仅适合调试引擎是否通。

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-4B-Instruct-2507-GPTQ-Int4",
    "messages": [{"role":"user","content":"用一句话介绍你自己"}],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

注意：`--model` 若为本地路径，`/v1/models` 返回的 **id 可能是绝对路径**；请求里的 `"model"` 字段需与之一致（或按你启动方式填写）。

### 11.2 Windows / YOUR_PC 连续对话

仓库提供 `scripts/chat.ps1`（PowerShell 多轮 + 可选流式）：

```powershell
cd C:\path\to\this\repo\scripts
powershell -ExecutionPolicy Bypass -File .\chat.ps1 -ShowTiming
```

**USB 网卡常见坑：** 板端可能是 `192.168.100.2/24`，而 PC 端 USB 网卡只有 **169.254.x.x（APIPA）** 时，**直连 HTTP 会超时**，但 **IPv6 链路本地 SSH** 仍可能可用。此时在 YOUR_PC：

```powershell
ssh -N -L 18000:127.0.0.1:8000 USER@JETSON
# 另开窗口：
.\chat.ps1 -BaseUrl http://127.0.0.1:18000/v1
```

默认 BaseUrl 示例为 `http://192.168.100.2:8000/v1`（请改成你的 IP）。链路本地 SSH 的 `%zone` / `%%zone` 索引在换 USB 口后可能变化，需在 `~/.ssh/config` 中更新。

### 11.3 粗测吞吐

固定 `max_tokens`，用壁钟估算输出 tok/s；正式指标（TTFT、TPOT、并发）见 [docs/benchmark-notes.md](./docs/benchmark-notes.md)。

---

## 12. 常见故障

| 现象 | 排查 |
|------|------|
| driver too old / 11040 | 是否 sudo？`echo $LD_LIBRARY_PATH` 有无 `cuda/compat` |
| NaN / 诡异输出 | `OPENBLAS_NUM_THREADS` 是否为 1 |
| `numpy.bool` / outlines 500 | guided → `None` 补丁；可选 sitecustomize |
| HF 下载失败 | 代理隧道；改 ModelScope |
| Address already in use | `fuser -k 8000/tcp` 后重启 |
| chat 格式乱 | `--chat-template` / tokenizer_config |
| OOM / 加载失败 | 降 `max-model-len`、`gpu-memory-utilization`；确认是 GPTQ 而非更大精度 |
| Windows 连不上 `:8000` | 网段是否一致；改用 SSH `-L` 转发 |
| 改代码不生效 | 是否重启了 api_server |

---

## 13. 性能与「可用」门槛

在本板本模型上，可参考：

| 项 | 建议 Pass |
|----|-----------|
| `/health` | 持续 200 |
| 单流暖机输出 | **≥ 10 tok/s**（实测常约 13–15） |
| 短 prompt TTFT | p50 约 ≤ 1 s（边缘板宽松 SLA） |
| 正确性 | 中文短答连贯；无 NaN；无 500 |
| 并发 | 从 1→2→4 慢升；Xavier 内存紧，破线即停 |

不要与旗舰桌面 GPU 比绝对值；比的是**本机交互是否可接受**。

---

## 14. 仓库文件说明

```
README.md                          ← 你正在读的主教程
LICENSE
.gitignore
scripts/start_vllm_qwen3.sh.example
scripts/chat.ps1                   ← Windows 多轮客户端
docs/qwen3-backport.md             ← 移植细节
docs/benchmark-notes.md            ← 测速指标与计划摘要
patches/README.md                  ← 补丁说明（不绑私人路径）
```

---

## 15. TODO / 已知缺口

- [ ] 未在本仓库附带完整 `qwen3.py` 源码快照（请按 [docs/qwen3-backport.md](./docs/qwen3-backport.md) 对照上游 0.8.4 + 本地 `qwen2.py` 自行移植；欢迎 PR 合法补丁）。
- [ ] PyTorch / xformers **逐步编译命令**因环境差异大，仅给原则与检查点；欢迎补充「从零 docker/脚本」PR。
- [ ] 未覆盖 JetPack 6 / Orin / 官方 vLLM 新版本原生 Qwen3。
- [ ] ShareGPT 全量 `benchmark_serving.py` 需额外磁盘与数据集，默认以轻量 chat 测试为准。
- [ ] 风扇与 nvpmodel 仅作提示，未绑定某一厂商配置文件。

---

## 致谢与引用

- [vLLM](https://github.com/vllm-project/vllm)（0.6.3 与上游 Qwen3 支持）
- Qwen3 模型与 GPTQ 权重发布方（Hugging Face / ModelScope）
- NVIDIA Jetson / JetPack 文档

本教程整理自真实 Xavier 32GB 部署经验，已去除私人主机名、用户目录与凭据；若步骤与你的镜像冲突，以你板子上的日志为准，并欢迎提 Issue / PR。
