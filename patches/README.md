# patches/

本目录提供针对 **vLLM 0.6.3** 的 **Qwen3 纯 Python 移植** 补丁（已脱敏，无私人路径/邮箱）。

## 布局

```
patches/vllm-0.6.3/
  series                                      # quilt/系列顺序
  0001-backport-qwen3-model-config.patch      # 新建 qwen3 模型+Config+注册
  0002-guided-decoding-params-none.patch      # guided 为空时返回 None
  files/                                      # 仅「新建文件」的完整副本，便于阅读
    vllm/model_executor/models/qwen3.py
    vllm/transformers_utils/configs/qwen3.py
```

基线：官方 / 发行版 **vLLM 0.6.3**（或 `0.6.3.post*`）源码树，在应用补丁前应能 `pip install --user --no-deps -e .`。  
补丁 **不包含** torch/xformers 编译改动；那些是环境依赖，见主 README。

## 如何应用

在 Jetson（或同架构开发机）上：

```bash
cd ~/src/vllm
# 确认大致在 v0.6.3 标签/提交上
git status

# 方式 A：git am（保留提交信息）
git am /path/to/Qwen3onJetsonXavier32GB/patches/vllm-0.6.3/0001-backport-qwen3-model-config.patch
git am /path/to/Qwen3onJetsonXavier32GB/patches/vllm-0.6.3/0002-guided-decoding-params-none.patch

# 方式 B：git apply
git apply /path/to/Qwen3onJetsonXavier32GB/patches/vllm-0.6.3/0001-backport-qwen3-model-config.patch
git apply /path/to/Qwen3onJetsonXavier32GB/patches/vllm-0.6.3/0002-guided-decoding-params-none.patch

# 方式 C：quilt
# quilt import ... && quilt push -a
```

若 `git am` 因换行/偏移失败，可改用 `git apply -3` 或手动把 `files/` 下两个新文件拷进对应路径，再手工合并 `registry.py` / `config.py` / `configs/__init__.py` / `sampling_params.py`（见补丁 hunk）。

应用后：

```bash
# 仅 Python 变更 → 重启 vLLM 服务即可，无需 build_ext
python3 -c "from vllm.transformers_utils.config import _CONFIG_REGISTRY; assert 'qwen3' in _CONFIG_REGISTRY"
```

细节与设计动机：[docs/qwen3-backport.md](../docs/qwen3-backport.md)。

## 覆盖的文件

| 路径 | 变更 |
|------|------|
| `vllm/transformers_utils/configs/qwen3.py` | 新建 |
| `vllm/transformers_utils/configs/__init__.py` | 导出 |
| `vllm/transformers_utils/config.py` | `_CONFIG_REGISTRY` |
| `vllm/model_executor/models/qwen3.py` | 新建 |
| `vllm/model_executor/models/registry.py` | 注册 `Qwen3ForCausalLM` |
| `vllm/sampling_params.py` | guided 空 → `None` |

## 许可

补丁基于 Apache-2.0 的 vLLM 源码结构与上游 Qwen3 思路改编；使用时请遵守 [vLLM](https://github.com/vllm-project/vllm) 与模型权重许可。本仓库补丁文件以本仓库 [MIT](../LICENSE) 再分发文档/脚本；**嵌入的代码行**仍受原项目许可证约束。
