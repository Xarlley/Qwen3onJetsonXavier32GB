# 本地 LLM 服务测速要点（Xavier）

## 核心指标

| 指标 | 含义 |
|------|------|
| TTFT | 发请求 → 首个输出 token（流式最敏感） |
| TPOT / ITL | 后续每 token 间隔 |
| E2E | 整次完成延迟 |
| 输出 tok/s | completion tokens / 时间 |
| Goodput | 满足延迟 SLO 的有效吞吐 |
| 并发曲线 | 1→N 时 p50/p99 延迟 |

工具参考：vLLM 树内 `benchmarks/benchmark_serving.py`、自写 httpx/curl、LLMPerf 类思路。  
公平对比：同一 prompt 集、同一 `max_tokens`、区分冷/热启动；Instruct 用 **chat** API。

## Xavier 建议门槛

- 单流暖机 ≥ 10 tok/s（实测常约 13–15）
- `/health` 持续 200
- 并发从低到高，出现 OOM/超时即停

磁盘紧时跳过大型 ShareGPT；用短 chat 即可验收。
