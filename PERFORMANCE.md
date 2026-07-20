# MiniTrain 性能基线

> 采集时间：2026-07-20
> 环境：macOS Apple Silicon，Python 3.11

## 数据管道

使用 `scripts/benchmark_pipeline.py` 对 1000 条样本（由 `data/sample_input.json` 复制 100 倍）进行清洗、去重、过滤、拆分：

| 指标 | 值 |
|---|---|
| 输入记录数 | 1000 |
| 处理耗时 | 0.508 s |
| 吞吐率 | **1968.3 records/s** |
| MinHash 去重后 | 10 |
| 质量过滤后 | 10 |
| 异常值过滤后 | 9 |
| 训练/验证/测试拆分 | 7 / 1 / 1 |

说明：
- 去重率接近 99%，因为复制数据高度重复；真实场景下去重率会更低。
- 该基线仅覆盖数据管道，未包含模型下载、LoRA 训练与评估（后者严重依赖 GPU 与模型大小）。

## 脚本位置

- `scripts/benchmark_pipeline.py`：可重复运行的数据管道基准测试。
