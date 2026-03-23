# 代码修改总结（自 `c870b01 蓝区电脑同步完整项目` 起）

## 1. 统计范围

- 基线提交：`c870b01` `蓝区电脑同步完整项目`
- 当前提交：`488be02` `Refactor code structure and remove redundant changes`
- 本文基于如下区间整理：

```bash
git diff c870b01..488be02
```

当前工作区相对 `HEAD` 为干净状态，因此本文总结的是已进入当前 git 历史的代码修改，而不是未提交改动。

## 2. 总体修改目标

这一阶段的修改，核心目标是将 `Qwen3-Embedding` 的 NPU 路径从“能跑通”推进到“结果正确、链路可分析、性能可对比”的状态，具体包括：

- 将 Qwen3 的关键算子替换为 NPU 原生实现
- 将注意力路径从 `npu_fusion_attention` 迁移到 `npu_fused_infer_attention_score(FIA)`
- 去掉调试残留和冗余逻辑，收敛代码结构
- 增加基准测试、精度对比和算子级分析材料
- 保留 origin/old 版本代码，便于回滚和性能对比

## 3. 核心运行时代码修改

### 3.1 `backends/python/server/text_embeddings_server/models/flash_qwen3.py`

#### 修改内容

- 删除了大量调试日志、张量统计打印和实验性辅助函数
- 精简 `load_weight`，统一支持单文件和分片权重加载
- 将 RoPE 计算收敛为 `apply_rotary_pos_emb_npu`，直接调用 `torch_npu.npu_rotary_mul`
- 将 `Q/K/V/O` 投影统一改为 `torch_npu.npu_linear`
- 将 `RMSNorm` 改为 `torch_npu.npu_rms_norm`
- 注意力输出改为直接接收 `attention(...)` 返回值，不再预分配 `attn_output`
- MLP 从 `gate_proj + up_proj + activation + multiply + down_proj` 的普通 PyTorch 路径，改为：
  - `torch_npu.npu_linear`
  - `torch_npu.npu_swiglu`
  - `torch_npu.npu_linear`
- 在 decoder 中引入 `torch_npu.npu_add_rms_norm`，将“残差相加 + RMSNorm”合并
- `embed()` 改为先在设备侧提取最后 token 对应 embedding，再搬运到 CPU
- 增加非有限值检查，避免输出 `NaN/Inf` 后静默返回 `null`

#### 修改原因

- 旧实现中包含大量调试打印、CPU/PyTorch 通用路径和多段中间张量操作，不利于 NPU 推理性能
- Qwen3 为 GQA 结构，注意力、RoPE、MLP 和 RMSNorm 都是主要热点，适合切到 NPU 原生算子
- 旧版本在 embedding 输出阶段会把整段 hidden states 拉回 CPU，再取最后 token，存在不必要的数据搬运

#### 修改目的

- 提升 Qwen3 embedding 模型在 NPU 上的端到端推理效率
- 降低中间张量复制、CPU 回传和 Python 侧调度开销
- 为注意力路径切换到 FIA 提供一致的上层调用结构

### 3.2 `backends/python/server/text_embeddings_server/utils/flash_attn.py`

#### 修改内容

- 重写了 NPU 注意力实现，替换原先基于 `torch_npu.npu_fusion_attention` 的路径
- 新实现改为调用 `torch_npu.npu_fused_infer_attention_score`
- 增加 `_actual_seq_lengths_for_tnd_fia(...)`，统一处理 `TND` layout 下的累计长度参数
- 增加 `_normalize_attention_output(...)` 和 `_ensure_attention_out(...)`，统一多后端返回值处理
- 增加 `_get_npu_causal_mask(...)` 和 `_NPU_CAUSAL_MASK_CACHE`，缓存 `2048x2048` causal mask
- 修正 `sparse_mode=3` 时的 mask 传参方式，避免错误地按请求构造 `(S, S)` 小矩阵
- NPU 路径改为直接返回 FIA 输出，不再做 `out.copy_`
- 保留 CUDA / FlashAttentionV2 / HPU / IPEX 的兼容分支

#### 修改原因

- `npu_fusion_attention` 路径对 GQA、TND varlen 和推理场景的适配不如 FIA 直接
- 旧实现里存在 `out.copy_`、重复创建 mask、不同 layout fallback 混杂等问题
- FIA 对 `TND + causal` 的 `atten_mask` 形状和 `actual_seq_lengths` 有更严格要求，需要按文档修正

#### 修改目的

- 用更适合推理场景的 FIA 替代通用融合注意力实现
- 降低额外 tensor move、mask 构造和接口适配成本
- 让 Qwen3 的 GQA 注意力直接走 `num_heads + num_key_value_heads` 原生表达

### 3.3 `backends/python/server/text_embeddings_server/models/flash_bert.py`

#### 修改内容

- 将 `attention(...)` 调用方式同步为“直接接返回值”的新接口形式
- 去掉了本地预分配的 `torch.empty_like(q)`

#### 修改原因

- `flash_attn.py` 的接口形态发生变化，调用侧需要同步

#### 修改目的

- 保持共享注意力 helper 的调用一致性
- 避免重复保留旧接口写法

### 3.4 `backends/python/server/text_embeddings_server/models/types.py`

#### 修改内容

- 删除了 `FlashBatch.from_pb()` 中残留的调试日志输出

#### 修改原因

- 该日志会在服务启动和请求过程中持续打印，干扰观察和压测

#### 修改目的

- 清理运行时噪声，避免日志对性能测试和问题定位产生干扰

## 4. 对照代码与回滚材料

### 4.1 备份/对照文件

新增或保留了以下对照文件：

- `backends/python/server/text_embeddings_server/models/old_flash_qwen3.py`
- `backends/python/server/text_embeddings_server/utils/old_flash_attn.py`
- `backends/python/server/text_embeddings_server/origin/origin_flash_qwen3.py`
- `backends/python/server/text_embeddings_server/origin/origin_flash_attn.py`
- `backends/python/server/text_embeddings_server/origin/origin_pooling.py`

#### 原因与目的

- 保留“原版实现”“中间实验版”“当前版”三类对照，方便：
  - 回滚
  - 精度对比
  - 性能 profiling 对比
  - 撰写适配复盘和性能分析文档

## 5. 测试与分析脚本新增

### 5.1 `performance-test/benchmark_2.py`

#### 修改内容

- 新增多线程 embedding/rerank 压测脚本
- 增加成功请求、失败请求、成功批次、有效 QPS、尝试 QPS 等统计
- 输出更完整的时延分布指标，并支持写入 Excel

#### 原因

- 原有压测结果难以区分“真实性能提升”和“失败请求导致的 QPS 虚高”

#### 目的

- 为并发压测提供更可靠的吞吐统计口径
- 支撑对 FIA 与旧注意力路径的公平比较

### 5.2 `performance-test/compare_emb_accuracy.py`

#### 修改内容

- 新增双服务 embedding 输出对比工具
- 支持批量请求、余弦相似度计算和差异检查

#### 原因

- 注意力、RMSNorm、MLP 等改为 NPU 原生实现后，需要验证输出一致性

#### 目的

- 为精度回归检查提供独立工具

## 6. 文档与性能分析材料新增

### 6.1 NPU 算子文档

新增：

- `docsForNPUFunction/AddRMSNorm.md`
- `docsForNPUFunction/torch_npu-npu_fused_infer_attention_score.md`

#### 目的

- 为 `npu_add_rms_norm` 与 FIA 的参数约束、返回值语义和错误定位提供依据

### 6.2 性能与分析输出

新增：

- `ops-run-detail/operator_detail_group_by_Operator Type_qwen3_embedding_latest_1773990027.csv`
- `ops-run-detail/operator_detail_group_by_Operator Type_qwen3_embedding_origin_1773990038.csv`
- `performance_test/performance-test_3_19_11_09.html`

#### 目的

- 从算子级和压测级两个维度比较优化前后的行为与性能

### 6.3 复盘与对外说明文档

`docsForWeiXin/` 下新增或更新了多篇总结、对比和推文草稿，用于沉淀：

- 适配过程复盘
- FIA 与旧注意力路径的实现差异
- `nn.linear` 与 `npu_linear` 对比
- 算子性能拖尾与 profiling 分析

#### 目的

- 将实验过程、性能结论和实现细节沉淀为可复用材料

## 7. 这轮修改的核心收益

从代码层面看，这一阶段最大的变化是：

- Qwen3 embedding 主链路从通用 PyTorch 实现切换到更强的 NPU 原生算子链
- NPU 注意力路径正式切换到 FIA，并补齐了 `TND + causal + GQA` 的接口约束
- embedding 输出阶段减少了不必要的数据搬运
- 增加了精度对比、并发压测和算子级分析工具，便于继续做性能迭代

## 8. 当前仍需注意的问题

- `flash_attn.py` 仍然保留多后端兼容分支，接口层尚未完全收敛为单一风格
- `npu_add_rms_norm`、FIA 等算子存在严格的文档约束，部署到服务器时必须保证实际运行代码与本地版本一致
- 高并发压测中，若服务端仍有失败请求，必须优先看成功率与有效 QPS，而不是只看表面吞吐

## 9. 建议的阅读顺序

如果要快速理解这段改动，建议按以下顺序阅读：

1. `backends/python/server/text_embeddings_server/models/flash_qwen3.py`
2. `backends/python/server/text_embeddings_server/utils/flash_attn.py`
3. `performance-test/benchmark_2.py`
4. `performance-test/compare_emb_accuracy.py`
5. `docsForNPUFunction/torch_npu-npu_fused_infer_attention_score.md`
6. `ops-run-detail/*.csv` 与 `performance_test/performance-test_3_19_11_09.html`

