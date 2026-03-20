# 从GPU到NPU：Qwen3-Embedding模型昇腾适配与性能优化实战

## 项目背景：跨平台迁移的技术挑战

在AI大模型蓬勃发展的今天，文本嵌入（Text Embedding）作为自然语言处理的基础能力，广泛应用于语义检索、文本聚类、相似度计算等场景。HuggingFace开源的`text-embedding-inference`框架因其高效的推理性能和灵活的模型支持，成为业界广泛采用的Embedding服务解决方案。

然而，该框架原生仅支持GPU和HPU平台。随着国产AI芯片生态的快速发展，将主流推理框架迁移到华为昇腾NPU平台，成为推动AI基础设施国产化的重要一环。本项目正是在这一背景下启动——将`text-embedding-inference`框架完整适配到昇腾NPU，使Qwen3-Embedding模型能够在国产硬件上高效运行。

作为项目核心开发者，我负责了从底层算子适配到上层模型重构的全链路优化工作。本文将详细分享这一过程中的技术决策、实现细节与性能成果。

---

## 核心优化一：TND格式适配——打破BNSD的性能瓶颈

### 什么是TND格式？为什么要做格式转换？

在传统的Transformer推理中，数据通常以**BNSD格式**（Batch × Sequence_length × Num_heads × Head_dim）组织。这种格式需要将不同长度的序列通过padding对齐到相同长度，存在两个明显缺陷：

1. **内存浪费**：短序列需要大量padding，无效计算占比高
2. **带宽压力**：padding位置的数据搬运消耗宝贵的显存带宽

**TND格式**（Total_tokens × Num_heads × Head_dim）采用变长序列压缩策略，将批次内所有token展平为连续存储，通过`cu_seqlens`（累积序列长度）记录每个序列的边界。这种格式天然适合NPU的融合算子，可显著减少无效计算。

### BNSD vs TND 格式对比图

<div style="margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 12px;">

<div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">

<div style="flex: 1; min-width: 300px; max-width: 450px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 20px; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
<div style="color: white; font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 15px;">❌ BNSD 格式（传统）</div>
<div style="background: white; border-radius: 12px; padding: 15px; margin-bottom: 15px;">
<div style="color: #4a5568; font-size: 12px; margin-bottom: 8px;">张量形状</div>
<div style="color: #2d3748; font-size: 16px; font-weight: bold; font-family: 'Courier New', monospace;">[B, S, N, D] = [3, 8, 4, 64]</div>
</div>
<div style="display: flex; flex-direction: column; gap: 4px; padding: 10px; background: #f7fafc; border-radius: 8px;">
<div style="color: #4a5568; font-size: 11px; margin-bottom: 5px;">Batch 0 (序列长度=8)</div>
<div style="display: flex; gap: 2px;">
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
</div>
<div style="color: #4a5568; font-size: 11px; margin: 10px 0 5px 0;">Batch 1 (序列长度=3)</div>
<div style="display: flex; gap: 2px;">
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
</div>
<div style="color: #4a5568; font-size: 11px; margin: 10px 0 5px 0;">Batch 2 (序列长度=6)</div>
<div style="display: flex; gap: 2px;">
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;">P</span>
</div>
</div>
<div style="background: white; border-radius: 12px; padding: 15px; margin-top: 10px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="color: #718096; font-size: 13px;">有效计算</span><span style="font-weight: bold; font-size: 13px; color: #e53e3e;">14 / 24 = 58.3%</span></div>
<div style="display: flex; justify-content: space-between;"><span style="color: #718096; font-size: 13px;">Padding开销</span><span style="font-weight: bold; font-size: 13px; color: #e53e3e;">41.7% ❌</span></div>
</div>
</div>

<div style="flex: 1; min-width: 300px; max-width: 450px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 16px; padding: 20px; box-shadow: 0 10px 40px rgba(17, 153, 142, 0.3);">
<div style="color: white; font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 15px;">✅ TND 格式（优化后）</div>
<div style="background: white; border-radius: 12px; padding: 15px; margin-bottom: 15px;">
<div style="color: #4a5568; font-size: 12px; margin-bottom: 8px;">张量形状</div>
<div style="color: #2d3748; font-size: 16px; font-weight: bold; font-family: 'Courier New', monospace;">[T, N, D] = [14, 4, 64]</div>
</div>
<div style="display: flex; flex-direction: column; gap: 4px; padding: 10px; background: #f7fafc; border-radius: 8px;">
<div style="color: #4a5568; font-size: 11px; margin-bottom: 5px;">连续存储（无Padding）</div>
<div style="display: flex; gap: 2px; flex-wrap: wrap;">
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
<span style="width: 24px; height: 24px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; color: white; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">T</span>
</div>
</div>
<div style="background: white; border-radius: 12px; padding: 15px; margin-top: 10px;">
<div style="color: #4a5568; font-size: 12px; margin-bottom: 8px;">cu_seqlens（序列边界索引）</div>
<div style="display: flex; gap: 8px; flex-wrap: wrap;">
<span style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); color: white; padding: 6px 12px; border-radius: 6px; font-size: 14px; font-weight: bold;">0</span>
<span style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); color: white; padding: 6px 12px; border-radius: 6px; font-size: 14px; font-weight: bold;">5</span>
<span style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); color: white; padding: 6px 12px; border-radius: 6px; font-size: 14px; font-weight: bold;">8</span>
<span style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); color: white; padding: 6px 12px; border-radius: 6px; font-size: 14px; font-weight: bold;">14</span>
</div>
</div>
<div style="background: white; border-radius: 12px; padding: 15px; margin-top: 10px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="color: #718096; font-size: 13px;">有效计算</span><span style="font-weight: bold; font-size: 13px; color: #38a169;">14 / 14 = 100%</span></div>
<div style="display: flex; justify-content: space-between;"><span style="color: #718096; font-size: 13px;">Padding开销</span><span style="font-weight: bold; font-size: 13px; color: #38a169;">0% ✅</span></div>
</div>
</div>

</div>

<div style="display: flex; gap: 15px; justify-content: center; margin-top: 15px; flex-wrap: wrap; font-size: 12px; color: #4a5568;">
<span style="display: flex; align-items: center; gap: 6px;"><span style="width: 16px; height: 16px; border-radius: 4px; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);"></span> 有效Token (T)</span>
<span style="display: flex; align-items: center; gap: 6px;"><span style="width: 16px; height: 16px; border-radius: 4px; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); opacity: 0.5;"></span> Padding (P) - 无效计算</span>
</div>

</div>

### 代码实现要点

在[flash_qwen3.py](file:///g:/大学资料/3_研0/InternProject/text-embedding-inference-all-project/backends/python/server/text_embeddings_server/models/flash_qwen3.py)中，TND格式适配贯穿整个模型前向传播流程：

```python
# embed函数中的TND格式处理
def embed(self, batch):
    if isinstance(batch, FlashBatch):
        cu_seqlens = batch.cu_seqlens  # 累积序列长度
        # 模型输出为 [T, H] 格式（T=总token数，H=隐藏维度）
        output = self.model.forward(input_ids=batch.input_ids, ...)
        hidden_states = output.last_hidden_state  # shape: [T, H]
        
        # 直接通过cu_seqlens索引取最后token，无需重组
        last_token_indices = cu_seqlens[1:] - 1
        embedding = hidden_states[last_token_indices]
```

在[flash_attn.py](file:///g:/大学资料/3_研0/InternProject/text-embedding-inference-all-project/backends/python/server/text_embeddings_server/utils/flash_attn.py)中，注意力算子同样需要适配TND格式：

```python
def npu_attn(q, k, v, num_heads, out, seqlen_q, ...):
    # npu_fusion_attention 原生支持TND layout
    out_ = torch_npu.npu_fusion_attention(
        query=q, key=k, value=v,
        head_num=num_heads,
        input_layout="TND",  # 关键：指定TND格式
        actual_seq_qlen=seqlen_q.tolist(),
        actual_seq_kvlen=seqlen_k.tolist(),
    )[0]
```

### 性能收益

TND格式适配带来的收益体现在两个方面：

- **内存效率提升**：消除padding开销，实际有效计算占比从约60%提升至接近100%
- **算子调用优化**：NPU融合注意力算子对TND格式有原生优化，减少数据搬运次数

---

## 核心优化二：NPU融合算子替换——释放硬件极致性能

昇腾NPU提供了丰富的融合算子库（通过`torch_npu`模块访问），这些算子针对硬件特性深度优化，相比通用PyTorch算子可实现数倍性能提升。本项目重点替换了以下核心算子：

### NPU融合算子替换架构图

<div style="margin: 20px 0; padding: 30px; background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-radius: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">

<div style="color: white; font-size: 20px; font-weight: bold; text-align: center; margin-bottom: 25px;">🔧 NPU融合算子替换架构</div>

<div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center; margin-bottom: 15px;">
<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 10px; font-weight: bold; min-width: 120px; text-align: center; font-size: 14px;">注意力层</span>
<span style="background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; text-decoration: line-through; opacity: 0.7;">flash_attn_cuda</span>
<span style="color: #4fd1c5; font-size: 20px;">→</span>
<span style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; font-weight: bold; box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);">npu_fusion_attention</span>
<span style="background: rgba(255, 255, 255, 0.1); color: #4fd1c5; padding: 6px 12px; border-radius: 6px; font-size: 11px;">+57.6% QPS</span>
</div>

<div style="height: 2px; background: linear-gradient(90deg, transparent, #4fd1c5, transparent); margin: 10px 0;"></div>

<div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center; margin-bottom: 15px;">
<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 10px; font-weight: bold; min-width: 120px; text-align: center; font-size: 14px;">归一化层</span>
<span style="background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; text-decoration: line-through; opacity: 0.7;">手动RMS Norm</span>
<span style="color: #4fd1c5; font-size: 20px;">→</span>
<span style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; font-weight: bold; box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);">npu_rms_norm</span>
<span style="background: rgba(255, 255, 255, 0.1); color: #4fd1c5; padding: 6px 12px; border-radius: 6px; font-size: 11px;">+2.5% QPS</span>
</div>

<div style="height: 2px; background: linear-gradient(90deg, transparent, #4fd1c5, transparent); margin: 10px 0;"></div>

<div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center; margin-bottom: 15px;">
<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 10px; font-weight: bold; min-width: 120px; text-align: center; font-size: 14px;">投影层</span>
<span style="background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; text-decoration: line-through; opacity: 0.7;">F.linear × 4</span>
<span style="color: #4fd1c5; font-size: 20px;">→</span>
<span style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; font-weight: bold; box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);">npu_linear × 4</span>
<span style="background: rgba(255, 255, 255, 0.1); color: #4fd1c5; padding: 6px 12px; border-radius: 6px; font-size: 11px;">+3.5% QPS</span>
</div>

<div style="height: 2px; background: linear-gradient(90deg, transparent, #4fd1c5, transparent); margin: 10px 0;"></div>

<div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center; margin-bottom: 15px;">
<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 10px; font-weight: bold; min-width: 120px; text-align: center; font-size: 14px;">位置编码</span>
<span style="background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; text-decoration: line-through; opacity: 0.7;">rotate_half + matmul</span>
<span style="color: #4fd1c5; font-size: 20px;">→</span>
<span style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; font-weight: bold; box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);">npu_rotary_mul</span>
<span style="background: rgba(255, 255, 255, 0.1); color: #4fd1c5; padding: 6px 12px; border-radius: 6px; font-size: 11px;">+2.0% QPS</span>
</div>

<div style="height: 2px; background: linear-gradient(90deg, transparent, #4fd1c5, transparent); margin: 10px 0;"></div>

<div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center; margin-bottom: 15px;">
<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 10px; font-weight: bold; min-width: 120px; text-align: center; font-size: 14px;">MLP层</span>
<span style="background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; text-decoration: line-through; opacity: 0.7;">linear×3 + SiLU</span>
<span style="color: #4fd1c5; font-size: 20px;">→</span>
<span style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; padding: 12px 18px; border-radius: 10px; font-size: 13px; font-weight: bold; box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);">linear×2 + npu_swiglu</span>
<span style="background: rgba(255, 255, 255, 0.1); color: #4fd1c5; padding: 6px 12px; border-radius: 6px; font-size: 11px;">+4.3% QPS</span>
</div>

<div style="background: rgba(255, 255, 255, 0.05); border: 2px solid #4fd1c5; border-radius: 15px; padding: 20px; margin-top: 20px;">
<div style="color: #4fd1c5; font-size: 16px; font-weight: bold; margin-bottom: 15px; text-align: center;">📊 总体性能提升</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;">
<div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; text-align: center;"><div style="color: #48bb78; font-size: 24px; font-weight: bold;">+69.9%</div><div style="color: #a0aec0; font-size: 12px; margin-top: 5px;">QPS提升</div></div>
<div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; text-align: center;"><div style="color: #48bb78; font-size: 24px; font-weight: bold;">5个</div><div style="color: #a0aec0; font-size: 12px; margin-top: 5px;">融合算子替换</div></div>
<div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; text-align: center;"><div style="color: #48bb78; font-size: 24px; font-weight: bold;">-41%</div><div style="color: #a0aec0; font-size: 12px; margin-top: 5px;">总处理时间</div></div>
<div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; text-align: center;"><div style="color: #48bb78; font-size: 24px; font-weight: bold;">119.45</div><div style="color: #a0aec0; font-size: 12px; margin-top: 5px;">最高QPS</div></div>
</div>
</div>

</div>

### 2.1 注意力层：`npu_fusion_attention`

这是最核心的优化点。原始代码依赖CUDA的Flash Attention，在NPU上无法运行。我们通过`torch_npu.npu_fusion_attention`实现了等效功能：

```python
# flash_attn.py 核心实现
def npu_attn(q, k, v, num_heads, out, seqlen_q, seqlen_k, ...):
    if is_causal:
        # 构造causal mask
        attn_mask_npu = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=q.device), 
            diagonal=1
        )
        out_ = torch_npu.npu_fusion_attention(
            query=q, key=k, value=v,
            head_num=num_heads,
            input_layout="TND",
            scale=softmax_scale,
            actual_seq_qlen=seqlen_q.tolist(),
            actual_seq_kvlen=seqlen_k.tolist(),
            sparse_mode=3,  # causal模式
            atten_mask=attn_mask_npu
        )[0]
```

### 2.2 归一化层：`npu_rms_norm`

RMS Norm是Qwen3模型使用的归一化方式，原始实现需要多次数据类型转换：

```python
# 原始实现（已弃用）
# hidden_states = hidden_states.to(torch.float32)
# variance = hidden_states.pow(2).mean(-1, keepdim=True)
# hidden_states = hidden_states * torch.rsqrt(variance + eps)

# NPU融合实现
return torch_npu.npu_rms_norm(
    hidden_states.to(input_dtype), 
    self.weight, 
    epsilon=self.variance_epsilon
)[0]
```

### 2.3 线性层：`npu_linear`

将所有投影层的`F.linear`替换为`npu_linear`，减少算子调度开销：

```python
# flash_qwen3.py Qwen3Attention.forward
q = self.q_norm.forward(
    torch_npu.npu_linear(hidden_states, self.q_proj_weight, bias=None)
    .view(*input_shape, self.num_heads, self.head_dim)
)
k = self.k_norm.forward(
    torch_npu.npu_linear(hidden_states, self.k_proj_weight, bias=None)
    .view(*input_shape, self.num_key_value_heads, self.head_dim)
)
```

#### 为什么`npu_linear`比`nn.Linear`更快？

这是一个关键的技术问题。经过深入分析，我们发现两者最终都调用同一个底层CANN算子`aclnnAddmm`，但调用路径不同：

**调用链对比：**

```
nn.Linear 调用链：
Python forward() → F.linear() → aten::linear → aten::t → aten::addmm → aclnnAddmm
（多层 Python/C++ dispatch 开销）

npu_linear 调用链：
torch_npu.npu_linear() → npu::npu_linear → aclnnAddmm
（直接调用，dispatch 开销极小）
```

#### 性能对比测试

我们在不同shape下进行了详细测试：

| Config (Batch, IN, OUT) | nn.Linear (µs) | npu_linear (µs) | 性能差异 |
|------------------------|----------------|-----------------|---------|
| B=4, IN=128, OUT=64 | 37.34 | **5.24** | npu_linear快 **7.1×** |
| B=32, IN=1024, OUT=512 | 24.85 | **6.99** | npu_linear快 **3.6×** |
| B=128, IN=4096, OUT=2048 | 103.21 | 103.36 | 基本持平 |
| B=1024, IN=4096, OUT=2048 | 759.41 | 759.40 | 基本持平 |

#### 原因分析：NPU异步执行模型

<div style="margin: 20px 0; padding: 30px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">

<div style="color: white; font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 25px;">⏱️ NPU异步执行模型：小shape vs 大shape</div>

<div style="background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; margin-bottom: 20px;">
<div style="color: #4fd1c5; font-size: 14px; font-weight: bold; margin-bottom: 15px;">📊 小Shape场景 (B=4, IN=128, OUT=64)</div>
<div style="display: flex; margin-bottom: 8px; align-items: center;"><span style="width: 60px; color: #a0aec0; font-size: 12px;">nn.Linear</span><span style="flex: 1; height: 24px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; position: relative; overflow: hidden;"><span style="height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold; width: 85%; background: linear-gradient(90deg, #f6ad55 0%, #ed8936 100%);">dispatch ~32µs</span></span></div>
<div style="display: flex; margin-bottom: 8px; align-items: center;"><span style="width: 60px; color: #a0aec0; font-size: 12px;">NPU</span><span style="flex: 1; height: 24px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; position: relative; overflow: hidden;"><span style="height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold; width: 15%; background: linear-gradient(90deg, #4299e1 0%, #3182ce 100%);">kernel ~5µs</span><span style="height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #718096; font-weight: bold; width: 85%; position: absolute; left: 15%; background: rgba(255, 255, 255, 0.1);">等待CPU...</span></span></div>
<div style="display: flex; margin-bottom: 8px; align-items: center;"><span style="width: 60px; color: #a0aec0; font-size: 12px;">npu_linear</span><span style="flex: 1; height: 24px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; position: relative; overflow: hidden;"><span style="height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold; width: 20%; background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);">dispatch ~0.2µs</span></span></div>
<div style="color: #fc8181; font-size: 12px; margin-top: 10px; text-align: center;">⚠️ NPU在等CPU慢吞吞地dispatch，dispatch开销占主导 (86%)</div>
</div>

<div style="background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; margin-bottom: 20px;">
<div style="color: #4fd1c5; font-size: 14px; font-weight: bold; margin-bottom: 15px;">📊 大Shape场景 (B=1024, IN=4096, OUT=2048)</div>
<div style="display: flex; margin-bottom: 8px; align-items: center;"><span style="width: 60px; color: #a0aec0; font-size: 12px;">nn.Linear</span><span style="flex: 1; height: 24px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; position: relative; overflow: hidden;"><span style="height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold; width: 2%; background: linear-gradient(90deg, #f6ad55 0%, #ed8936 100%);"></span></span></div>
<div style="display: flex; margin-bottom: 8px; align-items: center;"><span style="width: 60px; color: #a0aec0; font-size: 12px;">NPU</span><span style="flex: 1; height: 24px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; position: relative; overflow: hidden;"><span style="height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold; width: 98%; background: linear-gradient(90deg, #4299e1 0%, #3182ce 100%);">kernel ~760µs</span></span></div>
<div style="display: flex; margin-bottom: 8px; align-items: center;"><span style="width: 60px; color: #a0aec0; font-size: 12px;">npu_linear</span><span style="flex: 1; height: 24px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; position: relative; overflow: hidden;"><span style="height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold; width: 1%; background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);"></span></span></div>
<div style="color: #68d391; font-size: 12px; margin-top: 10px; text-align: center;">✅ dispatch时间完全被kernel时间覆盖，差异被淹没 (&lt;0.1%)</div>
</div>

<div style="background: rgba(79, 209, 197, 0.1); border: 1px solid #4fd1c5; border-radius: 10px; padding: 15px; margin-top: 15px; text-align: center;">
<div style="color: #4fd1c5; font-family: 'Courier New', monospace; font-size: 14px;">T_measured = T_dispatch + T_kernel</div>
<div style="color: #a0aec0; font-size: 11px; margin-top: 8px;">小shape: T_dispatch占主导 → npu_linear优势明显<br>大shape: T_kernel占主导 → 两者差异淹没</div>
</div>

<div style="background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%); border-left: 4px solid #4fd1c5; border-radius: 0 12px 12px 0; padding: 15px 20px; margin-top: 20px;">
<div style="color: #e2e8f0; font-size: 13px; line-height: 1.6;">💡 <strong>核心洞察：</strong>NPU算子是异步下发的，CPU调用后立即返回，不等NPU算完。小矩阵kernel执行极快（~5µs），NPU算完的速度比CPU下发下一条命令还快，出现空闲等待。此时dispatch开销占主导，npu_linear的短路径优势明显。大矩阵时kernel执行时间长（~760µs），dispatch开销被完全"藏"在NPU执行时间里。</div>
</div>

</div>

#### 结论：何时使用`npu_linear`

| 场景 | 建议 | 原因 |
|------|------|------|
| **小batch推理** | ✅ 优先用`npu_linear` | dispatch开销占比大，可节省大量时间 |
| **大矩阵计算密集** | 两者无差异 | kernel时间远大于dispatch，用`nn.Linear`可读性更好 |
| **Embedding服务** | ✅ 推荐`npu_linear` | 典型小batch场景，性能提升显著 |

### 2.4 旋转位置编码：`npu_rotary_mul`

RoPE（Rotary Position Embedding）是Transformer中关键的位置编码方式。我们实现了完整的NPU版本：

```python
def apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=1):
    # 预处理：将3D [T, N, D] 转换为 4D [1, T, N, D]
    def _pre_process(x, cos, sin):
        origin_shape = x.shape
        if len(origin_shape) == 3:
            x = x.unsqueeze(0)  # TND → BTND
        return x, cos, sin, origin_shape, x.dtype
    
    q_, cos, sin, q_shape, q_dtype = _pre_process(q, cos, sin)
    k_, cos, sin, k_shape, k_dtype = _pre_process(k, cos, sin)
    
    # cos/sin reshape为 [1, T, 1, D]
    cos = cos.reshape(1, -1, 1, head_dim)
    sin = sin.reshape(1, -1, 1, head_dim)
    
    # 调用NPU融合算子
    output_q = torch_npu.npu_rotary_mul(q_, cos, sin)
    output_k = torch_npu.npu_rotary_mul(k_, cos, sin)
    
    # 后处理：还原为3D格式
    return output_q.squeeze(0), output_k.squeeze(0)
```

### 2.5 MLP层：`npu_swiglu`融合

这是性能提升最大的优化点。Qwen3使用SwiGLU激活函数，原始实现需要三次矩阵乘法：

### MLP层优化前后对比图

<div style="margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 12px;">

<div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">

<div style="flex: 1; min-width: 300px; max-width: 450px; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.15);">
<div style="padding: 15px 20px; text-align: center; font-weight: bold; font-size: 16px; background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); color: white;">❌ 原始实现（3次矩阵乘法）</div>
<div style="background: white; padding: 25px;">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: #e2e8f0; color: #4a5568;">L</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">F.linear (gate_proj)</strong><br><span style="color: #718096; font-size: 12px;">hidden → [T, intermediate_size]</span></span></div>
<div style="text-align: center; color: #cbd5e0; font-size: 20px; margin: -5px 0;">↓</div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: #e2e8f0; color: #4a5568;">L</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">F.linear (up_proj)</strong><br><span style="color: #718096; font-size: 12px;">hidden → [T, intermediate_size]</span></span></div>
<div style="text-align: center; color: #cbd5e0; font-size: 20px; margin: -5px 0;">↓</div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: #fef3c7; color: #d97706;">A</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">SiLU(gate)</strong><br><span style="color: #718096; font-size: 12px;">激活函数</span></span></div>
<div style="text-align: center; color: #cbd5e0; font-size: 20px; margin: -5px 0;">↓</div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: #dbeafe; color: #2563eb;">×</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">element-wise mul</strong><br><span style="color: #718096; font-size: 12px;">SiLU(gate) * up</span></span></div>
<div style="text-align: center; color: #cbd5e0; font-size: 20px; margin: -5px 0;">↓</div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: #e2e8f0; color: #4a5568;">L</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">F.linear (down_proj)</strong><br><span style="color: #718096; font-size: 12px;">→ [T, hidden_size]</span></span></div>
<div style="margin-top: 20px; padding-top: 15px; border-top: 2px dashed #e2e8f0;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #718096;">矩阵乘法次数</span><span style="font-weight: bold; color: #e53e3e;">3次</span></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #718096;">中间结果存储</span><span style="font-weight: bold; color: #e53e3e;">2个 [T, intermediate_size]</span></div>
<div style="display: flex; justify-content: space-between; font-size: 13px;"><span style="color: #718096;">算子调度开销</span><span style="font-weight: bold; color: #e53e3e;">高（5次调度）</span></div>
</div>
</div>
</div>

<div style="flex: 1; min-width: 300px; max-width: 450px; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.15);">
<div style="padding: 15px 20px; text-align: center; font-weight: bold; font-size: 16px; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white;">✅ 优化实现（2次矩阵乘法）</div>
<div style="background: white; padding: 25px;">
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white;">L</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">npu_linear (gate_up_proj)</strong><br><span style="color: #718096; font-size: 12px;">hidden → [T, 2×intermediate_size]</span></span></div>
<div style="text-align: center; color: #cbd5e0; font-size: 20px; margin: -5px 0;">↓</div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white;">F</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">npu_swiglu</strong><br><span style="color: #718096; font-size: 12px;">融合: split + SiLU + mul</span></span></div>
<div style="text-align: center; color: #cbd5e0; font-size: 20px; margin: -5px 0;">↓</div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"><span style="width: 36px; height: 36px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 16px; background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white;">L</span><span style="flex: 1;"><strong style="color: #2d3748; font-size: 14px;">npu_linear (down_proj)</strong><br><span style="color: #718096; font-size: 12px;">→ [T, hidden_size]</span></span></div>
<div style="margin-top: 20px; padding-top: 15px; border-top: 2px dashed #e2e8f0;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #718096;">矩阵乘法次数</span><span style="font-weight: bold; color: #38a169;">2次 (-33%)</span></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #718096;">中间结果存储</span><span style="font-weight: bold; color: #38a169;">1个（融合算子内部）</span></div>
<div style="display: flex; justify-content: space-between; font-size: 13px;"><span style="color: #718096;">算子调度开销</span><span style="font-weight: bold; color: #38a169;">低（3次调度）</span></div>
</div>
<div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 2px solid #86efac; border-radius: 12px; padding: 15px; margin-top: 15px; text-align: center;">
<div style="color: #166534; font-weight: bold; font-size: 14px;">🚀 性能提升 +4.3% QPS</div>
<div style="color: #4ade80; font-size: 12px; margin-top: 5px;">减少33%矩阵乘法，消除中间显存分配</div>
</div>
</div>
</div>

</div>

</div>

```python
# 原始实现（已弃用）
# gated = F.linear(hidden_state, self.gate_proj_weight)
# uped = F.linear(hidden_state, self.up_proj_weight)
# output = F.linear(silu(gated) * uped, self.down_proj_weight)

# 优化实现：权重合并 + 融合算子
self.gate_up_proj_weight = torch.cat(
    [self.gate_proj_weight, self.up_proj_weight], dim=0
)

def forward(self, hidden_state):
    # 一次linear得到gate和up的拼接结果
    gate_up_states = torch_npu.npu_linear(
        hidden_state, self.gate_up_proj_weight, bias=None
    )
    # npu_swiglu融合SiLU激活和门控乘法
    hidden_states = torch_npu.npu_swiglu(gate_up_states, dim=-1)
    return torch_npu.npu_linear(hidden_states, self.down_proj_weight, bias=None)
```

这一优化将MLP层从**3次矩阵乘法**减少到**2次**，同时消除了中间结果的显存分配。

---

## 核心优化三：Pooling层重构——去除第三方依赖

原始代码依赖`sentence_transformers`库的Pooling实现，在NPU环境存在兼容性问题。我们完全重写了Pooling层，实现纯PyTorch原生版本：

```python
# pooling.py 核心实现
class LastTokenPooling(_Pooling):
    """取每个序列最后一个真实token（适合decoder-only模型）"""
    def forward(self, model_output, attention_mask) -> Tensor:
        token_embeddings = model_output[0]
        attention_mask = attention_mask.to(dtype=torch.bool)
        # 计算每个序列的实际长度
        last_indices = attention_mask.sum(dim=1, keepdim=True) - 1
        last_indices = last_indices.clamp(min=0)
        # 使用gather提取最后token
        last_indices_expanded = last_indices.unsqueeze(-1).expand(
            -1, -1, token_embeddings.shape[-1]
        )
        return token_embeddings.gather(1, last_indices_expanded).squeeze(1)

class DefaultPooling(_Pooling):
    """工厂路由类，根据pooling_mode派发到对应实现"""
    def __init__(self, hidden_size, pooling_mode="last"):
        if pooling_mode == "mean":
            self.pooling = MeanPooling(hidden_size, pooling_mode)
        elif pooling_mode == "max":
            self.pooling = MaxPooling(hidden_size, pooling_mode)
        elif pooling_mode == "cls":
            self.pooling = CLSPooling(hidden_size, pooling_mode)
        elif pooling_mode in ("last", "last_token"):
            self.pooling = LastTokenPooling(hidden_size, pooling_mode)
```

新增的`LastTokenPooling`类专为Qwen3等decoder-only模型设计，通过`attention_mask`计算真实序列长度，精确提取最后一个有效token的embedding。

---

## 性能对比：优化效果一目了然

经过三轮迭代优化，最终实现了显著的性能提升。以下是完整的性能测试数据：

### 性能对比可视化图表

<div style="margin: 20px 0; padding: 25px; background: white; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);">

<div style="font-size: 18px; font-weight: bold; color: #2d3748; margin-bottom: 20px; text-align: center;">📊 Qwen3-Embedding 性能对比分析</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-bottom: 30px;">
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 16px; padding: 20px; color: white; text-align: center;"><div style="font-size: 32px; font-weight: bold; margin-bottom: 5px;">119.45</div><div style="font-size: 13px; opacity: 0.9;">最高QPS 🏆</div><div style="font-size: 11px; opacity: 0.7; margin-top: 5px;">全NPU算子融合 + 4线程</div></div>
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 20px; color: white; text-align: center;"><div style="font-size: 32px; font-weight: bold; margin-bottom: 5px;">+69.9%</div><div style="font-size: 13px; opacity: 0.9;">性能提升</div><div style="font-size: 11px; opacity: 0.7; margin-top: 5px;">相比原生Flash Qwen3</div></div>
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 20px; color: white; text-align: center;"><div style="font-size: 32px; font-weight: bold; margin-bottom: 5px;">8.57s</div><div style="font-size: 13px; opacity: 0.9;">最低总时间</div><div style="font-size: 11px; opacity: 0.7; margin-top: 5px;">1024请求处理完成</div></div>
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 20px; color: white; text-align: center;"><div style="font-size: 32px; font-weight: bold; margin-bottom: 5px;">0.450s</div><div style="font-size: 13px; opacity: 0.9;">最低响应时间</div><div style="font-size: 11px; opacity: 0.7; margin-top: 5px;">单请求平均延迟</div></div>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px;">

<div>
<div style="font-size: 14px; font-weight: bold; color: #2d3748; margin-bottom: 15px; text-align: center;">QPS对比（每秒查询数）</div>
<table style="width: 100%; border-collapse: collapse; font-size: 12px;">
<tr style="background: #f7fafc;"><th style="padding: 10px; text-align: left; border-bottom: 2px solid #e2e8f0;">线程数</th><th style="padding: 10px; text-align: center; border-bottom: 2px solid #e2e8f0; color: #fc8181;">原生</th><th style="padding: 10px; text-align: center; border-bottom: 2px solid #e2e8f0; color: #f6ad55;">+注意力</th><th style="padding: 10px; text-align: center; border-bottom: 2px solid #e2e8f0; color: #68d391;">+RMSNorm</th><th style="padding: 10px; text-align: center; border-bottom: 2px solid #e2e8f0; color: #4299e1;">全融合</th></tr>
<tr><td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">1线程</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">30.54</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">43.40</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">43.75</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">42.53</td></tr>
<tr style="background: #f0fff4;"><td style="padding: 10px; border-bottom: 1px solid #e2e8f0;"><strong>4线程</strong></td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">70.33</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">110.87</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">113.68</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;"><strong style="color: #38a169;">119.45 🏆</strong></td></tr>
<tr><td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">8线程</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">68.99</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">105.49</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">111.53</td><td style="padding: 10px; text-align: center; border-bottom: 1px solid #e2e8f0;">117.89</td></tr>
<tr><td style="padding: 10px;">20线程</td><td style="padding: 10px; text-align: center;">68.89</td><td style="padding: 10px; text-align: center;">105.05</td><td style="padding: 10px; text-align: center;">111.31</td><td style="padding: 10px; text-align: center;">117.82</td></tr>
</table>
</div>

<div>
<div style="font-size: 14px; font-weight: bold; color: #2d3748; margin-bottom: 15px; text-align: center;">优化阶段累计性能提升</div>
<div style="display: flex; flex-direction: column; gap: 10px;">
<div style="display: flex; align-items: center; gap: 10px;"><span style="width: 120px; font-size: 11px; color: #4a5568;">+NPU融合注意力</span><span style="flex: 1; height: 24px; background: rgba(246, 173, 85, 0.3); border-radius: 4px; position: relative;"><span style="position: absolute; left: 0; top: 0; height: 100%; width: 72%; background: linear-gradient(90deg, #f6ad55 0%, #ed8936 100%); border-radius: 4px;"></span><span style="position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: bold; color: #2d3748;">+57.6%</span></span></div>
<div style="display: flex; align-items: center; gap: 10px;"><span style="width: 120px; font-size: 11px; color: #4a5568;">+RMSNorm</span><span style="flex: 1; height: 24px; background: rgba(104, 211, 145, 0.3); border-radius: 4px; position: relative;"><span style="position: absolute; left: 0; top: 0; height: 100%; width: 77%; background: linear-gradient(90deg, #68d391 0%, #48bb78 100%); border-radius: 4px;"></span><span style="position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: bold; color: #2d3748;">+61.6%</span></span></div>
<div style="display: flex; align-items: center; gap: 10px;"><span style="width: 120px; font-size: 11px; color: #4a5568;">+npu_linear</span><span style="flex: 1; height: 24px; background: rgba(99, 179, 237, 0.3); border-radius: 4px; position: relative;"><span style="position: absolute; left: 0; top: 0; height: 100%; width: 81%; background: linear-gradient(90deg, #63b3ed 0%, #4299e1 100%); border-radius: 4px;"></span><span style="position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: bold; color: #2d3748;">+65.1%</span></span></div>
<div style="display: flex; align-items: center; gap: 10px;"><span style="width: 120px; font-size: 11px; color: #4a5568;">+npu_rotary_mul</span><span style="flex: 1; height: 24px; background: rgba(159, 122, 234, 0.3); border-radius: 4px; position: relative;"><span style="position: absolute; left: 0; top: 0; height: 100%; width: 84%; background: linear-gradient(90deg, #9f7aea 0%, #805ad5 100%); border-radius: 4px;"></span><span style="position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: bold; color: #2d3748;">+67.1%</span></span></div>
<div style="display: flex; align-items: center; gap: 10px;"><span style="width: 120px; font-size: 11px; color: #4a5568;">+npu_swiglu</span><span style="flex: 1; height: 24px; background: rgba(72, 187, 120, 0.3); border-radius: 4px; position: relative;"><span style="position: absolute; left: 0; top: 0; height: 100%; width: 87%; background: linear-gradient(90deg, #48bb78 0%, #38a169 100%); border-radius: 4px;"></span><span style="position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: bold; color: #2d3748;">+69.9%</span></span></div>
</div>
</div>

</div>

</div>

### 关键性能指标

| 指标 | 最佳配置 | 数值 | 相比原生提升 |
|------|---------|------|-------------|
| 最高QPS | 全NPU算子融合 + 4线程 | **119.45** | 69.9% |
| 最低总时间 | 全NPU算子融合 + 4线程 | **8.57s** | 41.1% |
| 最低响应时间 | NPU融合注意力+RMSNorm + 1线程 | **0.450s** | 30.2% |

### 不同配置下的QPS对比

| 线程数 | 原生Flash Qwen3 | 仅NPU融合注意力 | NPU融合注意力+RMSNorm | 全NPU算子融合 |
|--------|----------------|----------------|---------------------|--------------|
| 1线程 | 30.54 | 43.40 | 43.75 | 42.53 |
| 4线程 | 70.33 | 110.87 | 113.68 | **119.45** |
| 8线程 | 68.99 | 105.49 | 111.53 | 117.89 |
| 20线程 | 68.89 | 105.05 | 111.31 | 117.82 |

### 各优化模块贡献分析

| 优化模块 | QPS提升 | 主要收益来源 |
|---------|--------|-------------|
| npu_fusion_attention | +57.6% | 消除注意力计算的显存瓶颈，支持TND格式 |
| npu_rms_norm | +2.5% | 减少数据类型转换，融合方差计算 |
| npu_linear | +3.5% | 减少算子调度开销，优化内存访问 |
| npu_rotary_mul | +2.0% | 融合旋转编码计算 |
| npu_swiglu | +4.3% | MLP层从3次矩阵乘减少到2次 |

---

## 技术难点与解决方案

### 难点一：GQA（Grouped Query Attention）兼容

Qwen3使用GQA架构，`num_key_value_heads`（如8）小于`num_attention_heads`（如32）。原始代码在TND格式下会导致shape不匹配。

### GQA架构示意图

<div style="margin: 20px 0; padding: 30px; background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">

<div style="color: white; font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 25px;">🔄 GQA (Grouped Query Attention) 架构</div>

<div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
<div style="text-align: center;">
<div style="color: #93c5fd; font-size: 14px; margin-bottom: 15px; font-weight: bold;">Query Heads (32个)</div>
<div style="display: grid; gap: 8px;">
<div style="display: flex; gap: 8px; justify-content: center;"><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q0</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q1</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q2</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q3</span></div>
<div style="display: flex; gap: 8px; justify-content: center;"><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q4</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q5</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q6</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Q7</span></div>
<div style="color: #a0aec0; font-size: 11px; text-align: center;">... 24个更多Query头 ...</div>
</div>
</div>

<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; color: #4fd1c5;">
<div style="font-size: 40px;">⟷</div>
<div style="font-size: 12px; margin-top: 10px; color: #a0aec0; text-align: center;">每4个Q头<br>共享1个KV头</div>
</div>

<div style="text-align: center;">
<div style="color: #93c5fd; font-size: 14px; margin-bottom: 15px; font-weight: bold;">Key/Value Heads (8个)</div>
<div style="display: grid; gap: 8px;">
<div style="display: flex; gap: 8px; justify-content: center;"><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV0</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV1</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV2</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV3</span></div>
<div style="display: flex; gap: 8px; justify-content: center;"><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV4</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV5</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV6</span><span style="width: 40px; height: 40px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);">KV7</span></div>
</div>
</div>
</div>

<div style="background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; margin-top: 25px;">
<div style="color: #4fd1c5; font-size: 14px; font-weight: bold; margin-bottom: 15px;">✅ 解决方案：repeat_interleave扩展KV头</div>
<div style="background: rgba(0, 0, 0, 0.3); border-radius: 8px; padding: 15px; font-family: 'Courier New', monospace; font-size: 13px; color: #a0e7e0; overflow-x: auto;">
if self.num_key_value_groups > 1:  # num_heads // num_kv_heads = 32 // 8 = 4<br>
&nbsp;&nbsp;&nbsp;&nbsp;k = k.repeat_interleave(self.num_key_value_groups, dim=1)  # [T, 8, D] → [T, 32, D]<br>
&nbsp;&nbsp;&nbsp;&nbsp;v = v.repeat_interleave(self.num_key_value_groups, dim=1)  # [T, 8, D] → [T, 32, D]
</div>
</div>

</div>

**解决方案**：在Attention forward中显式扩展k/v头数：

```python
if self.num_key_value_groups > 1:
    k = k.repeat_interleave(self.num_key_value_groups, dim=1)
    v = v.repeat_interleave(self.num_key_value_groups, dim=1)
```

### 难点二：cu_seqlens设备放置

`npu_fusion_attention`要求`actual_seq_qlen`参数为Python list。如果`cu_seqlens`在NPU上，`.tolist()`会触发D2H数据传输。

**解决方案**：在[types.py](file:///g:/大学资料/3_研0/InternProject/text-embedding-inference-all-project/backends/python/server/text_embeddings_server/models/types.py)中将`cu_seqlens`强制放在CPU：

```python
cu_seqlens = torch.tensor(pb.cu_seq_lengths, dtype=torch.int32, device="cpu")
```

### 难点三：RoPE的TND格式适配

`npu_rotary_mul`要求输入为4D张量`[B, T, N, D]`，而TND格式的q/k是3D张量`[T, N, D]`。

**解决方案**：通过`_pre_process`和`_post_process`函数实现3D↔4D转换：

```python
def _pre_process(x, cos, sin):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)  # [T, N, D] → [1, T, N, D]
    return x, ...

def _post_process(output, origin_shape, ...):
    if len(origin_shape) == 3:
        output = output.squeeze(0)  # [1, T, N, D] → [T, N, D]
    return output
```

---

## 项目价值与个人收获

### 项目价值

1. **推动国产AI生态**：成功将主流Embedding推理框架迁移到昇腾NPU，为国产AI基础设施添砖加瓦
2. **性能显著提升**：相比原生实现，QPS提升近70%，为实际业务部署提供了强有力的性能保障
3. **技术方案可复用**：TND格式适配、NPU融合算子替换等方案可推广到其他Transformer模型

### 个人收获

通过本项目，我深入理解了：

- **底层算子优化**的重要性：一个融合算子可能带来10%以上的性能提升
- **数据格式设计**对性能的影响：TND格式从架构层面解决了padding开销问题
- **跨平台适配**的挑战：不同硬件平台的算子接口、性能特性差异巨大，需要针对性优化

### 未来展望

1. **算子库持续完善**：基于[op-plugin](https://gitcode.com/Ascend/op-plugin)开源项目，探索更多NPU专用算子
2. **多模型支持**：将适配方案扩展到更多Embedding模型（如BGE、M3E等）
3. **分布式推理**：探索NPU集群上的分布式Embedding服务方案

---

## 结语

从GPU到NPU的迁移不仅是硬件平台的切换，更是对模型推理全链路的深度优化。本项目通过TND格式适配、NPU融合算子替换、Pooling层重构三大核心优化，实现了近70%的性能提升，证明了国产AI芯片在主流推理场景下的竞争力。

技术迭代永无止境，期待与更多开发者一起，共同推动国产AI生态的繁荣发展！

---

**参考资料**：
- [op-plugin开源项目](https://gitcode.com/Ascend/op-plugin)
- [昇腾CANN开发文档](https://www.hiascend.com/document)
- [text-embedding-inference原项目](https://github.com/huggingface/text-embeddings-inference)

**项目源码**：
- [flash_qwen3.py](file:///g:/大学资料/3_研0/InternProject/text-embedding-inference-all-project/backends/python/server/text_embeddings_server/models/flash_qwen3.py)
- [flash_attn.py](file:///g:/大学资料/3_研0/InternProject/text-embedding-inference-all-project/backends/python/server/text_embeddings_server/utils/flash_attn.py)
- [pooling.py](file:///g:/大学资料/3_研0/InternProject/text-embedding-inference-all-project/backends/python/server/text_embeddings_server/models/pooling.py)
