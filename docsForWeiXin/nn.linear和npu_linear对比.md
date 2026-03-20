User: 此脚本不import torch_npu也能正常调用npu，试找出原因

GitHub Copilot: 让我先查看相关文件，了解项目结构和依赖关系。 

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/init_address_lookup)

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library)

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/init_address_lookup/init_address_lookup.cpp#1-1), lines 1 to 80

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/init_address_lookup/CMakeLists.txt)

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/common/pytorch_npu_helper.hpp#1-1), lines 1 to 60

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/init_address_lookup/init_address_lookup.cpp#80-80), lines 80 to 100

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/init_address_lookup/build_ops.sh)

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/common)

Read [](file:///test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/common/CommonTorchOpConfig.cmake)

现在我已经找到了关键信息。让我再确认一下 `.so` 文件的链接依赖： 

Ran terminal command:  ldd /test/j50056782/RecSDK/cust_op/framework/torch_plugin/torch_library/init_address_lookup/build/libinit_address_lookup.so 2>/dev/null | grep -i "torch_npu\|torch\|npu" | head -20

Ran terminal command:  python3 -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"

Ran terminal command:  ls /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu/lib/ | grep -i "torch_npu\|libtorch_npu"

Ran terminal command:  python3 -c "
import torch_npu
import torch
# 查看 torch_npu 注册了哪些 PrivateUse1 相关内容
print('torch_npu version:', torch_npu.__version__)
print('torch version:', torch.__version__)
# 检查 npu 设备是否已注册
print('has npu:', hasattr(torch, 'npu'))
"

Ran terminal command:  python3 -c "
import torch
# 不 import torch_npu，看 npu 是否存在
print('has npu without torch_npu:', hasattr(torch, 'npu'))
"

Ran terminal command:  python3 -c "
import sys
# 检查 torch 导入时是否自动加载了 torch_npu
import torch
print('torch_npu in sys.modules:', 'torch_npu' in sys.modules)
# 检查 torch 的 __init__.py 是否有 npu 相关内容
import torch._C as _C
print(dir(_C))
" 2>&1 | head -30

关键发现：`import torch` 时，`torch_npu` 已经在 `sys.modules` 中了。让我进一步确认原因： 

Ran terminal command:  python3 -c "
import sys
import torch
# 找出是谁把 torch_npu 加入了 sys.modules
print('torch_npu in sys.modules:', 'torch_npu' in sys.modules)
print('torch_npu module file:', sys.modules.get('torch_npu', None))
" 2>&1 | head -10

Ran terminal command:  cat /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu/__init__.py | head -60

Ran terminal command:  ls /usr/local/python3.11.0/lib/python3.11/site-packages/ | grep -i "torch_npu\|pth\|usercustomize\|sitecustomize"

Ran terminal command:  ls /usr/local/python3.11.0/lib/python3.11/site-packages/torch/
# 查找 torch 的 autoload 机制
grep -r "TORCH_DEVICE_BACKEND_AUTOLOAD\|autoload\|torch_npu" /usr/local/python3.11.0/lib/python3.11/site-packages/torch/__init__.py 2>/dev/null | head -20

Ran terminal command:  grep -n "TORCH_DEVICE_BACKEND_AUTOLOAD\|autoload\|_load_backend\|device_backend" /usr/local/python3.11.0/lib/python3.11/site-packages/torch/__init__.py | head -30

Ran terminal command:  sed -n '2747,2810p' /usr/local/python3.11.0/lib/python3.11/site-packages/torch/__init__.py

Ran terminal command:  cat /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu-2.7.1.post3.dev20260205.dist-info/entry_points.txt 2>/dev/null || \
python3 -c "from importlib.metadata import entry_points; eps = entry_points(group='torch.backends'); [print(ep.name, ep.value) for ep in eps]"

Ran terminal command:  grep -n "_autoload" /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu/__init__.py | head -10

Ran terminal command:  sed -n '329,345p' /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu/__init__.py

Ran terminal command:  sed -n '329,360p' /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu/__init__.py

Ran terminal command:  wc -l /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu/__init__.py && tail -50 /usr/local/python3.11.0/lib/python3.11/site-packages/torch_npu/__init__.py

现在已经找到了完整的证据链。让我总结一下：

---

## 原因分析

**不 `import torch_npu` 也能正常调用 NPU，根本原因是：PyTorch 的"设备后端自动加载"机制（Device Backend Autoload）在 `import torch` 时自动加载了 `torch_npu`。**

整个调用链如下：

### 1. PyTorch 的 Entry Points 自动加载机制

`torch/__init__.py` 末尾有如下逻辑：

```python
# torch/__init__.py 第 2807 行
if _is_device_backend_autoload_enabled():
    _import_device_backends()
```

`_import_device_backends()` 会通过 Python 的 `importlib.metadata.entry_points` 机制，扫描所有注册在 `torch.backends` 组下的插件并自动调用其入口函数。

### 2. torch_npu 注册了 Entry Point

`torch_npu` 安装时在其 `dist-info/entry_points.txt` 中注册了：

```ini
[torch.backends]
torch_npu = torch_npu:_autoload
```

这意味着：**只要 `torch_npu` 已安装，`import torch` 就会自动触发 `torch_npu._autoload()`，从而完整加载整个 `torch_npu` 包**（包括 NPU 设备注册、算子注册等）。

### 3. 控制开关

可以通过环境变量禁用此行为：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

设置后，`import torch` 将不再自动加载 `torch_npu`，此时若不手动 `import torch_npu`，NPU 相关功能将不可用。

### 总结

| 步骤                                   | 发生了什么                                                  |
| -------------------------------------- | ----------------------------------------------------------- |
| `import torch`                         | 触发 `torch/__init__.py` 末尾的 `_import_device_backends()` |
| `_import_device_backends()`            | 读取所有 `torch.backends` entry points                      |
| 发现 `torch_npu = torch_npu:_autoload` | 自动 `import torch_npu` 并调用 `_autoload()`                |
| `torch_npu` 完整初始化                 | NPU 设备、算子（含 `PrivateUse1` dispatch key）全部注册完毕 |
| 脚本中 `import torch_npu`              | 实际上是**冗余的**，`sys.modules` 中已存在                  |

因此，测试脚本中第 24 行的 `import torch_npu` 是**完全多余的**——只要环境中安装了 `torch_npu`，`import torch` 就已经隐式完成了这件事。

User: 例如我现在使用了nn.linear函数，我怎么检查它是否调用了torch_npu.npu_linear来完成计算

GitHub Copilot: 好问题。有几种方法可以检查 `nn.Linear` 在 NPU 上是否实际调用了 `torch_npu` 的实现。让我先看看相关的 dispatch 机制： 

Ran terminal command:  python3 -c "
import torch
import torch_npu
# 查看 aten::linear 在 PrivateUse1 (NPU) 上注册了哪个 kernel
print(torch._C._dispatch_print_registrations_for_dispatch_key('PrivateUse1'))
" 2>&1 | grep -i "linear" | head -20

Ran terminal command:  python3 -c "
import torch
import torch_npu
# 查看 aten::linear 的完整 dispatch 表
print(torch._C._dispatch_dump('aten::linear'))
" 2>&1

Ran terminal command:  python3 -c "
import torch
import torch_npu
# 查看 aten::linear 在 CompositeImplicitAutograd 下会分解成什么
# 以及 aten::mm / aten::addmm 在 NPU 上的注册
print('=== aten::mm dispatch ===')
print(torch._C._dispatch_dump('aten::mm'))
" 2>&1 | grep -E "PrivateUse1|CompositeImplicit|npu_linear|name:|schema:" | head -20

现在我有了足够的信息，可以给出几种实用的检查方法：

---

## 检查 `nn.Linear` 是否调用了 NPU 实现的方法

### 方法一：`torch.autograd.profiler`（最直观）

```python
import torch
import torch.nn as nn

model = nn.Linear(128, 64).npu()
x = torch.randn(32, 128).npu()

with torch.autograd.profiler.profile(use_device='npu') as prof:
    y = model(x)

print(prof.key_averages().table(sort_by="self_npu_time_total"))
```

输出中会看到实际执行的算子名（如 `aten::mm`、`aten::addmm`），以及它们的 NPU 耗时，证明确实在 NPU 上执行。

---

### 方法二：`torch.jit.trace` + dispatch key 追踪（最精确）

```python
import torch
import torch.nn as nn

model = nn.Linear(128, 64).npu()
x = torch.randn(32, 128).npu()

# 查看 aten::linear 的 dispatch 表
print(torch._C._dispatch_dump('aten::linear'))
```

从上面的实际输出可以看到：

- `aten::linear` 注册在 **`CompositeImplicitAutograd`**（别名 key），意味着它会被**分解**为更底层的算子（`aten::addmm` 或 `aten::mm`）
- `aten::mm` 在 **`PrivateUse1`**（即 NPU）上注册了实现：`torch_npu/csrc/aten/RegisterNPU.cpp`

所以 `nn.Linear` → `aten::linear` → `aten::addmm`/`aten::mm` → NPU kernel，**并不经过 `npu_linear`**。

---

### 方法三：`__torch_dispatch__` 拦截（最灵活，可打印每个算子）

```python
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

class OpTracer(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"[DISPATCH] {func}")
        return func(*args, **(kwargs or {}))

model = nn.Linear(128, 64).npu()
x = torch.randn(32, 128).npu()

with OpTracer():
    y = model(x)
```

运行后会打印出实际经过 dispatch 的每一个算子，例如：
```
[DISPATCH] aten::t
[DISPATCH] aten::addmm
```

这是最直接的方式，能看到 `nn.Linear` 实际走了哪条路径。

---

### 结论

根据 dispatch 表分析，`nn.Linear` 在 NPU 上的实际调用链是：

```
nn.Linear.forward()
  └─ F.linear()
       └─ aten::linear  [CompositeImplicitAutograd，分解]
            └─ aten::addmm  [PrivateUse1 → torch_npu/csrc/aten/RegisterNPU.cpp]
```

**`aten::npu_linear` 是 torch_npu 额外提供的独立算子**，`nn.Linear` 默认不走它，除非你显式调用 `torch_npu.npu_linear()`。

User: “torch_npu 安装时在其 dist-info/entry_points.txt 中注册了：”中的entry_points.txt的路径是什么

GitHub Copilot: entry_points.txt

User: 怎么查看npu_linear调用了哪些算子来完成计算

GitHub Copilot: 直接用 `__torch_dispatch__` 拦截 `npu_linear` 的内部调用： 

Ran terminal command:  python3 -c "
import torch
import torch_npu
from torch.utils._python_dispatch import TorchDispatchMode

class OpTracer(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f'[DISPATCH] {func}')
        return func(*args, **(kwargs or {}))

x = torch.randn(4, 128).npu()
w = torch.randn(64, 128).npu()
b = torch.randn(64).npu()

with OpTracer():
    y = torch_npu.npu_linear(x, w, b)
" 2>&1

`__torch_dispatch__` 只能看到 Python dispatch 层的调用，`npu_linear` 是一个**叶子算子**，其内部直接调用 C++/NPU kernel，不会再向上冒泡到 Python dispatch 层。需要换用 profiler 来看底层执行： 

Ran terminal command:  python3 -c "
import torch
import torch_npu

x = torch.randn(4, 128).npu()
w = torch.randn(64, 128).npu()
b = torch.randn(64).npu()

# 预热
torch_npu.npu_linear(x, w, b)
torch.npu.synchronize()

with torch.autograd.profiler.profile(use_device='npu') as prof:
    y = torch_npu.npu_linear(x, w, b)
    torch.npu.synchronize()

print(prof.key_averages().table(sort_by='self_npu_time_total', row_limit=20))
" 2>&1 | grep -v "^\[Warning\]"

Ran terminal command:  python3 -c "
import torch
import torch_npu

x = torch.randn(4, 128).npu()
w = torch.randn(64, 128).npu()
b = torch.randn(64).npu()

torch_npu.npu_linear(x, w, b)
torch.npu.synchronize()

with torch.autograd.profiler.profile(use_device='npu') as prof:
    y = torch_npu.npu_linear(x, w, b)
    torch.npu.synchronize()

# 打印所有事件，包括 NPU kernel 名
for evt in prof.function_events:
    print(f'name={evt.name!r:50s}  cpu={evt.self_cpu_time_total}us  npu={evt.self_device_time_total}us')
" 2>&1 | grep -v "^\[Warning\]\|^\[ERROR\]"

结果非常清晰。

---

## 结论：`npu_linear` 的内部调用链

```
npu::npu_linear
  ├─ aten::t          （对 weight 做转置，等价于 weight.T）
  │    └─ aten::transpose
  │         └─ aten::as_strided
  ├─ empty_tensor     （分配输出 buffer）
  └─ aclnnAddmm       （实际计算：output = input @ weight.T + bias）
```

**核心算子是 `aclnnAddmm`**，即昇腾 CANN 的 `Addmm` 算子（矩阵乘加：$Y = \alpha \cdot A \cdot B + \beta \cdot C$）。

所以 `npu_linear` 和 `nn.Linear`（走 `aten::addmm`）在 NPU 上**最终调用的是同一个 CANN 底层算子 `aclnnAddmm`**，区别只是 `npu_linear` 在 C++ 层直接构造调用，跳过了 Python 层的 `aten::linear` 分解路径。

User: 请对比npu_linear和nn.linear哪个算得快

GitHub Copilot: Ran terminal command:  python3 -c "
import torch
import torch.nn as nn
import torch_npu

WARMUP = 20
REPEAT = 200

configs = [
    (4,    128,  64),
    (32,   1024, 512),
    (128,  4096, 2048),
    (1024, 4096, 2048),
]

print(f'{'Config (B,IN,OUT)':<25} {'nn.Linear(us)':>15} {'npu_linear(us)':>15} {'winner':>10}')
print('-' * 70)

for B, IN, OUT in configs:
    x = torch.randn(B, IN).npu()
    w = torch.randn(OUT, IN).npu()
    b = torch.randn(OUT).npu()
    linear = nn.Linear(IN, OUT).npu()
    # 同步权重，保证公平
    linear.weight.data.copy_(w)
    linear.bias.data.copy_(b)

    # 预热
    for _ in range(WARMUP):
        linear(x)
        torch_npu.npu_linear(x, w, b)
    torch.npu.synchronize()

    # 计时 nn.Linear
    start = torch.npu.Event(enable_timing=True)
    end   = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        linear(x)
    end.record()
    torch.npu.synchronize()
    t_nn = start.elapsed_time(end) * 1000 / REPEAT  # us

    # 计时 npu_linear
    start2 = torch.npu.Event(enable_timing=True)
    end2   = torch.npu.Event(enable_timing=True)
    start2.record()
    for _ in range(REPEAT):
        torch_npu.npu_linear(x, w, b)
    end2.record()
    torch.npu.synchronize()
    t_npu = start2.elapsed_time(end2) * 1000 / REPEAT  # us

    winner = 'npu_linear' if t_npu < t_nn else 'nn.Linear '
    print(f'B={B:<4} IN={IN:<5} OUT={OUT:<5}   {t_nn:>13.2f}   {t_npu:>13.2f}   {winner}  ({min(t_nn,t_npu)/max(t_nn,t_npu)*100:.1f}%)')
" 2>&1 | grep -v "^\[Warning\]\|^\[ERROR\]"


Ran terminal command:  python3 << 'EOF'
import torch
import torch.nn as nn
import torch_npu

WARMUP = 20
REPEAT = 200

configs = [
    (4,    128,  64),
    (32,   1024, 512),
    (128,  4096, 2048),
    (1024, 4096, 2048),
]

header = "{:<30} {:>15} {:>15} {:>12}".format("Config (B,IN,OUT)", "nn.Linear(us)", "npu_linear(us)", "winner")
print(header)
print('-' * 75)

for B, IN, OUT in configs:
    x = torch.randn(B, IN).npu()
    w = torch.randn(OUT, IN).npu()
    b = torch.randn(OUT).npu()
    linear = nn.Linear(IN, OUT).npu()
    linear.weight.data.copy_(w)
    linear.bias.data.copy_(b)

    for _ in range(WARMUP):
        linear(x)
        torch_npu.npu_linear(x, w, b)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end   = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        linear(x)
    end.record()
    torch.npu.synchronize()
    t_nn = start.elapsed_time(end) * 1000 / REPEAT

    start2 = torch.npu.Event(enable_timing=True)
    end2   = torch.npu.Event(enable_timing=True)
    start2.record()
    for _ in range(REPEAT):
        torch_npu.npu_linear(x, w, b)
    end2.record()
    torch.npu.synchronize()
    t_npu = start2.elapsed_time(end2) * 1000 / REPEAT

    winner = "npu_linear" if t_npu < t_nn else "nn.Linear "
    ratio = min(t_nn, t_npu) / max(t_nn, t_npu) * 100
    cfg = "B={} IN={} OUT={}".format(B, IN, OUT)
    print("{:<30} {:>15.2f} {:>15.2f} {:>8}  ({:.1f}%)".format(cfg, t_nn, t_npu, winner, ratio))
EOF


结果非常有意思，分析如下：

---

## 测试结果

| Config (B, IN, OUT)       | nn.Linear (µs) | npu_linear (µs) | 胜者                      |
| ------------------------- | -------------- | --------------- | ------------------------- |
| B=4, IN=128, OUT=64       | 37.34          | **5.24**        | npu_linear（快 **7.1×**） |
| B=32, IN=1024, OUT=512    | 24.85          | **6.99**        | npu_linear（快 **3.6×**） |
| B=128, IN=4096, OUT=2048  | 103.21         | 103.36          | 基本持平（差 0.1%）       |
| B=1024, IN=4096, OUT=2048 | 759.41         | 759.40          | 基本持平（差 0.0%）       |

---

## 原因分析

**小矩阵时 `npu_linear` 显著更快**，原因是调用路径更短：

```
nn.Linear:
  Python forward() → F.linear() → aten::linear → aten::t → aten::addmm → aclnnAddmm
  （多层 Python/C++ dispatch 开销）

npu_linear:
  torch_npu.npu_linear() → aclnnAddmm
  （直接调用，dispatch 开销极小）
```

小矩阵时，**NPU kernel 本身执行极快**（几微秒），dispatch 开销占比很大，`npu_linear` 的路径更短所以优势明显。

**大矩阵时两者完全一致**，因为 NPU kernel 执行时间（数百微秒）远大于 dispatch 开销，两者最终都落到同一个 `aclnnAddmm` kernel，计算时间完全相同。

---

## 结论

- **小 batch / 小矩阵**：优先用 `npu_linear`，可节省大量 dispatch overhead
- **大矩阵计算密集场景**：两者无差异，用 `nn.Linear` 可读性更好
- 实际训练中矩阵通常较大，差异可忽略；**推理小 batch 场景下 `npu_linear` 有明显优势**

User: 为什么小shape下dispatch时间明显，而大shape下dispatch时间几乎无感知。这里我把npu_linear和nn.Linear的时间差理解为dispatch的时间度量

GitHub Copilot: 这个理解是正确的。原因在于 **NPU 的异步执行模型**：

---

## 核心原因：NPU 算子是异步下发的

NPU 与 CPU 的交互模式是：

```
CPU                          NPU
 │                            │
 ├─ dispatch overhead ──►     │
 ├─ 下发 aclnnAddmm 命令 ──►  │ ← 开始排队
 │                            │   执行 kernel
 │  (CPU 立即返回)             │
 │                            │
```

CPU 调用 `aclnnAddmm` 后**立即返回**，不等 NPU 算完。NPU 在自己的队列里异步执行。

---

## 为什么小 shape 时差异大

小矩阵 kernel 执行极快（比如 5µs），**NPU 算完的速度比 CPU 下发下一条命令还快**，NPU 会出现**空闲等待**：

```
时间轴：
CPU:  [dispatch_nn(37µs)][dispatch_npu(5µs)]
NPU:  [kernel(~5µs)][idle...][kernel(~5µs)]
                     ↑
              NPU 在等 CPU 慢吞吞地 dispatch
```

此时 `elapsed_time(start, end)` 测的是 **CPU dispatch 时间 + NPU kernel 时间**，而 NPU kernel 时间极短，所以 **dispatch overhead 占主导**，两者差异显著。

---

## 为什么大 shape 时差异消失

大矩阵 kernel 执行很慢（比如 760µs），**CPU dispatch 速度远快于 NPU 执行速度**，NPU 始终满载，CPU 的 dispatch overhead 被完全"藏"在 NPU 执行时间里：

```
时间轴：
CPU:  [dispatch(~10µs)][空闲等 NPU...]
NPU:  [          kernel(760µs)        ]
       ↑
  dispatch 时间完全被 kernel 时间覆盖
```

`elapsed_time` 测的几乎全是 **NPU kernel 时间**，两者 dispatch 路径的差异（几微秒）相对 760µs 完全淹没，所以看起来一样。

---

## 用一个公式表达

$$T_{\text{measured}} = T_{\text{dispatch}} + T_{\text{kernel}}$$

$$\Delta T = T_{\text{dispatch}}^{\text{nn.Linear}} - T_{\text{dispatch}}^{\text{npu\_linear}}$$

| 场景     | $T_{\text{kernel}}$ | $\Delta T$ | $\Delta T / T_{\text{measured}}$ |
| -------- | ------------------- | ---------- | -------------------------------- |
| 小 shape | ~5 µs               | ~32 µs     | **86%**（差异显著）              |
| 大 shape | ~760 µs             | ~0 µs      | **<0.1%**（完全淹没）            |

所以你的理解完全正确：**时间差就是 dispatch overhead 的度量**，只是大 shape 下它被 kernel 时间淹没了，并非消失了，而是**相对占比趋近于零**。