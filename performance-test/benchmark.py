#!/usr/bin/env python3

-- coding: utf-8 --
"""
OpenAI Embedding API 压测工具（全局 batch 队列版）

改动重点：

复用 urllib3 连接池

支持 warmup（不计入正式结果）

固定随机种子，保证可复现

支持每秒打印 RPS

强制构造固定长度输入，避免 context_length 失效导致 QPS 虚高

改为“全局 batch 队列 + 多线程抢 batch”

不再先按线程切样本

避免高线程下制造大量碎批
"""

import random
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Union
import urllib3
import datetime
import logging
import numpy as np
import json

from openpyxl import Workbook

=========================
日志配置
=========================
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)
logger = logging.getLogger(name)

=========================
统计工具
=========================
def _safe_latency_stats(values: List[float]) -> Dict[str, float]:
if not values:
return {
"avg": 0.0,
"median": 0.0,
"p95": 0.0,
"min": 0.0,
"max": 0.0,
}

array = np.asarray(values, dtype=np.float64)
return {
    "avg": float(np.mean(array)),
    "median": float(np.percentile(array, 50)),
    "p95": float(np.percentile(array, 95)),
    "min": float(np.min(array)),
    "max": float(np.max(array)),
}
=========================
Benchmark 类
=========================
class Benchmark:
"""OpenAI Embedding API 压测类（全局 batch 队列版）"""

def __init__(self, url: str, batch_size: int = 10, test_type: str = "embedding"):
    self.url = url
    self.batch_size = batch_size
    self.test_type = test_type

    # 全局统计
    self.response_times: List[float] = []
    self.success_response_times: List[float] = []
    self.success_count = 0
    self.error_count = 0
    self.batch_count = 0
    self.success_batch_count = 0
    self.lock = threading.Lock()

    # 连接池复用
    self.http = urllib3.PoolManager(
        num_pools=128,
        maxsize=1024,
        block=False
    )
    self.timeout = urllib3.Timeout(connect=2.0, read=60.0)

    # 每秒 RPS 统计
    self._meter_stop = threading.Event()
    self._meter_thread = None
    self._meter_last_t = None
    self._meter_last_success = 0
    self._meter_last_total = 0

# -------------------------
# 每秒 RPS 打印
# -------------------------
def _rps_meter_loop(self):
    while not self._meter_stop.wait(1.0):
        with self.lock:
            now = time.time()
            success_now = self.success_count
            total_now = self.success_count + self.error_count

            dt = now - self._meter_last_t if self._meter_last_t else 1.0
            if dt <= 0:
                dt = 1.0

            succ_rps = (success_now - self._meter_last_success) / dt
            total_rps = (total_now - self._meter_last_total) / dt

            self._meter_last_t = now
            self._meter_last_success = success_now
            self._meter_last_total = total_now

        logger.info(
            f"[RPS] instant_effective_qps={succ_rps:.2f}, "
            f"instant_attempted_qps={total_rps:.2f}, "
            f"success={success_now}, total={total_now}"
        )

def _start_rps_meter(self):
    self._meter_stop.clear()
    with self.lock:
        self._meter_last_t = time.time()
        self._meter_last_success = self.success_count
        self._meter_last_total = self.success_count + self.error_count
    self._meter_thread = threading.Thread(target=self._rps_meter_loop, daemon=True)
    self._meter_thread.start()

def _stop_rps_meter(self):
    self._meter_stop.set()
    if self._meter_thread is not None:
        self._meter_thread.join(timeout=2.0)

# -------------------------
# 请求函数
# -------------------------
def send_emb_request(self, data: Dict[str, Any]) -> List[Any]:
    encoded_data = json.dumps(data).encode("utf-8")
    r = self.http.request(
        "POST",
        self.url,
        body=encoded_data,
        headers={"content-type": "application/json;charset=UTF-8"},
        timeout=self.timeout
    )

    result = []
    if r.status == 200:
        res = json.loads(r.data)
        for e in res["data"]:
            result.append(e["embedding"])
        return result
    return []

def send_rerank_request(self, data: Dict[str, Any]) -> List[Any]:
    encoded_data = json.dumps(data).encode("utf-8")
    r = self.http.request(
        "POST",
        self.url,
        body=encoded_data,
        headers={"content-type": "application/json;charset=UTF-8"},
        timeout=self.timeout
    )

    if r.status == 200:
        res = json.loads(r.data)
        return res["results"]
    return []

def get_batch_embedding(self, texts: List[str]) -> Tuple[bool, float, str]:
    start_time = time.time()
    try:
        response = self.send_emb_request({"input": texts})
        response_time = time.time() - start_time
        if response and len(response) == len(texts):
            return True, response_time, ""
        return False, response_time, (
            f"Response data length mismatch: expected {len(texts)}, "
            f"got {len(response) if response else 0}"
        )
    except Exception as e:
        response_time = time.time() - start_time
        return False, response_time, str(e)

def get_batch_rerank(self, texts: Dict[str, Any]) -> Tuple[bool, float, str]:
    start_time = time.time()
    try:
        response = self.send_rerank_request(
            {"query": texts["query"][:20], "documents": texts["documents"]}
        )
        response_time = time.time() - start_time
        if response and len(response) == len(texts["documents"]):
            return True, response_time, ""
        return False, response_time, (
            f"Response data length mismatch: expected {len(texts['documents'])}, "
            f"got {len(response) if response else 0}"
        )
    except Exception as e:
        response_time = time.time() - start_time
        return False, response_time, str(e)

# -------------------------
# 构造全局 batch 列表
# -------------------------
def _build_embedding_batches(
    self,
    inputs: List[str],
    requests_per_thread: int = None,
    num_threads: int = None
) -> List[List[str]]:
    if requests_per_thread is not None and num_threads is not None:
        target_count = min(len(inputs), requests_per_thread * num_threads)
        inputs = inputs[:target_count]

    batches = []
    for i in range(0, len(inputs), self.batch_size):
        batch = inputs[i:i + self.batch_size]
        if batch:
            batches.append(batch)
    return batches

def _build_rerank_batches(
    self,
    inputs: List[Dict[str, Any]],
    requests_per_thread: int = None,
    num_threads: int = None
) -> List[Dict[str, Any]]:
    if requests_per_thread is not None and num_threads is not None:
        target_count = min(len(inputs), requests_per_thread * num_threads)
        inputs = inputs[:target_count]
    return list(inputs)

# -------------------------
# 线程工作函数：从全局队列抢 batch
# -------------------------
def _worker_from_queue(
    self,
    task_queue: "queue.Queue[Dict[str, Any]]",
    thread_id: int
) -> Dict[str, Any]:
    thread_stats = {
        "thread_id": thread_id,
        "success_count": 0,
        "error_count": 0,
        "response_times": [],
        "success_response_times": [],
        "errors": [],
        "batch_count": 0,
        "success_batch_count": 0,
    }

    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        try:
            if self.test_type == "embedding":
                batch_texts = task["payload"]
                current_batch_size = len(batch_texts)
                success, response_time, error_msg = self.get_batch_embedding(batch_texts)

            elif self.test_type == "rerank":
                batch_texts = task["payload"]
                current_batch_size = 1
                success, response_time, error_msg = self.get_batch_rerank(batch_texts)

            else:
                raise ValueError(f"不支持的测试类型: {self.test_type}")

            thread_stats["response_times"].append(response_time)
            thread_stats["batch_count"] += 1

            if success:
                thread_stats["success_count"] += current_batch_size
                thread_stats["success_response_times"].append(response_time)
                thread_stats["success_batch_count"] += 1
            else:
                thread_stats["error_count"] += current_batch_size
                thread_stats["errors"].append(
                    f"global_batch_id={task['batch_id']}: {error_msg}"
                )

            with self.lock:
                self.response_times.append(response_time)
                self.batch_count += 1
                if success:
                    self.success_response_times.append(response_time)
                    self.success_batch_count += 1
                    self.success_count += current_batch_size
                else:
                    self.error_count += current_batch_size

        finally:
            task_queue.task_done()

    return thread_stats

# -------------------------
# 单轮执行
# -------------------------
def _run_once(
    self,
    inputs: Union[List[str], List[Dict[str, Any]]],
    num_threads: int,
    requests_per_thread: int = None,
    print_rps: bool = False
) -> Dict[str, Any]:
    # 重置统计
    self.response_times = []
    self.success_response_times = []
    self.success_count = 0
    self.error_count = 0
    self.batch_count = 0
    self.success_batch_count = 0

    # 构造全局任务
    if self.test_type == "embedding":
        all_batches = self._build_embedding_batches(
            inputs=inputs,
            requests_per_thread=requests_per_thread,
            num_threads=num_threads
        )
    elif self.test_type == "rerank":
        all_batches = self._build_rerank_batches(
            inputs=inputs,
            requests_per_thread=requests_per_thread,
            num_threads=num_threads
        )
    else:
        raise ValueError(f"不支持的测试类型: {self.test_type}")

    task_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    for batch_id, payload in enumerate(all_batches):
        task_queue.put({
            "batch_id": batch_id,
            "payload": payload
        })

    # 打印一下 batch 分布，确认没有被线程切碎
    if self.test_type == "embedding":
        batch_sizes = [len(b) for b in all_batches]
        if batch_sizes:
            logger.info(
                f"global_batches={len(all_batches)}, "
                f"batch_size_min={min(batch_sizes)}, "
                f"batch_size_max={max(batch_sizes)}, "
                f"batch_size_avg={sum(batch_sizes) / len(batch_sizes):.2f}"
            )
            logger.info(f"前10个 batch 大小: {batch_sizes[:10]}")
    else:
        logger.info(f"global_batches={len(all_batches)}")

    if print_rps:
        self._start_rps_meter()

    start_time = time.time()
    thread_results = []
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self._worker_from_queue, task_queue, i)
                for i in range(num_threads)
            ]
            for future in as_completed(futures):
                thread_results.append(future.result())
    finally:
        if print_rps:
            self._stop_rps_meter()

    total_time = time.time() - start_time
    total_requests = self.success_count + self.error_count

    attempted_qps = total_requests / total_time if total_time > 0 else 0.0
    effective_qps = self.success_count / total_time if total_time > 0 else 0.0
    attempted_batch_qps = self.batch_count / total_time if total_time > 0 else 0.0
    effective_batch_qps = self.success_batch_count / total_time if total_time > 0 else 0.0

    response_stats = _safe_latency_stats(self.response_times)
    success_response_stats = _safe_latency_stats(self.success_response_times)
    success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0.0

    return {
        "num_thread": num_threads,
        "total_time": total_time,
        "total_requests": total_requests,
        "total_batches": self.batch_count,
        "success_batches": self.success_batch_count,
        "planned_global_batches": len(all_batches),
        "batch_size": self.batch_size,
        "success_count": self.success_count,
        "error_count": self.error_count,
        "success_rate": success_rate,
        "qps": effective_qps,
        "attempted_qps": attempted_qps,
        "batch_qps": effective_batch_qps,
        "attempted_batch_qps": attempted_batch_qps,
        "avg_response_time": response_stats["avg"],
        "median_response_time": response_stats["median"],
        "p95_response_time": response_stats["p95"],
        "min_response_time": response_stats["min"],
        "max_response_time": response_stats["max"],
        "avg_success_response_time": success_response_stats["avg"],
        "median_success_response_time": success_response_stats["median"],
        "p95_success_response_time": success_response_stats["p95"],
        "min_success_response_time": success_response_stats["min"],
        "max_success_response_time": success_response_stats["max"],
        "thread_results": thread_results,
    }

# -------------------------
# 对外 benchmark
# -------------------------
def benchmark(
    self,
    inputs: Union[List[str], List[Dict[str, Any]]],
    num_threads: int = 10,
    requests_per_thread: int = None,
    warmup_rounds: int = 1,
    print_rps: bool = True
) -> Dict[str, Any]:
    logger.info(
        f"开始{self.test_type}压测，线程数: {num_threads}, 输入数量: {len(inputs)}, "
        f"批量大小: {self.batch_size}, warmup_rounds: {warmup_rounds}"
    )

    if warmup_rounds > 0:
        logger.info("开始 warmup ...")
        for i in range(warmup_rounds):
            _ = self._run_once(
                inputs=inputs,
                num_threads=num_threads,
                requests_per_thread=requests_per_thread,
                print_rps=False
            )
            logger.info(f"warmup round {i + 1}/{warmup_rounds} 完成")
        logger.info("warmup 完成，开始正式计时")

    return self._run_once(
        inputs=inputs,
        num_threads=num_threads,
        requests_per_thread=requests_per_thread,
        print_rps=print_rps
    )

def print_results(self, results: Dict[str, Any]):
    print("\n" + "=" * 60)
    print(f"OpenAI {self.test_type} benchmark results")
    print("=" * 60)
    print(f"test_type: {self.test_type}")
    print(f"total_time: {results['total_time']:.2f} s")
    print(f"num_thread: {results['num_thread']}")
    print(f"batch_size: {results['batch_size']}")
    print(f"planned_global_batches: {results['planned_global_batches']}")
    print(f"total_batches: {results['total_batches']}")
    print(f"success_batches: {results['success_batches']}")
    print(f"total_requests: {results['total_requests']}")
    print(f"success_count: {results['success_count']}")
    print(f"error_count: {results['error_count']}")
    print(f"success_rate: {results['success_rate']:.2f}%")
    print(f"effective_qps: {results['qps']:.2f}")
    print(f"attempted_qps: {results['attempted_qps']:.2f}")
    print(f"effective_batch_qps: {results['batch_qps']:.2f}")
    print(f"attempted_batch_qps: {results['attempted_batch_qps']:.2f}")
    print(f"avg_success_response_time: {results['avg_success_response_time']:.3f} s")
    print(f"median_success_response_time: {results['median_success_response_time']:.3f} s")
    print(f"p95_success_response_time: {results['p95_success_response_time']:.3f} s")
    print(f"min_success_response_time: {results['min_success_response_time']:.3f} s")
    print(f"max_success_response_time: {results['max_success_response_time']:.3f} s")
    print(f"avg_all_response_time: {results['avg_response_time']:.3f} s")
    print(f"median_all_response_time: {results['median_response_time']:.3f} s")
    print(f"p95_all_response_time: {results['p95_response_time']:.3f} s")
    print("=" * 60)
=========================
测试语料
=========================
text = """
上图为俄罗斯冷冻阿拉斯加狭鳕的出口格局,俄罗斯作为全球最大的捕捞阿拉斯加狭鳕的国家这些年的出口量比较稳定,
平均每年出口80万吨左右的冷冻阿拉斯加狭鳕。主要出口到中国和韩国。中国能够成为全球最重要的鳕鱼加工中心,
很多俄罗斯捕捞阿拉斯加狭鳕的船队并没有捕捞加工一体化的能力，他们往往将捕捞上来的鳕鱼冷冻后送到中国进行加工。
美国冷冻阿拉斯加狭鳕的出口量近些年一直是下滑的，这是因为美国阿拉斯加狭鳕的捕捞量不如俄罗斯高，而且有很大一部分
在渔船上被加工成了鱼糜，然后再出口，而且美国政府鼓励国内消费本土产的鳕鱼。挪威冷冻黑线鳕、蓝鳕、真鳕、盐渍真鳕、
干制真鳕等产品也在全球贸易中占据重要位置。中国每年进口冷冻真鳕、冷冻阿拉斯加狭鳕、黑线鳕、绿青鳕、无须鳕、蓝鳕
等大量原料进行加工，再出口到欧美、日韩等国家和地区。随着一次冷冻、船冻、鱼柳、鱼糜、冰鲜等消费趋势变化，
不同国家的加工、出口与进口格局也发生了持续变化。
""".strip()

=========================
固定长度文本构造
=========================
def build_fixed_length_text(base_text: str, target_length: int, prefix: str = "") -> str:
if target_length <= 0:
raise ValueError("target_length 必须大于 0")

base_text = (base_text or "").strip()
if not base_text:
    raise ValueError("base_text 不能为空")

payload = prefix + base_text
while len(payload) < target_length:
    payload += base_text

return payload[:target_length]
def create_test_texts(count: int = 1024, context_length: int = 512) -> List[str]:
texts = []
for _ in range(count):
s = str(random.randint(1, 10000))
sample = build_fixed_length_text(
base_text=text,
target_length=context_length,
prefix=s
)
texts.append(sample)
return texts

def create_test_rerank_inputs(
count: int = 1024,
context_length: int = 512,
batch_size: int = 3
) -> List[Dict[str, Any]]:
texts = []
rounds = int(count / batch_size)

for _ in range(rounds):
    docs = []
    for _ in range(batch_size):
        s = str(random.randint(1, 10000))
        doc = build_fixed_length_text(
            base_text=text,
            target_length=context_length,
            prefix=s
        )
        docs.append(doc)

    query = build_fixed_length_text(
        base_text="中国进口鳕鱼主要来自哪些国家，这些国家的供给格局和加工链路有什么变化？",
        target_length=min(128, context_length)
    )
    texts.append({"query": query, "documents": docs})

return texts
def validate_inputs(inputs: Union[List[str], List[Dict[str, Any]]], test_type: str, context_length: int):
if not inputs:
raise ValueError("inputs 为空，无法开始测试")

if test_type == "embedding":
    lengths = [len(x) for x in inputs[:10]]
    logger.info(f"embedding 样本前10条长度: {lengths}")
    logger.info(f"embedding 第1条样本前120字符: {inputs[0][:120]}")
    if min(lengths) < context_length:
        raise ValueError(
            f"检测到 embedding 样本长度不足，预期 >= {context_length}，实际最小值 = {min(lengths)}"
        )

elif test_type == "rerank":
    doc_lengths = []
    for item in inputs[:5]:
        for doc in item["documents"]:
            doc_lengths.append(len(doc))
    logger.info(f"rerank 文档样本长度: {doc_lengths[:10]}")
    logger.info(f"rerank 第1条 query 前120字符: {inputs[0]['query'][:120]}")
    if doc_lengths and min(doc_lengths) < context_length:
        raise ValueError(
            f"检测到 rerank 文档长度不足，预期 >= {context_length}，实际最小值 = {min(doc_lengths)}"
        )
=========================
主流程
=========================
def main(
url: str,
total_count_text: int,
context_length: int,
batch_size: int,
num_thread: int,
test_type: str,
warmup_rounds: int = 1,
print_rps: bool = True
):
print(f"开始 {test_type} 接口压测...")

benchmark = Benchmark(
    url=url,
    batch_size=batch_size,
    test_type=test_type,
)

if test_type == "embedding":
    test_inputs = create_test_texts(total_count_text, context_length)
elif test_type == "rerank":
    test_inputs = create_test_rerank_inputs(total_count_text, context_length, batch_size)
else:
    raise ValueError(f"不支持的测试类型: {test_type}")

validate_inputs(test_inputs, test_type, context_length)

results = benchmark.benchmark(
    inputs=test_inputs,
    num_threads=num_thread,
    requests_per_thread=None,
    warmup_rounds=warmup_rounds,
    print_rps=print_rps
)

benchmark.print_results(results)
return results
if name == "main":
# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
logger.info(f"固定随机种子: {SEED}")

batch_sizes = [20]
num_threads = [8]
total_count_text = 1024
context_lengths = [1024]
warmup_rounds = 1

wb = Workbook()
ws = wb.create_sheet("test", 0)

headers = [
    "num_thread",
    "total_time (s)",
    "context_length",
    "total_requests",
    "planned_global_batches",
    "total_batches",
    "success_batches",
    "batch_size",
    "success_count",
    "error_count",
    "success_rate",
    "effective_qps",
    "attempted_qps",
    "effective_batch_qps",
    "attempted_batch_qps",
    "avg_success_response_time (s)",
    "median_success_response_time (s)",
    "p95_success_response_time (s)",
    "min_success_response_time (s)",
    "max_success_response_time (s)",
    "avg_response_time (s)",
    "median_response_time (s)",
    "p95_response_time (s)",
    "min_response_time (s)",
    "max_response_time (s)",
]
ws.append(headers)

# 4090 的 API
url = "http://90.90.102.12:8998/v1/embeddings"

for context_length in context_lengths:
    for batch_size in batch_sizes:
        for num_thread in num_threads:
            result = main(
                url=url,
                total_count_text=total_count_text,
                context_length=context_length,
                batch_size=batch_size,
                num_thread=num_thread,
                test_type="embedding",
                warmup_rounds=warmup_rounds,
                print_rps=True
            )

            row = [
                result["num_thread"],
                result["total_time"],
                context_length,
                result["total_requests"],
                result["planned_global_batches"],
                result["total_batches"],
                result["success_batches"],
                result["batch_size"],
                result["success_count"],
                result["error_count"],
                result["success_rate"],
                result["qps"],
                result["attempted_qps"],
                result["batch_qps"],
                result["attempted_batch_qps"],
                result["avg_success_response_time"],
                result["median_success_response_time"],
                result["p95_success_response_time"],
                result["min_success_response_time"],
                result["max_success_response_time"],
                result["avg_response_time"],
                result["median_response_time"],
                result["p95_response_time"],
                result["min_response_time"],
                result["max_response_time"],
            ]
            ws.append(row)

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
filename = f"Qwen3-Embedding-0.6B-4090-global-batch-queue-{current_time}.xlsx"
wb.save(filename)
print(f"测试完成，结果已保存到: {filename}")

