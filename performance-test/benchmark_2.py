#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Embedding API 压测工具
使用多线程对OpenAI embedding接口进行压力测试，统计平均耗时和QPS
支持批量处理，默认batch size = 10
"""
import argparse
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Union
import urllib3

import logging
import numpy as np
import json

from openpyxl import Workbook

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


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


class Benchmark:
    """OpenAI Embedding API 压测类"""

    def __init__(self, url: str, batch_size: int = 10, test_type: str = "embedding"):
        """
        初始化压测工具

        Args:
            url: API基础URL（可选，用于自定义端点）
            batch_size: 批量处理大小，默认为10
            test_type: 测试类型，"embedding" 或 "score"
        """
        self.url = url
        self.batch_size = batch_size
        self.test_type = test_type

        # 统计数据
        self.response_times = []
        self.success_response_times = []
        self.success_count = 0
        self.error_count = 0
        self.batch_count = 0
        self.success_batch_count = 0
        self.lock = threading.Lock()

    def send_emb_request(self, data):
        http = urllib3.PoolManager()
        encoded_data = json.dumps(data).encode("utf-8")
        r = http.request(
            "POST",
            self.url,
            body=encoded_data,
            headers={
                'content-type': 'application/json;charset=UTF-8'
            }
        )
        result = []
        if r.status == 200:
            res = json.loads(r.data)
            for e in res["data"]:
                result.append(e["embedding"])

            return result
        else:
            return []

    def send_rerank_request(self, data):
        http = urllib3.PoolManager()
        encoded_data = json.dumps(data).encode("utf-8")
        r = http.request(
            "POST",
            self.url,
            body=encoded_data,
            headers={
                'content-type': 'application/json;charset=UTF-8'
            }
        )

        if r.status == 200:
            res = json.loads(r.data)
            return res["results"]
        else:
            return []

    def get_batch_embedding(self, texts: List[str]) -> Tuple[bool, float, str]:
        """
        获取批量文本的embedding

        Args:
            texts: 输入文本列表
            client: OpenAI客户端实例

        Returns:
            Tuple[bool, float, str]: (是否成功, 响应时间, 错误信息)
        """
        start_time = time.time()
        try:
            response = self.send_emb_request({"input": texts})
            end_time = time.time()
            response_time = end_time - start_time
            # 验证响应
            if response and len(response) == len(texts):
                return True, response_time, ""
            else:
                return False, response_time, f"Response data length mismatch: expected {len(texts)}, got {len(response) if response else 0}"

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            error_msg = str(e)
            logger.error(f"批量API调用失败: {error_msg}")
            return False, response_time, error_msg

    def get_batch_rerank(self, texts: Dict[str, str]) -> Tuple[bool, float, str]:
        """
        获取批量文本的embedding

        Args:
            texts: 输入文本列表
            client: OpenAI客户端实例

        Returns:
            Tuple[bool, float, str]: (是否成功, 响应时间, 错误信息)
        """
        start_time = time.time()
        try:
            response = self.send_rerank_request({"query": texts["query"][:20], "documents": texts["documents"]})
            end_time = time.time()
            response_time = end_time - start_time

            # 验证响应
            if response and len(response) == len(texts["documents"]):
                return True, response_time, ""
            else:
                return False, response_time, f"Response data length mismatch: expected {len(texts['documents'])}, got {len(response) if response else 0}"

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            error_msg = str(e)
            logger.error(f"批量API调用失败: {error_msg}")
            return False, response_time, error_msg

    def _worker_thread(self, texts: Union[List[str], List[Dict[str, str]]], thread_id: int) -> Dict[str, Any]:
        """
        工作线程函数

        Args:
            texts:
                  test_type为embeedding时该线程处理的文本列表，类型为List[str]
                  test_type为reranker时该线程处理的query-documents列表，类型为List[Dict]
            thread_id: 线程ID

        Returns:
            Dict: 线程执行结果统计
        """
        thread_stats = {
            'thread_id': thread_id,
            'success_count': 0,
            'error_count': 0,
            'response_times': [],
            'success_response_times': [],
            'errors': [],
            'batch_count': 0,
            'success_batch_count': 0,
            'total_count_text': len(texts)
        }

        if self.test_type == "embedding":
            logger.info(f"线程 {thread_id} 开始处理 {len(texts)} 个文本，批量大小: {self.batch_size}")

            # 将文本分批处理
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_size = len(batch_texts)

                success, response_time, error_msg = self.get_batch_embedding(batch_texts)

                thread_stats['response_times'].append(response_time)
                thread_stats['batch_count'] += 1

                if success:
                    thread_stats['success_count'] += batch_size
                    thread_stats['success_response_times'].append(response_time)
                    thread_stats['success_batch_count'] += 1
                else:
                    thread_stats['error_count'] += batch_size
                    thread_stats['errors'].append(f"Batch {thread_stats['batch_count']}: {error_msg}")

                # 更新全局统计
                with self.lock:
                    self.response_times.append(response_time)
                    self.batch_count += 1
                    if success:
                        self.success_response_times.append(response_time)
                        self.success_batch_count += 1
                        self.success_count += batch_size
                    else:
                        self.error_count += batch_size

        elif self.test_type == "rerank":
            # 将文本分批处理
            for i in range(0, len(texts)):
                success, response_time, error_msg = self.get_batch_rerank(texts[i])
                thread_stats['response_times'].append(response_time)
                thread_stats['batch_count'] += 1

                if success:
                    thread_stats['success_count'] += 1
                    thread_stats['success_response_times'].append(response_time)
                    thread_stats['success_batch_count'] += 1
                else:
                    thread_stats['error_count'] += 1
                    thread_stats['errors'].append(f"Batch {thread_stats['batch_count']}: {error_msg}")

                # 更新全局统计
                with self.lock:
                    self.response_times.append(response_time)
                    self.batch_count += 1
                    if success:
                        self.success_response_times.append(response_time)
                        self.success_batch_count += 1
                        self.success_count += 1
                    else:
                        self.error_count += 1

        logger.info(f"线程 {thread_id} 完成，处理了 {thread_stats['batch_count']} 个批次，"
                    f"成功: {thread_stats['success_count']}, 失败: {thread_stats['error_count']}")
        return thread_stats

    def benchmark(self, inputs: Union[List[str], List[np.ndarray]], num_threads: int = 10,
                  requests_per_thread: int = None) -> Dict[str, Any]:
        """
        执行压测

        Args:
            inputs: 要测试的输入列表（文本列表或score输入数组列表）
            num_threads: 线程数量
            requests_per_thread: 每个线程处理的请求数（如果为None，则平均分配所有输入）

        Returns:
            Dict: 压测结果统计
        """
        input_type = "文本" if self.test_type == "embedding" else "reranker输入"
        logger.info(
            f"开始{self.test_type}压测，线程数: {num_threads}, {input_type}数量: {len(inputs)}, 批量大小: {self.batch_size}")

        # 重置统计数据
        self.response_times = []
        self.success_response_times = []
        self.success_count = 0
        self.error_count = 0
        self.batch_count = 0
        self.success_batch_count = 0

        # 准备测试数据
        if requests_per_thread:
            # 每个线程处理指定数量的请求
            test_inputs = []
            for _ in range(num_threads):
                thread_inputs = inputs[:requests_per_thread] if len(inputs) >= requests_per_thread else inputs
                test_inputs.append(thread_inputs)
        else:
            # 平均分配所有输入
            chunk_size = len(inputs) // num_threads
            test_inputs = []
            for i in range(num_threads):
                start_idx = i * chunk_size
                if i == num_threads - 1:  # 最后一个线程处理剩余的所有输入
                    end_idx = len(inputs)
                else:
                    end_idx = start_idx + chunk_size
                test_inputs.append(inputs[start_idx:end_idx])

        # 开始压测
        print("total = ", len(test_inputs))
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, thread_inputs in enumerate(test_inputs):
                if thread_inputs:  # 只提交非空的输入列表
                    print("thread_inputs", len(thread_inputs))
                    future = executor.submit(self._worker_thread, thread_inputs, i)
                    futures.append(future)

            # 等待所有线程完成
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"线程执行异常: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        # 计算统计结果
        total_requests = self.success_count + self.error_count
        attempted_qps = total_requests / total_time if total_time > 0 else 0
        effective_qps = self.success_count / total_time if total_time > 0 else 0
        attempted_batch_qps = self.batch_count / total_time if total_time > 0 else 0
        effective_batch_qps = (
            self.success_batch_count / total_time if total_time > 0 else 0
        )

        response_stats = _safe_latency_stats(self.response_times)
        success_response_stats = _safe_latency_stats(self.success_response_times)

        success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0

        results = {
            'num_thread': num_threads,
            'total_time': total_time,
            'total_requests': total_requests,
            'total_batches': self.batch_count,
            'success_batches': self.success_batch_count,
            'batch_size': self.batch_size,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'qps': effective_qps,
            'attempted_qps': attempted_qps,
            'batch_qps': effective_batch_qps,
            'attempted_batch_qps': attempted_batch_qps,
            'avg_response_time': response_stats['avg'],
            'median_response_time': response_stats['median'],
            'p95_response_time': response_stats['p95'],
            'min_response_time': response_stats['min'],
            'max_response_time': response_stats['max'],
            'avg_success_response_time': success_response_stats['avg'],
            'median_success_response_time': success_response_stats['median'],
            'p95_success_response_time': success_response_stats['p95'],
            'min_success_response_time': success_response_stats['min'],
            'max_success_response_time': success_response_stats['max'],
        }

        return results

    def _print_results_legacy_broken(self, results: Dict[str, Any]):
        """打印压测结果"""
        test_name = "Embedding" if self.test_type == "embedding" else "Score"
        item_name = "文本" if self.test_type == "embedding" else "输入"

        print("\n" + "=" * 60)
        print(f"OpenAI {test_name} API 压测结果 (批量处理)")
        print("=" * 60)
        print(f"测试类型: {self.test_type}")
        print(f"总耗时: {results['total_time']:.2f} 秒")
        print(f"num_thread: {results['num_thread']}")
        print(f"batch_size: {results['batch_size']}")
        print(f"total_batches: {results['total_batches']}")
        print(f"总{item_name}数: {results['total_requests']}")
        print(f"成功{item_name}数: {results['success_count']}")
        print(f"失败{item_name}数: {results['error_count']}")
        print(f"成功率: {results['success_rate']:.2f}%")
        print(f"QPS (每秒{item_name}数): {results['qps']:.2f}")
        print(f"批次QPS (每秒批次数): {results['batch_qps']:.2f}")
        print(f"平均响应时间: {results['avg_response_time']:.3f} 秒")
        print(f"中位数响应时间: {results['median_response_time']:.3f} 秒")
        print(f"p95数响应时间: {results['p95_response_time']:.3f} 秒")
        print(f"最小响应时间: {results['min_response_time']:.3f} 秒")
        print(f"最大响应时间: {results['max_response_time']:.3f} 秒")
        print("=" * 60)

    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print(f"OpenAI {self.test_type} benchmark results")
        print("=" * 60)
        print(f"test_type: {self.test_type}")
        print(f"total_time: {results['total_time']:.2f} s")
        print(f"num_thread: {results['num_thread']}")
        print(f"batch_size: {results['batch_size']}")
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
        print(
            f"avg_success_response_time: "
            f"{results['avg_success_response_time']:.3f} s"
        )
        print(
            f"median_success_response_time: "
            f"{results['median_success_response_time']:.3f} s"
        )
        print(
            f"p95_success_response_time: "
            f"{results['p95_success_response_time']:.3f} s"
        )
        print(
            f"min_success_response_time: "
            f"{results['min_success_response_time']:.3f} s"
        )
        print(
            f"max_success_response_time: "
            f"{results['max_success_response_time']:.3f} s"
        )
        print(f"avg_all_response_time: {results['avg_response_time']:.3f} s")
        print(f"median_all_response_time: {results['median_response_time']:.3f} s")
        print(f"p95_all_response_time: {results['p95_response_time']:.3f} s")
        print("=" * 60)


chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

text = "<br><img><br>上图为俄罗斯冷冻阿拉斯加狭鳕的出口格局,俄罗斯作为全球最大的捕捞阿拉斯加狭鳕的国家这些年的出口量比较稳定,平均每年出口80万吨左右的冷冻阿拉斯加狭鳕。主要出口到中国(加工成鱼柳等产品后出口到世界各地)和韩国(本地消费一部分,加工后出口到日本一部分)。<br>这就是为什么中国能够成为全球最重要的鳕鱼加工中心,很多俄罗斯捕捞阿拉斯加狭鳕的船队并没有捕捞加工一体化的能力,他们往往将捕捞上来的鳕鱼冷冻后送到中国进行加工。<br><img><br>如上图所示,美国冷冻阿拉斯加狭鳕的出口量近些年一直是下滑的,这是因为美国阿拉斯加狭鳕的捕捞量不如俄罗斯高,而且有很大一部分在渔船上被加工成了鱼糜,然后再出口,而且美国政府鼓励国内消费本土产的鳕鱼。美国冷冻阿拉斯加狭鳕的主要出口国家为中国、韩国和乌克兰。<br><img><br>如上图所示,挪威冷冻黑线鳕的出口量这些年一直是下滑的,其主要的出口国家为中国、英国和立陶宛,中国是挪威黑线鳕最重要的海外加工基地。<br><img><br>上图为挪威冷冻蓝鳕的出口格局,挪威冷冻蓝鳕这两年的出口量增长的比较迅速,这根挪威在北大西洋水域蓝鳕的捕捞量不断增长有关。其主要的出口国家有中国和立陶宛。<br>大家看到这里可能注意到了我们在分析挪威鳕鱼出口情况时反复提到了立陶宛,由此可以推断出立陶宛逐渐成为了挪威鳕鱼在欧洲的加工基地,这些年的趋势是挪威逐渐把在中国加工的鳕鱼逐渐转到欧洲的立陶宛加工。<br><img><br>如上图所示,挪威干制真鳕的出口量近些年虽然不断的下降,但是每年的量仍然很大(4万吨左右),主要的出口国家为葡萄牙、巴西和意大利。其中居住着大批葡萄牙裔的巴西正逐渐成为干制鳕鱼的新兴市场。<br>这些年干制真鳕出口量不断下滑的主要原因有两个:第一,全球的大西洋真鳕的捕捞量这些年在缓慢的下降。第二,干鳕鱼这种历史悠久的传统鳕鱼制品不再为年轻人所喜,而选择鳕鱼鱼柳或冰鲜的鳕鱼。<br><img><br>挪威盐渍真鳕的出口情况跟干制真鳕的情况差不多,也是出口量不断的下滑,出口主要的市场为葡萄牙、西班牙和希腊。<br>挪威虽然每年都生产大量的干制和盐渍鳕鱼,但是其本土的消费量并不大,主要出口到南欧和葡萄牙语系国家这些干制和盐渍鳕鱼传统消费国家。<br>中国的鳕鱼原料进口<br>在前文中我们讲了全球主要鳕鱼捕捞国家的鳕鱼及鳕鱼制品的出口情况,下面让我们来说说全球最重要的鳕鱼加工中心:中国,加工鳕鱼的原料都从哪些国家来呢?<br><img><br>如上图所示,中国每年进口冷冻真鳕的数量非常庞大,在2016年和2017年高达20万吨,但是这两年出现了下滑。主要的进口国有俄罗斯、挪威和美国。其中俄罗斯和挪威主要提供的是在巴伦支海捕捞的大西洋真鳕,美国提供的是在西白令海捕捞的太平洋真鳕。<br><img><br>由上图我们可以更加清楚的,近几年俄罗斯向中国出口的冷冻真鳕的量不断上升,美国和挪威向中国出口的冷冻真鳕的量不断下降。<br><img><br>上图为中国冷冻阿拉斯加狭鳕的进口格局,毫无疑问,俄罗斯是中国冷冻阿拉斯加狭鳕最主要的供应国,美国和日本(图中红框部分)的量并不多。中国每年进口冷冻阿拉斯加狭鳕的量非常惊人,2019年高达70万吨左右。<br><img><br>如上图所示,中国进口的黑线鳕主要来自挪威和俄罗斯这两个国家,在巴伦支海捕捞的。<br><img><br>如上图所示,中国进口的绿青鳕只进口自一个国家,那就是挪威。因为挪威是全球最主要的绿青鳕捕捞国家,其他国家的捕捞量非常的少。<br><img><br>上图为中国无须鳕的进口格局,我们在前文中说过在全球有很多的国家捕捞无须鳕,所以中国进口很多国家的无须鳕,其中加拿大是最主要的供应国。了解这行的读者可能会问中国为什么不进口西班牙的无须鳕?这是因为西班牙捕捞的无须鳕很少加工成冷冻鱼柳,而是把鲜鱼出售给欧洲各国。<br><img><br>由上图可知,中国每年进口蓝鳕的量非常大,近些年平均每年60万吨左右,只比狭鳕略少一些,绝大多数都是来自于俄罗斯。<br>说到为什么俄罗斯每年都要把大批的蓝鳕和狭鳕出口到中国还有一个重要的原因,那就是这两种鳕鱼的经济价值相对来说比较低,直接在大型一体化渔船上加工不太划算,他们跟愿意把捕捞上来的这两种鳕鱼冷冻后送到中国进行加工。<br><img><br>上图为全球干制真鳕的进口格局,主要进口国家有瑞典、葡萄牙、德国、西班牙和意大利。这里面要指出来的是瑞典并不是一个干制鳕鱼的消费市场而是一个加工中心,主要的消费国家差不多都在南欧。<br><img><br>全球盐渍真鳕的进口格局跟干制真鳕的情况差不多,主要的进口国家有瑞典、巴西、德国、西班牙和意大利。其中瑞典是加工中心。巴西是新兴市场。<br>中国及其他国家鳕鱼加工及出口<br>下面让我们讲一下鳕鱼的加工和出口,读过前文的读者都知道中国是全球最重要的鳕鱼加工中心, 但是在鳕鱼加工这个行业中除了中国还有很多其他的国家,下面就让老樊把他们放到一块给大家一起介绍一下。<br>首先让我们看一下中国进口了这么的鳕鱼原料加工后都出口了那些国家?<br><img><br>上图为中国岸冻真鳕鱼鱼柳的出口格局,中国岸冻真鳕鱼鱼柳主要出口美国、英国、德国、西班牙、法国和加拿大,基本上都是欧美国家。(前面说过欧美国家是传统的鳕鱼消费市场),出口量这些年相对保持平稳(平均每年12万吨左右)。<br><img><br>图所示,中国近些年开始出口一种新的鳕鱼加工品,那就是盐渍鳕鱼,特别是从2016年以后,中国每年出口盐渍鳕鱼的量就一直处于快速上升状态。主要出口到葡萄牙和巴西。<br><img><br>俄罗斯也是鳕鱼加工的重要国家之一,他不光把原料卖给中国进行加工,自己也有很多岸上的加工厂加工鳕鱼鱼柳出口。俄罗斯近些年新造的大型一体化渔船,俄罗斯用这些渔船加工真鳕。我们可以从上图中看到,俄罗斯在高峰时期每年出口超过3万吨的冷冻真鳕鱼柳,主要出口到荷兰(图中黄色部分,占绝大多数)。<br><img><br>冰岛也是一个非常重要的鳕鱼加工国家,是我们前文中讲过的四大重要的鳕鱼捕捞国家之一(俄罗斯、美国、挪威和冰岛)。冰岛捕捞的鳕鱼很多都是在自己的大型一体化渔船上加工的,另外冰岛也有一些岸上的鳕鱼加工厂,近些年发展的也很快。这些年冰岛每年出口的冷冻真鳕鱼柳平均每年4万吨左右,最主要的市场是英国、西班牙和荷兰。<br><img><br>上图为冰岛岸冻真鳕鱼柳的出口格局,其中左图为冷冻包冰鱼块的出口情况,鱼块的出口量并不是非常的大(近些年平均每年3千吨左右),而且近两年出口量出现了下滑,主要消费市场是在英国。<br>右图为鳕鱼鱼片的出口情况,鱼片的出口量比较大(近些年平均每年1万5千多吨),这两年也出现了下滑,主要的出口国家是西班牙、荷兰和英国。<br><img><br>近些年在大型渔船上直接进行加工的一次冷冻的鳕鱼制品成了主要的消费趋势,如上图所示,同样是冰岛加工的鳕鱼鱼块和鱼片,船冻的鳕鱼制品这些年的出口量处于增长状态。这就说明了欧洲市场的消费者更愿意购买一次冷冻的鳕鱼制品。<br><img><br>如上图所示,中国岸上加工的狭鳕鱼鱼柳主要出口到德国、美国和英国。其中德国几乎占了一半,这是因为德国是狭鳕鱼重要的消费市场。<br><img><br>前文中我们提到过很多美国捕捞的狭鳕鱼并没有被做成鱼柳,而是被加工成了鱼糜。所以如上图所示,美国的冷冻狭鳕鱼鱼糜出口量非常的大,近些年每年都出口15到20万吨左右的鱼糜,主要出口到日本和韩国。<br><img><br>上图为中国岸冻黑线鳕鱼柳的出口格局,这两年的出口量有些缓慢的下滑,平均的出口量为2万5千吨左右,主要的出口国家为美国、加拿大和英国。<br><img><br>上图为中国岸冻绿青鳕鱼柳的出口格局,这些年处于稳步增长状态,但是出口的量不多,这些年平均每年不到1万吨,出口的主要市场有:瑞典、法国、德国和日本。<br><img><br>在前文中我们讲过无须鳕的捕捞国家很多,所以向中国提供原料的国家也很多,所以如上图所示,中国向很多国家出口岸冻无须鳕鱼柳,这些年的出口量有些下滑,平均的出口量为1万4千吨左右,主要的出口国为巴西、美国和日本。<br>鳕鱼鱼糜市场:咱们身边的故事<br>下面老樊来说说前文中提到过很多次的鳕鱼鱼糜,鱼糜主要用狭鳕鱼制作而且消费市场离我们很近。<br><img><br>上图为阿拉斯加狭鳕鱼糜成品的照片,它看起来像一块砖头,它在鳕鱼捕捞后直接在船上进行加工,主要在日韩进行消费,被做成我们非常熟悉的蟹棒和拟蟹钳等产品。<br><img><br>上图为美国阿拉斯加狭鳕鱼糜的出口情况,近些年平均每年都要出口15到20万吨左右,这个量非常高,主要的市场有两个:日本和韩国。这两个我们的邻国每年都消费大量的鳕鱼鱼糜制品。<br><img><br>除了阿拉斯加狭鳕鱼之外,一部分的太平洋真鳕也被加工成鱼糜。如上图所示,其主要市场也是日本和韩国。不过和阿拉斯加狭鳕鱼糜的出口量相比,太平洋真鳕鱼糜的出口量比较少,近些年平均每年只有1万吨左右,这两年还出现了下滑。这说明美国将大量的太平洋真鳕做成鱼柳在本土消费,做鱼糜出口的量越来越少。<br>冰鲜鳕鱼市场:遥远世界的故事<br>最后让我们来说一下冰鲜鳕鱼,说起冰鲜鳕鱼,大家都很陌生,因为中国人常吃的鱼并不包括冰鲜鳕鱼。<br><img><br>上面的两张照片是老樊去挪威卑尔根在当地海鲜市场拍摄的冰鲜大西洋真鳕的照片。其中左边的是切成段的冰鲜大西洋真鳕,右边的是整条的冰鲜大西洋真鳕。在挪威鳕鱼(cod)有个特殊的名字叫做 “Skrel” ,这种冰鲜的鳕鱼在欧洲非常受欢迎,只是我们中国人不太了解。<br><img><br>上图为冰鲜的大西洋真鳕的出口情况,其主要的出口国家为北欧三国:挪威、丹麦和瑞典。(其中丹麦和瑞典是负责加工鳕鱼的)<br><img><br>由以上两张图我们可以看出这两年挪威的冰鲜大西洋真鳕处于增长状态,但是在2019年出现了一个比较明显的下滑。丹麦的出口量也出现了下滑。老樊个人估计这是一个市场的信号,说明从2019年开始,整个的大西洋真鳕的供应量开始不足了。还有一点是老樊在前文中提到的随着市场的走俏,运来越多的大西洋真鳕被加工成了较为廉价和易储存的一次冷冻的鳕鱼产品。<br><img><br>上图为冰鲜黑线鳕的出口情况,其主要的出口国家有挪威、丹麦、冰岛和瑞典,其中丹麦和瑞典负责加工。从出口量上来说,挪威是最主要的国家。<br><img><br>从上图我们可以看出近些年冰鲜黑线鳕的出口量并不大,平均每年都在1万5千吨到2万吨之间,而且出口量逐年减少。主要消费的国家有英国和丹麦。<br><img><br>如上图所示,冰鲜绿青鳕的出口国家主要有三个国家:丹麦、挪威和德国,从量上来说挪威是最多的,丹麦和德国也不少。<br><img><br>上图左图为丹麦的冰鲜绿青鳕出口趋势,右图为挪威的冰鲜绿青鳕出口趋势。从图中我们可以看出这些年丹麦的出口量增长的比较迅速,平均每年出口1万5千多吨。挪威近些年的出口量增长的也比较迅速,平均每年出口1万5千多吨。只不过他俩出口的国家不一样,挪威主要向丹麦出口鳕鱼,丹麦再卖到其它欧洲国家。而丹麦主要向荷兰、波兰和法国等国家出口。<br><img><br>如上图所示,有很多的国家出口冰鲜的无须鳕,在这些国家中西班牙的出口量遥遥领先。<br><img><br>上图为冰鲜无须鳕的出口情况,其主要出口到葡萄牙、阿尔及利亚、法国、意大利和德国,向这些国家的出口量比较平均,没有哪个国家占主导的地位。<br><img><br>由上图可知,冰鲜阿拉斯加狭鳕的出口量和冷冻的相比并不多,美俄这些阿拉斯加狭鳕捕捞大国并没有冰鲜阿拉斯加狭鳕出口,目前主要是日本在向韩国出口,但是量不大,2019年只有2千吨左右。(韩国是在东亚地区非常喜欢吃狭鳕鱼的国家,并称其为明太鱼)<br>全球鳕鱼消费趋势<br>可持续消费:消费者越来越认可获得可持续认证的鳕鱼产品。<br>船冻更美味:消费升级,更多消费者认可船冻鳕鱼产品。<br>冰鲜有市场:冰鲜鳕鱼消费,从欧洲市场向全球扩展。<br>鱼糜有市场:鱼糜制品在日韩流行,可能会在中国流行起来。<br>中国可期待:鳕鱼消费在中国国内可能会逐渐兴起。<br>原创声明<br>本文为八鲜过海原创文章。未经授权,请勿转载。"


def create_test_texts(count: int = 1024, context_length: int = 512) -> List[str]:
    texts = []
    for i in range(count):
        s = str(random.randint(1, 10000))
        texts.append((s + text)[:context_length])

    return texts


def create_test_rerank_inputs(count: int = 1024, context_length: int = 512, batch_size: int = 3) -> List[
    Dict[str, str]]:
    texts = []
    curls = int(count/batch_size)
    for i in range(curls):
        docs = []
        for j in range(batch_size):
            s = str(random.randint(1, 10000))
            s = (s + text)[:context_length]
            docs.append(s)

        texts.append({"query": "中国进口鳕鱼主要那些国家", "documents": docs})
    return texts


def main(url: str, total_count_text: int, context_length: int, batch_size: int, num_thread: int, test_type: str):
    """主函数示例"""
    # 配置参数
    # 选择测试类型: "embedding" 或 "score"
    # test_type = "embedding"  # 可以改为 "embedding" 来测试embedding接口

    print(f"开始 {test_type} 接口压测...")

    if test_type == "embedding":
        # 创建embedding压测实例
        benchmark = Benchmark(
            url=url,
            batch_size=batch_size,
            test_type="embedding",
        )

        # 准备测试数据
        test_inputs = create_test_texts(total_count_text, context_length)  # 生成100个测试文本


    elif test_type == "rerank":
        # 创建score压测实例
        benchmark = Benchmark(
            url=url,
            batch_size=batch_size,
            test_type="rerank"
        )

        # 准备测试数据
        test_inputs = create_test_rerank_inputs(total_count_text, context_length, batch_size)  # 生成100个score测试输入

    else:
        raise ValueError(f"不支持的测试类型: {test_type}")

    # 执行压测
    results = benchmark.benchmark(
        inputs=test_inputs,
        num_threads=num_thread,  # 使用5个线程
        requests_per_thread=None  # 平均分配所有输入
    )

    # 打印结果
    benchmark.print_results(results)
    return results


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="llm run parameters")
    # parser.add_argument('--test_type', type=str, default="embedding", help="embedding or reranker")
    # parser.add_argument('--url', type=str, default="http://51.38.68.109:9095/v1/embeddings", help="url")
    # parser.add_argument('--batch_size', type=int, default=1, help="per request batch size")
    # parser.add_argument('--total_count_text', type=int, default=1024, help="request total num of texts")
    # parser.add_argument('--num_thread', type=int, default=4, help="number of thread")
    # parser.add_argument('--context_length', type=int, default=512, help="请求文本长度")
    # parser_args = parser.parse_args()
    # main(parser_args.url, parser_args.total_count_text, parser_args.context_length,
    #      parser_args.batch_size, parser_args.num_thread)

    batch_sizes = [20]
    num_threads = [1, 4, 8, 20, 40]
    total_count_text = 1024
    context_lengths = [1024]

    wb = Workbook()

    ws = wb.create_sheet(f"test", 0)
    headers = [
        'num_thread',
        'total_time (s)',
        "context_length",
        'total_requests',
        'total_batches',
        'success_batches',
        'batch_size',
        'success_count',
        'error_count',
        'success_rate',
        'effective_qps',
        'attempted_qps',
        'effective_batch_qps',
        'attempted_batch_qps',
        'avg_success_response_time (s)',
        'median_success_response_time (s)',
        'p95_success_response_time (s)',
        'min_success_response_time (s)',
        'max_success_response_time (s)',
        'avg_response_time (s)',
        'median_response_time (s)',
        'p95_response_time (s)',
        'min_response_time (s)',
        'max_response_time (s)',
    ]

    ws.append(headers)  # 写入表头
    #url = "http://10.169.68.47:4000/v1/rerank"
    url = "http://10.44.124.218:2662/v1/embeddings"
    for context_length in context_lengths:
        for batch_size in batch_sizes:
            for num_thread in num_threads:
                result = main(url, total_count_text, context_length, batch_size, num_thread, "embedding")
                row = [
                    result['num_thread'],
                    result['total_time'],
                    context_length,
                    result['total_requests'],
                    result['total_batches'],
                    result['success_batches'],
                    result['batch_size'],
                    result['success_count'],
                    result['error_count'],
                    result['success_rate'],
                    result['qps'],
                    result['attempted_qps'],
                    result['batch_qps'],
                    result['attempted_batch_qps'],
                    result['avg_success_response_time'],
                    result['median_success_response_time'],
                    result['p95_success_response_time'],
                    result['min_success_response_time'],
                    result['max_success_response_time'],
                    result['avg_response_time'],
                    result['median_response_time'],
                    result['p95_response_time'],
                    result['min_response_time'],
                    result['max_response_time'],
                ]
                ws.append(row)  # 写入一行数据

    wb.save("Qwen3-Embedding-0.6B-3-19-10-56.xlsx")

