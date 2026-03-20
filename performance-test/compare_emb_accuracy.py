#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化服务比较工具
发送文本列表到两个向量化服务，批量获取向量结果并计算余弦距离
"""
from os import path

import requests
import numpy as np
import pandas as pd

import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='"%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorComparisonTool:
    """向量化服务比较工具类"""

    def __init__(self, service1_url: str, service2_url: str, timeout: int = 10, batch_size: int = 1):
        """
        初始化工具

        Args:
            service1_url: 第一个向量化服务的URL
            service2_url: 第二个向量化服务的URL
            timeout: 请求超时时间（秒）
            batch_size: 批量请求大小（默认10）
        """
        self.service1_url = service1_url
        self.service2_url = service2_url
        self.timeout = timeout
        self.batch_size = batch_size
        self.session = requests.Session()  # 复用会话，提高性能

    def get_vectors(self, texts: List[str], service_url: str) -> Optional[List[Tuple[str, np.ndarray]]]:
        """
        批量发送请求到向量化服务获取向量

        Args:
            texts: 输入文本列表
            service_url: 服务URL

        Returns:
            (文本, 向量) 元组列表，失败时返回None
        """
        try:
            # 准备请求数据（支持批量输入的API格式）
            payload = {
                "input": texts
            }

            response = self.session.post(
                service_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )

            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"服务 {service_url} 返回错误状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return None

            # 解析响应（批量结果）
            result = response.json()
            if "data" not in result or len(result["data"]) != len(texts):
                logger.error(f"服务 {service_url} 返回无效响应格式，结果数量不匹配")
                logger.error(f"请求文本数: {len(texts)}, 响应结果数: {len(result['data']) if 'data' in result else 0}")
                return None

            # 构造 (文本, 向量) 元组列表
            vectors = []
            for i, item in enumerate(result["data"]):
                if "embedding" not in item:
                    logger.warning(f"文本索引 {i} 缺少 embedding 字段，跳过")
                    continue
                vector = np.array(item["embedding"], dtype=np.float32)
                vectors.append((texts[i], vector))

            logger.info(f"从服务 {service_url} 批量获取向量成功，有效结果数: {len(vectors)}/{len(texts)}")
            return vectors

        except requests.exceptions.Timeout:
            logger.error(f"请求服务 {service_url} 超时")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到服务 {service_url}")
            return None
        except json.JSONDecodeError:
            logger.error(f"服务 {service_url} 返回无效的JSON格式")
            return None
        except Exception as e:
            logger.error(f"请求服务 {service_url} 时发生错误: {str(e)}")
            return None

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> Optional[float]:
        """
        计算两个向量的余弦相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量

        Returns:
            余弦相似度（范围[-1, 1]），失败时返回None
        """
        try:
            # 检查向量维度是否匹配
            if vec1.shape != vec2.shape:
                logger.error(f"向量维度不匹配: {vec1.shape} vs {vec2.shape}")
                return None

            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            # 避免除以零
            if norm1 == 0 or norm2 == 0:
                logger.error("向量范数为零，无法计算余弦相似度")
                return None

            similarity = dot_product / (norm1 * norm2)
            return similarity

        except Exception as e:
            logger.error(f"计算余弦相似度时发生错误: {str(e)}")
            return None

    def compare_services(self, texts: List[str]) -> List[Dict[str, Union[str, np.ndarray, float, None]]]:
        """
        比较两个向量化服务的结果（支持批量文本）

        Args:
            texts: 输入文本列表

        Returns:
            比较结果列表，每个元素包含文本、两个服务的向量和余弦相似度
        """
        logger.info(f"开始批量比较，共 {len(texts)} 个文本，批量大小: {self.batch_size}")

        all_results = []

        # 分批次处理文本
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.info(
                f"处理批次 {i // self.batch_size + 1}/{(len(texts) + self.batch_size - 1) // self.batch_size}，文本数: {len(batch_texts)}")

            # 获取两个服务的批量向量
            vectors1 = self.get_vectors(batch_texts, self.service1_url)
            vectors2 = self.get_vectors(batch_texts, self.service2_url)

            if vectors1 is None or vectors2 is None:
                logger.error(f"批次 {i // self.batch_size + 1} 获取向量失败，跳过")
                # 为该批次的文本添加失败结果
                for text in batch_texts:
                    all_results.append({
                        "text": text,
                        "vector1": None,
                        "vector2": None,
                        "similarity": None,
                        "status": "failed"
                    })
                continue

            # 将向量结果转换为字典，方便按文本匹配
            vec_dict1 = {text: vec for text, vec in vectors1}
            vec_dict2 = {text: vec for text, vec in vectors2}

            # 逐个文本比较
            for text in batch_texts:
                result = {"text": text, "status": "success"}

                # 获取当前文本的两个向量
                vec1 = vec_dict1.get(text)
                vec2 = vec_dict2.get(text)

                if vec1 is None or vec2 is None:
                    result["status"] = "partial_failed"
                    result["vector1"] = vec1
                    result["vector2"] = vec2
                    result["similarity"] = None
                    logger.warning(f"文本 '{text[:30]}...' 向量获取不完整")
                else:
                    # 计算余弦相似度
                    similarity = self.calculate_cosine_similarity(vec1, vec2)
                    result["vector1"] = vec1
                    result["vector2"] = vec2
                    result["similarity"] = similarity

                all_results.append(result)

        logger.info(f"批量比较完成，共处理 {len(all_results)} 个文本")
        return all_results

def read_file(file_path):

    if not path.exists(file_path):
        logger.error(f"-------------------{file_path}  not exist")
        return
    df_corpus = pd.read_parquet(file_path)[:512].text.tolist()
    t = []
    for cor in df_corpus:
        t.append(cor[:1024])
    return t

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='向量化服务比较工具（支持批量文本）')
    parser.add_argument('--service1', type=str, default="http://10.44.124.218:51515/v1/embeddings", help='第一个向量化服务的URL')
    parser.add_argument('--service2', type=str, default="http://10.44.124.218:2662/v1/embeddings", help='第二个向量化服务的URL')
    # parser.add_argument('--texts', type=str, nargs='+', required=True, help='要向量化的文本列表（多个文本用空格分隔）')
    parser.add_argument('--timeout', type=int, default=10, help='请求超时时间（秒）')
    parser.add_argument('--batch-size', type=int, default=5, help='批量请求大小（默认10）')

    # https://www.modelscope.cn/datasets/C-MTEB/T2Retrieval/files
    texts = read_file("C://Users//l50056623//Desktop//tei_performance_test//dataset//corpus-00000-of-00001-8afe7b7a7eca49e3.parquet")
    # 更改成个人主机上的路径
    args = parser.parse_args()

    # 创建工具实例
    tool = VectorComparisonTool(
        service1_url=args.service1,
        service2_url=args.service2,
        timeout=args.timeout,
        batch_size=args.batch_size
    )

    # 执行批量比较
    results = tool.compare_services(texts)

    # 输出汇总结果
    print("\n" + "=" * 70)
    print("向量化服务批量比较结果汇总")
    print("=" * 70)
    print(f"服务1 URL: {args.service1}")
    print(f"服务2 URL: {args.service2}")
    print(f"总文本数: {len(results)}")
    print(f"批量大小: {args.batch_size}")

    # 统计各状态数量
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"成功比较: {status_counts.get('success', 0)}")
    print(f"部分失败: {status_counts.get('partial_failed', 0)}")
    print(f"完全失败: {status_counts.get('failed', 0)}")

    # 如果有成功结果，计算平均相似度
    similarities = [r["similarity"] for r in results if r["status"] == "success" and r["similarity"] is not None]
    if similarities:
        avg_similarity = np.mean(similarities)
        print(f"平均余弦相似度: {avg_similarity:.6f}")
        print(f"平均相似度百分比: {avg_similarity * 100:.2f}%")

    print("=" * 70)

    # 输出详细结果
    print("\n详细比较结果:")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        text = result["text"]
        text_display = text[:100] + "..." if len(text) > 100 else text

        # print(f"\n文本 {i}: {text_display}")
        # print(f"状态: {result['status']}")

        # if result["vector1"] is not None:
        #     print(f"服务1向量维度: {result['vector1'].shape}")
        #     print(f"服务1向量前3个元素: {result['vector1'][:3]}")
        #
        # if result["vector2"] is not None:
        #     print(f"服务2向量维度: {result['vector2'].shape}")
        #     print(f"服务2向量前3个元素: {result['vector2'][:3]}")

        if result["similarity"] is not None:
            similarity = result["similarity"]
            # print(f"余弦相似度: {similarity:.6f} ({similarity * 100:.2f}%)")

            # 相似度解释
            if similarity < 0.99:
                print("  结论: 差异较大")
            # if similarity > 0.99:
            #     print("  结论: 高度相似")
            # elif similarity > 0.7:
            #     print("  结论: 较为相似")
            # elif similarity > 0.5:
            #     print("  结论: 存在一定相似性")
            # else:
            #     print("  结论: 差异较大")

        # print("-" * 70)


if __name__ == "__main__":
    main()