"""
图像向量化模块

使用阿里云百炼多模态 Embedding 模型将图片转为向量。
与 text_embedding.py 共用同一 API Key、同一端点、同一缓存方案。

【与架构文档的对应关系】
- 位置：embeddings/image_embedding.py
- 模型：百炼 multimodal-embedding-v1（多模态，同时理解文本和图片）
- 下游：multimodal_embedding.py、tools/knowledge_retrieval_tool.py

【和 text_embedding.py 的关系】
- 相同的 API 端点（/compatible-mode/v1/embeddings）
- 相同的认证方式（Bearer token，共用 DASHSCOPE_API_KEY）
- 相同的缓存策略（Redis，MD5 key）
- 不同点：input 格式从纯文本字符串变为 {"image": "url"} 对象

【API 格式】
请求：
{
    "model": "multimodal-embedding-v1",
    "input": [
        {"image": "https://example.com/engine.jpg"},
        {"image": "https://example.com/bearing.jpg"}
    ]
}
响应：与 text-embedding-v4 一致
{
    "data": [
        {"index": 0, "embedding": [0.12, -0.34, ...]},
        {"index": 1, "embedding": [0.08, -0.21, ...]}
    ]
}
"""

import hashlib
import redis
import httpx
from typing import Optional, List

from config.settings import get_settings


class ImageEmbedding:
    """
    图像向量化服务

    封装阿里云百炼多模态 Embedding API，将图片 URL 转为特征向量。
    带 Redis 缓存，避免重复调用。
    """

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.dashscope_api_key
        self.model = "multimodal-embedding-v1"
        self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = httpx.AsyncClient(
            timeout=60.0,  # 图片处理比纯文本慢
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        self.redis = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            db=self.settings.redis_db,
            decode_responses=False
        )
        self.cache_ttl = self.settings.redis_ttl

    def _get_cache_key(self, image_url: str) -> str:
        """用图片URL的MD5作为缓存key，与 text_embedding 的缓存隔离"""
        return f"img_emb:v1:{hashlib.md5(image_url.encode()).hexdigest()}"

    def _get_from_cache(self, image_url: str) -> Optional[List[float]]:
        key = self._get_cache_key(image_url)
        data = self.redis.get(key)
        if data:
            import pickle
            return pickle.loads(data)
        return None

    def _set_to_cache(self, image_url: str, embedding: List[float]) -> None:
        key = self._get_cache_key(image_url)
        import pickle
        self.redis.setex(key, self.cache_ttl, pickle.dumps(embedding))

    async def embed(self, image_url: str) -> List[float]:
        """
        单张图片向量化

        Args:
            image_url: 图片 URL（百炼API会自行下载）

        Returns:
            向量列表（维度取决于模型，通常1024或1536）
        """
        cached = self._get_from_cache(image_url)
        if cached is not None:
            return cached

        result = await self._call_api([image_url])
        embedding = result[0]
        self._set_to_cache(image_url, embedding)
        return embedding

    async def embed_batch(self, image_urls: List[str]) -> List[List[float]]:
        """
        批量图片向量化

        先查缓存，命中的直接用；未命中的批量调 API，然后写回缓存。

        Args:
            image_urls: 图片 URL 列表

        Returns:
            向量列表，与输入顺序一致
        """
        results: List[Optional[List[float]]] = []
        uncached_indices: List[int] = []
        uncached_urls: List[str] = []

        for i, url in enumerate(image_urls):
            cached = self._get_from_cache(url)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_urls.append(url)

        if uncached_urls:
            new_embeddings = await self._call_api(uncached_urls)
            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                self._set_to_cache(image_urls[idx], emb)

        return results

    async def _call_api(self, image_urls: List[str]) -> List[List[float]]:
        """
        调用百炼多模态 Embedding API

        Args:
            image_urls: 图片 URL 列表

        Returns:
            向量列表，与输入顺序一致

        Raises:
            httpx.HTTPStatusError: API 调用失败
            ValueError: 响应格式异常
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        params = {
            "model": self.model,
            "input": [{"image": url} for url in image_urls]
        }

        response = await self.client.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=params
        )
        response.raise_for_status()
        result = response.json()

        if result.get("data"):
            dim = len(result["data"][0]["embedding"])
            print(f"[DEBUG] Image Embedding Model: {self.model}, Dimension: {dim}")

        if "data" in result:
            embeddings = [
                item["embedding"]
                for item in sorted(result["data"], key=lambda x: x["index"])
            ]
            return embeddings

        raise ValueError(f"Unexpected API response: {result}")


# 单例
_image_embedding: Optional[ImageEmbedding] = None


def get_image_embedding() -> ImageEmbedding:
    """获取图像向量化服务单例"""
    global _image_embedding
    if _image_embedding is None:
        _image_embedding = ImageEmbedding()
    return _image_embedding
