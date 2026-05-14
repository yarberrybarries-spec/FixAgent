"""
知识入库服务

编排 文档解析 → 向量化 → Redis 向量库 的完整流程。
只做编排，不自己解析、不自己向量化、不自己写 Redis。

【执行流程】
1. DocumentParserTool 解析 PDF → sections
2. text_chunks → TextEmbedding.embed_batch() → VectorService.add_vector_batch()
3. tables → 转 markdown 文本 → TextEmbedding → VectorService
4. images → 用图注文本向量化 → VectorService（metadata 存本地路径，后续可升级为 ImageEmbedding）
5. 返回导入统计
"""

import time
import hashlib
import logging
from typing import List, Optional

from tools.document_tool import get_document_parser
from embeddings.text_embedding import get_text_embedding
from services.vector_service import get_vector_service

logger = logging.getLogger(__name__)


class KnowledgeService:
    """知识入库服务"""

    # embed_batch 单批最大条数（百炼 API 限制）
    _BATCH_SIZE = 25

    def __init__(self):
        self.parser = get_document_parser()
        self.text_emb = get_text_embedding()
        self.vector_svc = get_vector_service()

    async def import_document(
        self,
        file_url: str,
        file_type: str = "pdf",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> dict:
        """
        导入文档：解析 → 向量化 → 入库

        Returns:
            {
                "file_name": str,
                "total_pages": int,
                "text_count": int,       # 入库文本块数
                "image_count": int,      # 入库图片数
                "table_count": int,      # 入库表格数
                "sections": [...],       # 各章节统计摘要
                "extraction_summary": {...},
                "process_time_ms": int
            }
        """
        t0 = time.time()

        # 1. 解析文档
        parse_result = await self.parser._execute(file_url, file_type)
        file_name = parse_result["file_name"]
        total_pages = parse_result["total_pages"]
        sections = parse_result["sections"]
        extraction_summary = parse_result["extraction_summary"]

        doc_prefix = hashlib.md5(file_name.encode()).hexdigest()[:8]

        text_count = 0
        image_count = 0
        table_count = 0

        # 2. 逐 section 处理
        for sec_idx, section in enumerate(sections):
            section_title = section.get("section_title", f"第{sec_idx + 1}章")
            page_range = section.get("page_range", "")
            sec_category = category or section_title

            # 2a. 文本块 → 分批 embed_batch → 入库
            raw_chunks = section.get("text_chunks", [])
            valid_chunks = [t for t in raw_chunks if len(t.strip()) >= 10]
            for batch_start in range(0, len(valid_chunks), self._BATCH_SIZE):
                batch = valid_chunks[batch_start:batch_start + self._BATCH_SIZE]
                vectors = await self.text_emb.embed_batch(batch)
                docs = []
                for i, (chunk, vec) in enumerate(zip(batch, vectors)):
                    global_i = batch_start + i
                    chunk_id = f"{doc_prefix}:{sec_idx:02d}:txt:{global_i:04d}"
                    docs.append({
                        "doc_id": chunk_id,
                        "text": chunk,
                        "vector": vec,
                        "category": sec_category,
                        "tags": tags,
                        "metadata": {
                            "file_name": file_name,
                            "section_title": section_title,
                            "page_range": page_range,
                            "chunk_type": "text"
                        }
                    })
                self.vector_svc.add_vector_batch(docs)
                text_count += len(docs)

            # 2b. 表格 → 转文本 → 入库
            for t_idx, table in enumerate(section.get("tables", [])):
                table_text = self._table_to_text(table)
                if not table_text.strip():
                    continue
                vec = await self.text_emb.embed(table_text)
                table_id = f"{doc_prefix}:{sec_idx:02d}:tbl:{t_idx:04d}"
                self.vector_svc.add_vector(
                    doc_id=table_id,
                    text=table_text,
                    vector=vec,
                    category=sec_category,
                    tags=tags,
                    metadata={
                        "file_name": file_name,
                        "section_title": section_title,
                        "page_range": page_range,
                        "chunk_type": "table",
                        "page": table.get("page"),
                        "caption": table.get("caption", "")
                    }
                )
                table_count += 1

            # 2c. 图片 → 图注文本向量化 → 入库
            for img_idx, img in enumerate(section.get("images", [])):
                caption = img.get("caption", "").strip()
                img_name = img.get("image_name", f"img_{img_idx}")
                local_path = img.get("local_path", "")

                img_text = caption if caption else f"{section_title} 第{img.get('page', '?')}页插图"
                vec = await self.text_emb.embed(img_text)
                img_id = f"{doc_prefix}:{sec_idx:02d}:img:{img_idx:04d}"
                self.vector_svc.add_vector(
                    doc_id=img_id,
                    text=img_text,
                    vector=vec,
                    category=sec_category,
                    tags=tags,
                    metadata={
                        "file_name": file_name,
                        "section_title": section_title,
                        "page_range": page_range,
                        "chunk_type": "image",
                        "page": img.get("page"),
                        "image_name": img_name,
                        "local_path": local_path,
                        "caption": caption
                    }
                )
                image_count += 1

        t1 = time.time()

        return {
            "file_name": file_name,
            "total_pages": total_pages,
            "text_count": text_count,
            "image_count": image_count,
            "table_count": table_count,
            "sections": [
                {
                    "section_title": s.get("section_title", ""),
                    "page_range": s.get("page_range", ""),
                    "text_chunks": len(s.get("text_chunks", [])),
                    "images": len(s.get("images", [])),
                    "tables": len(s.get("tables", []))
                }
                for s in sections
            ],
            "extraction_summary": extraction_summary,
            "process_time_ms": int((t1 - t0) * 1000)
        }

    @staticmethod
    def _table_to_text(table: dict) -> str:
        """将表格 dict 转为可向量化的 markdown 文本"""
        rows = table.get("rows", [])
        if not rows:
            return ""

        lines = []
        caption = table.get("caption", "")
        if caption:
            lines.append(f"表格：{caption}")

        for row in rows:
            if row and any(cell for cell in row):
                lines.append(" | ".join(str(cell).strip() for cell in row))

        return "\n".join(lines)


# 单例
_knowledge_service: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    """获取知识入库服务单例"""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service
