"""
图谱查询工具

封装 Neo4j 知识图谱诊断路径查询和设备搜索。
查询逻辑与 Java GraphQueryServiceImpl 对齐。

【调用链】
用户描述 → TextEmbedding（向量化） → Neo4j 向量索引匹配部件/故障
         → 设备名模糊匹配 Device
         → 5分支 Cypher 查诊断路径
         → 返回路径列表（含向量分数、历史记录标记、可读路径文本）
"""

from typing import List, Optional, Dict, Any
import logging

from tools.base_tool import BaseTool, ToolException
from services.graph_service import get_graph_service
from embeddings.text_embedding import get_text_embedding

logger = logging.getLogger(__name__)


class GraphQueryTool(BaseTool):
    """
    图谱诊断路径查询工具

    查询策略（与 Java 一致）：
    1. keyword → 在 Device 节点中模糊匹配设备
    2. component_description → embedding → Neo4j 向量索引查匹配部件
    3. fault_description → embedding → Neo4j 向量索引查匹配故障
    4. 根据有哪些信息，自动走5分支Cypher查询
    """

    @property
    def name(self) -> str:
        return "graph_query_diagnosis_path"

    @property
    def description(self) -> str:
        return (
            "从设备检修知识图谱中查询诊断路径（设备→部件→故障→解决方案）。"
            "支持设备名称关键字模糊匹配，部件描述和故障描述的向量语义匹配。"
            "返回每条路径的向量匹配分数、历史故障标记、可读路径文本。"
            "适用场景：分析设备故障的因果关系、查找已知解决方案、确认故障-方案对应关系。"
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "设备名称关键字，用于模糊匹配 Device 节点（可选）"
                },
                "component_description": {
                    "type": "string",
                    "description": "部件描述，用于向量语义匹配 Component 节点（可选）"
                },
                "fault_description": {
                    "type": "string",
                    "description": "故障描述，用于向量语义匹配 Fault 节点（可选）"
                },
                "page": {
                    "type": "integer",
                    "description": "页码，从0开始，默认0",
                    "default": 0
                },
                "size": {
                    "type": "integer",
                    "description": "每页数量，默认5",
                    "default": 5
                }
            },
            "required": []
        }

    async def _execute(
        self,
        keyword: str = None,
        component_description: str = None,
        fault_description: str = None,
        page: int = 0,
        size: int = 5
    ) -> List[Dict[str, Any]]:
        graph = get_graph_service()
        emb_svc = get_text_embedding()

        # 1. 部件向量匹配
        component_ids = []
        component_score_map: Dict[str, float] = {}
        if component_description and component_description.strip():
            try:
                comp_emb = await emb_svc.embed(component_description)
                comp_results = graph.search_components_by_embedding(comp_emb, 20, 0.50)
                component_ids = [r["id"] for r in comp_results]
                for r in comp_results:
                    cid = r["id"]
                    component_score_map[cid] = max(
                        component_score_map.get(cid, 0.0), r.get("score", 0.0)
                    )
            except Exception as e:
                raise ToolException(
                    code="EMBEDDING_FAILED",
                    message=f"部件向量匹配失败: {e}"
                )

        # 2. 故障向量匹配
        fault_ids = []
        fault_score_map: Dict[str, float] = {}
        if fault_description and fault_description.strip():
            try:
                fault_emb = await emb_svc.embed(fault_description)
                fault_results = graph.search_faults_by_embedding(fault_emb, 20, 0.80)
                fault_ids = [r["id"] for r in fault_results]
                for r in fault_results:
                    fid = r["id"]
                    fault_score_map[fid] = max(
                        fault_score_map.get(fid, 0.0), r.get("score", 0.0)
                    )
            except Exception as e:
                raise ToolException(
                    code="EMBEDDING_FAILED",
                    message=f"故障向量匹配失败: {e}"
                )

        if not component_ids and not fault_ids:
            raise ToolException(
                code="NO_MATCH",
                message="未能从部件描述或故障描述中匹配到图谱节点，请提供更具体的信息。"
            )

        # 3. 5分支查询
        try:
            result = graph.find_diagnosis_paths(
                keyword=keyword,
                component_ids=component_ids,
                fault_ids=fault_ids,
                component_score_map=component_score_map,
                fault_score_map=fault_score_map,
                page=page,
                size=size
            )
        except Exception as e:
            raise ToolException(
                code="GRAPH_QUERY_FAILED",
                message=f"图谱查询失败: {e}"
            )

        records = result.get("records", [])
        if not records:
            return []

        return [{
            "device_id": r.get("device_id"),
            "device_name": r.get("device_name", ""),
            "component_id": r.get("component_id"),
            "component_name": r.get("component_name", ""),
            "fault_id": r.get("fault_id"),
            "fault_name": r.get("fault_name", ""),
            "fault_severity": r.get("fault_severity", ""),
            "solution_id": r.get("solution_id"),
            "solution_title": r.get("solution_title", ""),
            "estimated_time": r.get("estimated_time"),
            "verified": r.get("verified"),
            "has_history": r.get("has_history", False),
            "fault_score": r.get("fault_score"),
            "component_score": r.get("component_score"),
            "path_text": r.get("path_text", ""),
        } for r in records]


class GraphSearchDeviceTool(BaseTool):
    """图谱设备搜索工具，按关键字模糊匹配 Device 节点。"""

    @property
    def name(self) -> str:
        return "graph_search_devices"

    @property
    def description(self) -> str:
        return (
            "从知识图谱中按关键字搜索设备节点。"
            "支持按设备名称、编码、型号、位置进行模糊匹配。"
            "适用场景：不确定设备全名时搜索设备列表，为诊断路径查询缩小范围。"
        )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "设备搜索关键字"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限，默认10",
                    "default": 10
                }
            },
            "required": ["keyword"]
        }

    async def _execute(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            graph = get_graph_service()
            devices = graph.find_devices(keyword=keyword, limit=limit)
        except Exception as e:
            raise ToolException(
                code="GRAPH_ERROR",
                message=f"设备搜索失败: {e}"
            )

        return [{
            "id": d["id"],
            "name": d["name"],
            "code": d.get("code", ""),
            "model": d.get("model", ""),
            "location": d.get("location", ""),
            "manufacturer": d.get("manufacturer", ""),
        } for d in devices]


# 单例
_query_tool: Optional[GraphQueryTool] = None
_search_tool: Optional[GraphSearchDeviceTool] = None


def get_graph_query_tool() -> GraphQueryTool:
    global _query_tool
    if _query_tool is None:
        _query_tool = GraphQueryTool()
    return _query_tool


def get_graph_search_device_tool() -> GraphSearchDeviceTool:
    global _search_tool
    if _search_tool is None:
        _search_tool = GraphSearchDeviceTool()
    return _search_tool
