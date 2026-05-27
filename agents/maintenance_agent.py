"""
检修步骤生成 Agent（MaintenanceAgent）

接收检修任务信息（故障描述、设备类型、图片等），
通过 RAG 增强的 ReAct 模式生成结构化检修步骤。

【调用链】
MQ consumer → MaintenanceAgent.generate_steps() → ReAct循环(RAG工具) → 结构化JSON

【工具】
- knowledge_retrieval：检索维修手册知识库
- graph_search_java：查询设备→部件→故障→解决方案图谱

【输出】
JSON 结构化步骤列表，每步包含 title/content/safetyNote/requirePhoto/requireNote/estimatedMinutes
"""

import json
import logging
from typing import List, Dict, Any, Optional

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


MAINTENANCE_SYSTEM_PROMPT = """你是一名专业的设备检修流程生成AI，负责根据故障信息生成详细的检修步骤。

## 你的任务
根据提供的故障描述、设备信息，利用知识检索和图谱查询工具获取相关维修知识，然后生成一份结构化的检修步骤。

## 可用工具

### knowledge_retrieval
从向量知识库中检索与查询语义最相似的维修手册文档。
- 适用：查找设备维修方法、操作规程、安全注意事项
- 参数：query（查询文本）、top_k（返回数量，默认5）
- 用户有图片时必须传入 image_urls

### graph_search_java
从设备检修知识图谱中查询诊断路径：设备→部件→故障→解决方案。
- 适用：分析设备故障的因果关系、查找已知解决方案
- 参数：keyword（设备名称）、fault_description（故障描述）、image_urls（故障图片）、limit（数量）
- 四个参数至少传一个

## 工作流程
1. 先用 graph_search_java 查询设备故障的已知诊断路径和解决方案
2. 再用 knowledge_retrieval 检索相关维修手册内容
3. 综合以上信息，生成结构化检修步骤

## 输出格式要求
你必须严格返回以下JSON格式，不要添加任何其他文字：

```json
{
  "steps": [
    {
      "title": "步骤标题（简明扼要）",
      "content": "详细操作说明",
      "safetyNote": "安全注意事项（涉及高压、高温、旋转部件等必须写具体防护措施）",
      "requirePhoto": true或false,
      "requireNote": true或false,
      "estimatedMinutes": 预估耗时（整数，单位分钟）
    }
  ]
}
```

## 生成规则
1. 步骤数量通常 4-8 步，根据复杂度调整
2. 第一步通常是安全准备（断电/泄压/冷却等）
3. 最后一步通常是验证测试和复原
4. 涉及拆卸的步骤 requirePhoto = true（拍照留证）
5. 涉及测量数据的步骤 requireNote = true（记录数值）
6. safetyNote 必须具体，不能泛泛而谈
7. 每一步的 content 要足够详细，让初级维修工也能操作
8. 基于检索到的知识生成，不要凭空编造技术参数
"""


class MaintenanceAgent(BaseAgent):
    """检修步骤生成Agent"""

    def __init__(self, llm_service: LLMService):
        super().__init__(llm_service)
        self._tools = None

    @property
    def name(self) -> str:
        return "maintenance_agent"

    @property
    def description(self) -> str:
        return "检修步骤生成Agent：RAG增强生成结构化检修流程"

    def get_system_prompt(self) -> str:
        return MAINTENANCE_SYSTEM_PROMPT

    def get_tools(self) -> list:
        if self._tools is None:
            from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
            from tools.graph_java_tool import get_graph_java_tool

            self._tools = [
                get_knowledge_retrieval_tool(),
                get_graph_java_tool(),
            ]
        return self._tools

    async def generate_steps(
        self,
        fault_description: str,
        device_id: Optional[str] = None,
        device_name: Optional[str] = None,
        urgency_level: int = 1,
        report_images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        生成检修步骤

        Args:
            fault_description: 故障描述
            device_id: 设备ID
            device_name: 设备名称
            urgency_level: 紧急等级
            report_images: 报修图片URL列表

        Returns:
            {"success": True, "steps": [...]} 或 {"success": False, "error": "..."}
        """
        urgency_map = {0: "低", 1: "普通", 2: "紧急"}
        urgency_text = urgency_map.get(urgency_level, "普通")

        # 构建用户消息
        user_msg = f"请为以下故障生成检修步骤：\n\n"
        user_msg += f"**故障描述**：{fault_description}\n"
        if device_name:
            user_msg += f"**设备名称**：{device_name}\n"
        if device_id:
            user_msg += f"**设备ID**：{device_id}\n"
        user_msg += f"**紧急等级**：{urgency_text}\n"

        if report_images:
            user_msg += f"\n已附带 {len(report_images)} 张故障图片，请在工具调用时传入 image_urls。\n"

        input_data = AgentInput(
            user_message=user_msg,
            session_id=f"task-generate-{device_name or 'unknown'}",
            images=report_images,
            context={
                "device_id": device_id,
                "device_name": device_name,
                "urgency_level": urgency_level,
            }
        )

        try:
            result = await self.run_with_react(input_data, max_iterations=8)

            if result.metadata.get("status") == "error":
                return {
                    "success": False,
                    "error": result.metadata.get("error_detail", "Agent执行失败"),
                }

            # 解析JSON输出
            steps = self._parse_steps(result.message)
            if steps is None:
                return {
                    "success": False,
                    "error": "无法解析LLM输出为结构化步骤",
                }

            return {
                "success": True,
                "steps": steps,
                "latency_ms": result.latency_ms,
            }

        except Exception as e:
            logger.exception("[MaintenanceAgent] 生成步骤异常")
            return {
                "success": False,
                "error": str(e),
            }

    def _parse_steps(self, message: str) -> Optional[List[Dict]]:
        """从LLM输出中提取JSON步骤列表"""
        # 尝试直接解析
        try:
            data = json.loads(message)
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 尝试从 ```json ... ``` 代码块中提取
        import re
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', message)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "steps" in data:
                    return data["steps"]
            except json.JSONDecodeError:
                pass

        # 尝试提取数组
        json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', message)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        # 最后尝试找大括号块
        brace_match = re.search(r'\{[\s\S]*"steps"\s*:\s*\[[\s\S]*\]\s*\}', message)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return data.get("steps", [])
            except json.JSONDecodeError:
                pass

        logger.warning("[MaintenanceAgent] 无法解析步骤JSON，原始输出: %s", message[:500])
        return None


# 单例
_maintenance_agent = None


def get_maintenance_agent() -> MaintenanceAgent:
    global _maintenance_agent
    if _maintenance_agent is None:
        from services.llm_service import get_llm_service
        _maintenance_agent = MaintenanceAgent(get_llm_service())
    return _maintenance_agent
