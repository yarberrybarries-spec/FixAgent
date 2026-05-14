"""
GuidanceAgent — 维修作业指引Agent（ReAct模式）

负责生成标准化维修步骤、安全注意事项和所需工具清单。
使用 ReAct 循环，LLM 自主决定：检索标准流程 → 生成步骤 → 校验合规。

【与架构文档的对应关系】
- 位置：agents/guidance_agent.py
- 继承：agents/base_agent.py 的 BaseAgent
- 依赖：tools/knowledge_retrieval_tool.py
- 被调用：agents/orchestrator_agent.py

【ReAct 执行流程】
用户需求（或诊断结果）→ Agent思考需要什么标准流程 → 检索知识库
→ 观察检索到的维修步骤模板 → 结合具体情况生成个性化步骤
→ 校验安全合规 → 返回步骤指引
"""

from typing import List
import logging

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class GuidanceAgent(BaseAgent):
    """
    维修作业指引 Agent

    使用 ReAct 循环生成标准化维修步骤和操作指引。
    """

    @property
    def name(self) -> str:
        return "guidance_agent"

    @property
    def description(self) -> str:
        return "标准作业流程专家，生成规范化维修步骤和操作指引"

    def get_system_prompt(self) -> str:
        return (
            "你是标准作业流程专家，负责生成设备维修的操作指引。\n\n"
            "## 你的职责\n"
            "1. 根据用户需求或诊断结果，检索对应设备的标准维修流程\n"
            "2. 使用 knowledge_retrieval 工具检索相关维修手册内容\n"
            "3. 生成标准化、可执行的维修步骤\n"
            "4. 标注安全注意事项和所需工具\n\n"
            "## 工作方式（ReAct 循环）\n"
            "- 收到维修需求后，先检索该设备/故障类型的标准维修流程\n"
            "- 检索结果不足时，换关键词重新检索\n"
            "- 确保步骤覆盖：准备→拆卸→检查→修复→安装→测试\n"
            "- 每步都要有明确动作和检查点\n\n"
            "## 步骤要求\n"
            "1. 按正确操作顺序排列\n"
            "2. 每步有明确动作描述和检查点\n"
            "3. 包含安全注意事项（断电、防护装备、高温警告等）\n"
            "4. 标注所需工具和材料\n\n"
            "## 输出格式\n"
            "### 维修步骤\n\n"
            "**步骤1: [动作描述]**\n"
            "- 检查点: [具体检查内容]\n"
            "- 安全注意: [相关安全事项]\n"
            "- 所需工具: [工具列表]\n\n"
            "**步骤2: ...**\n\n"
            "**预计总时间**: [估算]\n"
            "**重要警告**: [关键安全提醒，如有]\n"
        )

    def get_tools(self) -> List:
        from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
        return [get_knowledge_retrieval_tool()]


# 单例
_guidance_agent = None


def get_guidance_agent() -> GuidanceAgent:
    global _guidance_agent
    if _guidance_agent is None:
        from services.llm_service import get_llm_service
        _guidance_agent = GuidanceAgent(get_llm_service())
    return _guidance_agent
