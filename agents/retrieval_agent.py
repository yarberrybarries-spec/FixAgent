"""
RetrievalAgent — 知识检索Agent（ReAct模式）

负责从知识库中检索相关文档、案例和资料。
使用 ReAct 循环，LLM 自主决定何时检索、何时追问用户补充信息。

【与架构文档的对应关系】
- 位置：agents/retrieval_agent.py
- 继承：agents/base_agent.py 的 BaseAgent
- 依赖：tools/knowledge_retrieval_tool.py
- 被调用：agents/orchestrator_agent.py

【ReAct 执行流程】
用户询问 → Agent思考需要什么信息 → 调用 knowledge_retrieval 工具
→ 观察检索结果 → 不够则追问用户 / 够了则整理答案 → 返回
"""

from typing import List
import logging

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    """
    知识检索 Agent

    使用 ReAct 循环自主检索知识库，必要时追问用户。
    """

    @property
    def name(self) -> str:
        return "retrieval_agent"

    @property
    def description(self) -> str:
        return "设备检修知识检索助手，帮助查找技术资料和历史案例"

    def get_system_prompt(self) -> str:
        return (
            "你是设备检修知识检索专家。\n\n"
            "## 你的职责\n"
            "1. 理解用户的技术问题\n"
            "2. 使用 knowledge_retrieval 工具从知识库中检索最相关的内容\n"
            "3. 根据检索结果整理出准确、有用的信息\n\n"
            "## 工作方式\n"
            "- 收到用户问题后，先提取核心关键词再检索\n"
            "- 如果检索结果不够相关，可以换关键词重新检索\n"
            "- 如果用户问题信息不足（如没说明设备型号），先追问再检索\n"
            "- 检索完成后，用简洁的中文整理答案\n\n"
            "## 输出要求\n"
            "- 优先引用检索到的权威资料（检修手册、技术规格）\n"
            "- 关键点用列表呈现\n"
            "- 如果信息不足，明确说明并建议用户补充哪些信息\n"
        )

    def get_tools(self) -> List:
        from tools.knowledge_retrieval_tool import get_knowledge_retrieval_tool
        return [get_knowledge_retrieval_tool()]


# 单例
_retrieval_agent = None


def get_retrieval_agent() -> RetrievalAgent:
    global _retrieval_agent
    if _retrieval_agent is None:
        from services.llm_service import get_llm_service
        _retrieval_agent = RetrievalAgent(get_llm_service())
    return _retrieval_agent
