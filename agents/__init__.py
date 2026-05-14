"""
Agent 层模块

所有 AI Agent 的实现，采用 BaseAgent 模板方法模式 + ReAct function calling 架构。

Agent 清单：
- OrchestratorAgent — 总调度：意图识别 → 分发 → 流式响应
- RetrievalAgent    — 知识检索 Agent
- DiagnosisAgent    — 故障诊断 Agent（图谱查询 + 知识检索）
- GuidanceAgent     — 维修指导 Agent
- MemoryAgent       — 工作记忆整理 Agent
"""

from .base_agent import BaseAgent, AgentInput, AgentOutput
from .orchestrator_agent import OrchestratorAgent, get_orchestrator_agent
from .retrieval_agent import RetrievalAgent, get_retrieval_agent
from .diagnosis_agent import DiagnosisAgent, get_diagnosis_agent
from .guidance_agent import GuidanceAgent, get_guidance_agent
from .memory_agent import MemoryAgent, get_memory_agent

__all__ = [
    # 基类
    "BaseAgent",
    "AgentInput",
    "AgentOutput",
    # 调度
    "OrchestratorAgent",
    "get_orchestrator_agent",
    # 检索
    "RetrievalAgent",
    "get_retrieval_agent",
    # 诊断
    "DiagnosisAgent",
    "get_diagnosis_agent",
    # 指导
    "GuidanceAgent",
    "get_guidance_agent",
    # 记忆
    "MemoryAgent",
    "get_memory_agent",
]
