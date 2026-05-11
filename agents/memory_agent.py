"""
工作记忆整理 Agent

将多条原始对话记录压缩为结构化记忆摘要。
Java 端在对话达到阈值（如30条）时触发整理，调用本 Agent 提取关键信息。

【执行流程】
1. 从 context 中取出对话列表
2. 格式化为文本块
3. 调用 LLM 生成摘要
4. 解析 JSON 返回结构化数据

【与其他模块的关系】
- 继承 BaseAgent，复用模板方法
- 由 api/main.py 的 /ai/memory/consolidate 端点调用
- 调用 services/llm_service.py 的百炼 API
"""

import json
import re
from datetime import datetime
from typing import Dict, Any

from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from services.llm_service import LLMService


# 记忆整理专用系统提示词
MEMORY_SYSTEM_PROMPT = """你是一个工作记忆整理助手。你的任务是将多轮对话记录压缩为一条结构化的记忆摘要。

## 提取规则
1. **核心问题**：用户最关心的问题是什么？用一句话概括。
2. **关键结论**：对话中给出了哪些重要诊断结果、建议或答案？逐条列出。
3. **用户偏好**：用户表达了哪些特殊偏好或约束？（如"我不要理论解释，只要实操步骤"）
4. **未解决问题**：用户问了但没得到明确答案的问题有哪些？
5. **整体摘要**：用200字以内概述这段对话的主要内容。

## 输出格式
请严格按以下 JSON 格式输出，不要输出其他内容：
```json
{
  "core_question": "用户的核心问题",
  "key_conclusions": ["结论1", "结论2"],
  "user_preferences": ["偏好1"],
  "unresolved": ["未解决问题1"],
  "brief_summary": "200字以内的整体摘要"
}
```

## 注意事项
- 如果某类信息没有，对应字段返回空数组 [] 或空字符串 ""
- 不要编造对话中没有的内容
- 摘要要精炼，去掉寒暄和重复内容"""


class MemoryAgent(BaseAgent):
    """
    工作记忆整理 Agent

    把30条对话喂给 LLM，吐出5个字段的结构化摘要。
    Java 端拿到摘要后存数据库，清掉原始对话。
    """

    @property
    def name(self) -> str:
        return "memory_agent"

    @property
    def description(self) -> str:
        return "工作记忆整理Agent：将多条原始对话压缩为结构化摘要"

    def get_system_prompt(self) -> str:
        return MEMORY_SYSTEM_PROMPT

    def _format_conversations(self, conversations: list) -> str:
        """将对话列表格式化为 LLM 可读的文本块"""
        lines = ["## 对话记录\n"]
        for item in conversations:
            role_label = "用户" if item.get("role") == "user" else "助手"
            seq = item.get("seq", "?")
            content = item.get("content", "")
            lines.append(f"[{seq}] {role_label}：{content}")
        lines.append("\n请根据以上对话记录生成记忆摘要，严格按 JSON 格式输出。")
        return "\n".join(lines)

    def _build_messages(self, input_data: AgentInput) -> list:
        """重写：将对话列表格式化后作为用户消息"""
        conversations = input_data.context.get("conversations", []) if input_data.context else []
        formatted = self._format_conversations(conversations)
        return [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": formatted}
        ]

    @staticmethod
    def _extract_json(text: str) -> dict:
        """从 LLM 返回内容中提取 JSON（处理 markdown 代码块包裹的情况）"""
        cleaned = text.strip()

        # 去掉 ```json ... ``` 包裹
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        return json.loads(cleaned)

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        执行记忆整理

        流程：构建消息 → 调 LLM → 解析 JSON → 存入 metadata 返回
        """
        import time
        start_time = time.time()

        messages = self._build_messages(input_data)
        response = await self._call_llm(messages, stream=False)

        latency_ms = int((time.time() - start_time) * 1000)

        content = response.get("content", "")

        # 尝试解析 JSON，失败时返回原始内容作为降级
        try:
            summary = self._extract_json(content)
        except (json.JSONDecodeError, ValueError):
            summary = {
                "core_question": "",
                "key_conclusions": [],
                "user_preferences": [],
                "unresolved": [],
                "brief_summary": content[:200]
            }

        return AgentOutput(
            agent_name=self.name,
            message=summary.get("brief_summary", ""),
            intention=None,
            tools_used=[],
            metadata={
                "summary": summary,
                "latency_ms": latency_ms
            },
            latency_ms=latency_ms
        )
