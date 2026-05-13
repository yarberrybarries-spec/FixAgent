# API 模块

## 模块职责

FastAPI Web 服务入口，HTTP 接口定义、请求路由、参数校验。所有 AI 推理逻辑由 `agents/` 完成，业务数据由 Java 后端管理。

## 接口列表

| 接口 | 方法 | 描述 | 状态 |
|------|------|------|------|
| `/ai/chat` | POST | 对话接口（意图识别 + 路由分发） | **已实现** |
| `/ai/chat/stream` | POST | SSE 流式响应（仅 CHAT 模式） | **已实现** |
| `/ai/retrieval` | POST | 直接调用 RetrievalAgent | **占位** |
| `/ai/diagnosis` | POST | 直接调用 DiagnosisAgent | **占位** |
| `/ai/guidance` | POST | 直接调用 GuidanceAgent | **占位** |
| `/ai/pipeline` | POST | 完整流程（检索→诊断→指引） | **占位** |
| `/ai/memory/consolidate` | POST | 记忆整理（function calling + 向量存储） | **已实现** |

## 请求模型

`schemas/request.py` 中定义：

- `ChatRequest` — session_id / message / mode / images / stream
- `MemoryConsolidateRequest` — session_id / memoryMessages / memoryPreferenceVOList / memoryUnresolvedVOList

## 响应模型

`schemas/response.py` 中定义：

- `ChatResponse` — session_id / message / intention / tools_used / latency_ms
- `MemoryConsolidateResponse` — session_id / summary(MemorySummary) / original_count / consolidated_at

## 日志输出点

关键位置输出 INFO 级别日志（控制台实时可见）：

| 接口 | 位置 | 日志内容 |
|------|------|---------|
| `/ai/chat` | api/main.py chat() | 请求入参（session/mode/消息长度）、完成耗时 |
| `/ai/memory/consolidate` | api/main.py memory_consolidate() | 对话条数、Agent 错误详情、完成耗时 |

日志格式：`2026-05-13 19:32:15 | INFO     | api.main | [chat] session=abc123 mode=DIAGNOSIS msg_len=24`

## 调用关系

```
api/main.py
    ├── schemas/request.py      — 请求模型
    ├── schemas/response.py     — 响应模型
    ├── agents/orchestrator_agent.py — 调度中枢（单例，惰性创建）
    └── agents/memory_agent.py      — 记忆整理（通过 get_memory_agent() 获取）

orchestrator_agent.py
    ├── agents/intention/         — 意图识别
    ├── chains/orchestrator.py    — 意图→模式映射
    └── 子Agent（retrieval/diagnosis/guidance）
            └── tools/*            — 工具能力
                    └── services/* — 外部服务
```

## 与 Java 后端的交互

```
Java Backend                    FixAgent (Python)
  POST /ai/chat                     → OrchestratorAgent → 子Agent → ChatResponse
  POST /ai/memory/consolidate       → MemoryAgent → MemorySummary → MemoryConsolidateResponse
  POST /ai/chat/stream (SSE)       → OrchestratorAgent.run_stream() → SSE token 流
```

## 错误处理

- Agent 执行失败（`metadata.status="error"`）→ API 层检测后 raise HTTPException(500)
- LLM 返回 content=null（tool_call 场景）→ `content or ""` 兜底，JSON 解析失败 → fallback + warning 日志
- 请求参数校验失败 → FastAPI 自动返回 422

## 启动方式

```bash
# 开发环境（热重载）
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 生产环境（多进程）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 文件结构

```
api/
├── __init__.py
└── main.py                     # FastAPI 入口，含 /ai/* 所有端点
```

## 注意事项

1. **日志级别**：生产环境将 `logging.basicConfig(level=logging.INFO)` 改为 `WARNING`
2. **Agent 惰性初始化**：应用启动时不加载 LLM，首次请求时才创建实例
3. **会话追踪**：`session_id` 由 Java 生成并传递，用于日志分片和链路追踪
4. **超时设置**：建议 HTTP 超时 > 60s（AI 推理耗时较长）
5. **SSE 协议**：当前流式接口仅推送 `session_id` / `token` / `done` 事件，ReAct 步骤流式推送待实现