"""
文档解析工具 — TODO 待实现

负责解析 PDF/Word 文档，提取文本、表格和图片内容。

【计划功能】
- PDF 解析：提取文本、表格和图片
- Word 解析：提取格式化文本和表格
- 结构化输出：按页/章节组织解析结果

【关联】
- 上游：agents/retrieval_agent.py（知识入库时解析上传的文档）
- 继承：tools/base_tool.py 的 BaseTool

TODO: 实现 DocumentParserTool(BaseTool)，包括：
- name / description 属性
- get_parameters_schema() → JSON Schema（file_url, file_type）
- _execute() → 调用解析服务并返回结构化文档内容
"""
