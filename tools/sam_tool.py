"""
SAM 图像分割工具 — TODO 待实现

负责对图像进行精细分割，提取部件轮廓和故障区域掩码。

【计划功能】
- 图像分割：根据边界框或点击点生成目标区域分割掩码
- 配合 YOLO：对低置信度检测结果做精细分割
- 提取区域特征：供后续分类识别

【关联】
- 上游：agents/diagnosis_agent.py（细化故障区域）
- 继承：tools/base_tool.py 的 BaseTool

TODO: 实现 SamSegmentTool(BaseTool)，包括：
- name / description 属性
- get_parameters_schema() → JSON Schema（image_url, bbox, point）
- _execute() → 调用 SAM 模型并返回分割掩码
"""
