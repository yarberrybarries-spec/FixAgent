"""
YOLO 目标检测工具 — TODO 待实现

负责对故障图片进行目标检测，识别设备部件和异常区域。

【计划功能】
- 目标检测：识别图片中的设备部件（轴承、齿轮、电机等）
- 异常检测：识别磨损、裂纹、烧蚀等故障区域
- 返回检测框坐标、类别和置信度

【关联】
- 上游：agents/diagnosis_agent.py（诊断时分析用户上传的故障图片）
- 继承：tools/base_tool.py 的 BaseTool

TODO: 实现 YoloDetectTool(BaseTool)，包括：
- name / description 属性
- get_parameters_schema() → JSON Schema（image_url, conf_threshold）
- _execute() → 调用 YOLO 模型并返回 DetectionResult 列表
"""
