"""
Agent 审批包

飞书审批（Human-in-the-loop）全链路：
- feishu_client：创建飞书审批实例
- approval_service：执行包装层（中断检测 / 审批单落库 / 恢复图执行）
- callback：飞书事件回调验签与解密
"""
