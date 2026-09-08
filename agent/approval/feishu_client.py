"""
飞书审批客户端

基于 lark-oapi 创建飞书审批实例（设计文档 §7.3）。

要点：
- 审批流编码 feishu_approval_code 由飞书后台配置，审批人/角色自动路由；
- 表单仅携带**脱敏摘要**（操作类型 / 目标表 / 参数摘要 / 发起人），不展示完整参数值（§11）；
- 飞书配置缺失或调用失败时抛出 FeishuApprovalError，由上层决定降级。
"""
import json
import logging

import lark_oapi as lark
from lark_oapi.api.approval.v4 import CreateInstanceRequest, InstanceCreate

logger = logging.getLogger(__name__)

# 飞书请求超时时间（秒）
FEISHU_TIMEOUT_SECONDS = 10


class FeishuApprovalError(Exception):
    """飞书审批客户端错误（配置缺失 / 调用失败）"""


class FeishuApprovalClient:
    """飞书审批客户端：创建审批实例"""

    def __init__(self, app_id: str = "", app_secret: str = ""):
        """
        :param app_id: 飞书应用 App ID
        :param app_secret: 飞书应用 App Secret
        """
        if not app_id or not app_secret:
            raise FeishuApprovalError("飞书审批未配置（缺少 app_id / app_secret）")
        self._client = (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .timeout(FEISHU_TIMEOUT_SECONDS)
            .log_level(lark.LogLevel.ERROR)
            .build()
        )

    def create_instance(
        self,
        approval_code: str,
        user_id: str,
        form_fields: list[dict],
    ) -> str:
        """
        创建审批实例，返回 instance_code。

        :param approval_code: 飞书审批流编码（审批人/角色在后台配置，自动路由）
        :param user_id: 发起人飞书 user_id/open_id
        :param form_fields: 表单摘要，如 [{"name": "操作类型", "value": "删除数据"}, ...]
        :return: 飞书审批实例编码
        """
        if not approval_code:
            raise FeishuApprovalError("飞书审批未配置（缺少 approval_code）")
        if not form_fields:
            raise FeishuApprovalError("审批表单不能为空")

        # v4 接口 form 字段为 JSON 字符串
        body = (
            InstanceCreate.builder()
            .approval_code(approval_code)
            .user_id(user_id)
            .form(json.dumps(form_fields, ensure_ascii=False))
            .build()
        )
        req = CreateInstanceRequest.builder().request_body(body).build()

        resp = self._client.approval.v4.instance.create(req)
        if not resp.success():
            logger.error(
                f"[Feishu] 创建审批实例失败: code={resp.code}, msg={resp.msg}"
            )
            raise FeishuApprovalError(f"创建飞书审批失败（code={resp.code}）")

        instance_code = getattr(getattr(resp, "data", None), "instance_code", None)
        if not instance_code:
            raise FeishuApprovalError("创建飞书审批成功但未返回 instance_code")
        logger.info(f"[Feishu] 审批实例创建成功: instance_code={instance_code}")
        return instance_code
