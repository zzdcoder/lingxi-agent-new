"""
飞书事件回调处理（设计文档 §7.4）

安全基线：
- 验签：校验 verification_token，未配置或校验失败一律拒绝（debug 模式放行本地联调）；
- 解密：encrypt 载荷使用 AESCipher（SHA256(key) + AES-256-CBC + PKCS7）解密；
- 审计：回调原文（脱敏）落库 callback_payload。

事件订阅：
- approval.instance：审批实例状态变更（APPROVED/REJECTED/CANCELED）
- approval.task：审批任务状态变更（APPROVE/REJECT）
两者均映射为审批单状态流转，并触发后台图恢复。
"""
import json
import logging
from typing import Optional

import lark_oapi
from sqlalchemy import select

from core.config import settings
from core.database import AsyncSessionLocal
from models.approval_model import ApprovalRequest
from agent.approval.approval_service import resume_graph, set_completed, spawn_background

logger = logging.getLogger(__name__)


# =============================================================================
# 验签与解密
# =============================================================================

def _extract_token(payload: dict) -> Optional[str]:
    """兼容 schema 1.0 / 2.0 的 token 位置。"""
    return payload.get("token") or (payload.get("header") or {}).get("token")


def verify_and_decrypt(payload: dict) -> Optional[dict]:
    """
    校验飞书回调并解密，返回事件对象；校验失败返回 None。

    - url_verification 类型返回 {"challenge": ...} 用于事件订阅配置；
    - 其余类型返回解密后的事件 JSON。
    """
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    # 验签
    token = _extract_token(payload)
    if settings.feishu_verification_token:
        if token != settings.feishu_verification_token:
            logger.warning("[Feishu] 回调验签失败：token 不匹配")
            return None
    elif not settings.debug:
        logger.warning("[Feishu] 飞书验签令牌未配置，拒绝回调")
        return None
    # debug 模式且未配置令牌时放行（本地联调用）

    # 解密
    encrypt = payload.get("encrypt")
    if encrypt:
        try:
            cipher = lark_oapi.AESCipher(settings.feishu_encrypt_key or "")
            event = json.loads(cipher.decrypt_str(encrypt))
        except Exception as e:
            logger.error(f"[Feishu] 回调解密失败: {e}")
            return None
    else:
        event = payload.get("event") or payload.get("header") or payload

    return event


# =============================================================================
# 事件解析
# =============================================================================

def parse_approval_decision(event: dict) -> Optional[tuple[str, str]]:
    """
    解析审批事件，返回 (instance_code, decision)。

    decision ∈ {approved, rejected, canceled}；其他状态（PENDING/RUNNING 等）返回 None。
    """
    data = event.get("event", {}) or event.get("data", {}) or {}
    # 兼容未加密直传载荷（debug 本地联调）：event 本身即审批数据
    if not data.get("instance_code"):
        data = event
    # 兼容事件类型字段在不同层级
    event_type = event.get("type") or (event.get("header") or {}).get("event_type", "")

    instance_code = data.get("instance_code")
    status = (data.get("status") or "").upper()
    if not instance_code or not status:
        logger.info(f"[Feishu] 忽略无审批实例信息的回调: event_type={event_type}")
        return None

    if status in ("APPROVED", "APPROVE"):
        return instance_code, "approved"
    if status in ("REJECTED", "REJECT"):
        return instance_code, "rejected"
    if status == "CANCELED":
        return instance_code, "canceled"
    logger.info(f"[Feishu] 忽略非终态审批回调: instance_code={instance_code}, status={status}")
    return None


# =============================================================================
# 事件处理
# =============================================================================

async def _find_by_instance_code(db, instance_code: str) -> Optional[ApprovalRequest]:
    result = await db.execute(
        select(ApprovalRequest).where(ApprovalRequest.feishu_instance_code == instance_code)
    )
    return result.scalars().first()


async def _resume_in_background(approval_id: str, decision: str) -> None:
    """后台恢复图执行（独立数据库会话，避免与回调会话生命周期冲突）。"""
    try:
        async with AsyncSessionLocal() as db:
            await resume_graph(approval_id, decision, db)
    except Exception:
        logger.exception(f"[Feishu] 后台恢复失败: approval_id={approval_id}")
    finally:
        # 通知 SSE 等待方：恢复完成（含最终结果推送）
        set_completed(approval_id)


async def handle_approval_event(event: dict) -> Optional[str]:
    """
    处理审批事件并触发后台恢复。

    :return: 处理结果描述（未命中返回 None）
    """
    parsed = parse_approval_decision(event)
    if parsed is None:
        return None
    instance_code, decision = parsed

    async with AsyncSessionLocal() as db:
        approval = await _find_by_instance_code(db, instance_code)
        if approval is None:
            logger.warning(f"[Feishu] 审批实例未关联本地审批单: instance_code={instance_code}")
            return None

        # 审计：记录回调载荷
        approval.callback_payload = {
            "instance_code": instance_code,
            "decision": decision,
            "event_type": (event.get("header") or {}).get("event_type"),
        }
        await db.commit()
        approval_id = approval.id

    # 触发后台恢复（幂等由审批单状态机守卫）
    spawn_background(_resume_in_background(approval_id, decision))
    logger.info(f"[Feishu] 审批事件已受理: approval_id={approval_id}, decision={decision}")
    return approval_id
