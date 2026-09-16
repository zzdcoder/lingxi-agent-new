"""
工具注册表模型（tool_registry 表）

企业级工具管理表：集中存储 Agent 全部工具的元数据、运行状态与熔断状态，
是"工具生命周期管理"的数据底座（设计文档 §8.3 / §15 工具治理）：

- 状态（status）：1=生效 / 0=失效 / 2=熔断（人工下线与熔断器自动打开均落此列）；
- 参数（parameters）：工具参数 JSON Schema（@tool 工具取 args_schema，执行器方法取签名推导），
  供管理台展示与后续参数级校验；
- 熔断字段：熔断器状态机（CLOSED/OPEN/HALF_OPEN）、连续失败次数、冷却期、
  半开探测配额、调用统计等，支撑自动熔断与恢复（设计文档 §7.7）。

写入时机：
1. 启动时由 agent/tools/registrar.py 扫描 @tool 装饰器工具 + ToolSpec 注册表 upsert；
2. 运行时由熔断器 / 工具管理 API 更新状态与指标。

约束：
- name 唯一（工具名即 Agent 调用名）；
- 状态流转：1→2（自动/手动熔断）、2→1（半开探测成功自动恢复 / 人工恢复）、
  1↔0（人工下线上线）、0/2→1（人工恢复）。
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String, DateTime, Text, Integer, JSON, SmallInteger, Float, Index, func,
)
from sqlalchemy.orm import Mapped, mapped_column

from core import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class ToolStatus:
    """工具状态（status 列取值）"""
    ACTIVE = 1          # 生效：允许绑定给 Agent
    DISABLED = 0        # 失效：人工下线（运维禁用，不再绑定）
    CIRCUIT_BREAK = 2   # 熔断：熔断器打开（连续失败/失败率超阈值，自动或人工触发）


class ToolCircuitState:
    """熔断器内部状态机（circuit_state 列取值）"""
    CLOSED = "CLOSED"        # 关闭：正常放行并统计错误率
    OPEN = "OPEN"            # 打开：熔断生效，快速失败拒绝调用
    HALF_OPEN = "HALF_OPEN"  # 半开：冷却期后放行少量探测请求验证恢复


class ToolRegistry(Base):
    """工具注册表 - Agent 工具元数据与运行状态（设计文档 §8.3）"""
    __tablename__ = "tool_registry"

    # ---- 基础元数据 ----
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid, comment="UUID")
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, comment="工具名（Agent 调用名）")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="工具能力描述（供 LLM 选择工具）")
    parameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="工具参数 JSON Schema（管理台展示）")
    category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="db", comment="工具分类: db/knowledge/search/custom")
    risk_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, default="read", comment="风险等级: read/write")
    requires_approval: Mapped[bool] = mapped_column(SmallInteger, nullable=False, default=0, comment="是否触发人工审批: 1/0")

    # ---- 状态 ----
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=ToolStatus.ACTIVE, comment="状态: 1=生效 0=失效 2=熔断")
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="工具来源: executor:<方法名> / module:<模块路径>:<属性名>")
    version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, default="1.0.0", comment="工具版本（工具变更时递增）")

    # ---- 熔断器参数（配置快照，随工具落地） ----
    failure_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=5, comment="连续失败阈值：CLOSED 下连续失败达此值触发熔断")
    failure_ratio: Mapped[float] = mapped_column(Float, nullable=True, comment="窗口失败率阈值快照（如 0.5），按时间窗口统计")
    window_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=60, comment="失败率统计窗口（秒）")
    cooldown_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=60, comment="熔断冷却期（秒），到期后进入半开探测")
    half_open_max_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=3, comment="半开探测最大放行次数（防雪崩）")

    # ---- 熔断器运行时状态 ----
    circuit_state: Mapped[str] = mapped_column(String(16), nullable=False, default=ToolCircuitState.CLOSED, comment="熔断状态: CLOSED/OPEN/HALF_OPEN")
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="连续失败次数")
    total_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="累计调用次数")
    success_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="累计成功次数")
    fail_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="累计失败次数")
    window_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="当前窗口失败次数")
    window_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="当前窗口调用次数")
    window_start_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="当前统计窗口开始时间")
    half_open_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="半开已放行探测次数")
    circuit_open_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="熔断打开时间")
    circuit_open_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="熔断到期时间（到期进入半开探测）")
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="最近一次失败原因（脱敏）")
    last_call_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="最近一次调用时间")
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="最近一次成功时间")
    avg_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="平均耗时（毫秒，指数平滑）")

    # ---- 审计 ----
    remark: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="备注（熔断原因 / 下线原因等）")
    created_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="system", comment="创建人（默认 system，注册器写入）")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.current_timestamp(), onupdate=func.current_timestamp(), comment="更新时间")

    __table_args__ = (
        Index("idx_tool_status", "status"),
        Index("idx_tool_category", "category"),
    )
