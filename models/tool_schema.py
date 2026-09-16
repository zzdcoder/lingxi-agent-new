"""
工具注册表 Schema

供工具管理 API（api/routes/tools.py）与外部调用使用的 Pydantic 模型。
"""
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from models.tool_model import ToolStatus


class ToolOut(BaseModel):
    """工具信息输出（管理台列表 / 详情）"""
    id: str
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    risk_level: Optional[str] = None
    requires_approval: bool = False
    status: int = ToolStatus.ACTIVE
    source: Optional[str] = None
    version: Optional[str] = None
    circuit_state: Optional[str] = "CLOSED"
    consecutive_failures: int = 0
    total_calls: int = 0
    success_calls: int = 0
    fail_calls: int = 0
    circuit_open_at: Optional[datetime] = None
    circuit_open_until: Optional[datetime] = None
    last_error: Optional[str] = None
    last_call_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    avg_latency_ms: int = 0
    remark: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ToolStatusUpdate(BaseModel):
    """
    工具状态变更请求（管理台操作）

    合法流转（其余组合由服务层拒绝）：
    - 1 → 0：人工下线（失效）；
    - 0 → 1：人工恢复（重新生效）；
    - 1 → 2：人工熔断（立即打开熔断器）；
    - 2 → 1：人工恢复（关闭熔断器）；
    - 2 → 0：熔断后直接下线。
    """
    status: Literal[0, 1, 2] = Field(description="目标状态: 1=生效 0=失效 2=熔断")
    remark: Optional[str] = Field(default=None, max_length=255, description="操作备注（熔断/下线原因，审计用）")


class ToolSyncResult(BaseModel):
    """启动注册结果（registrar.sync_tool_registry 返回值）"""
    scanned_modules: int = 0
    decorated_tools: int = 0          # 扫描到的 @tool 装饰器工具数
    registry_tools: int = 0           # ToolSpec 注册表工具数
    inserted: int = 0                 # 新增入库
    updated: int = 0                  # 元数据更新
    skipped_status_preserved: int = 0  # 跳过（已存在且保持原状态，仅刷新元数据）
    errors: list[str] = Field(default_factory=list)
