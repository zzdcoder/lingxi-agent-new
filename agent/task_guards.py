"""
任务 Agent 的循环防护中间件（设计文档 §25）

背景（生产 trace=0e7fc2dd93554f24）：用户问「统计用户 dada 有多少条消息会话」，
工具能力先天不可达（`query_data` 没有 COUNT/GROUP BY，`conversation_message`
也没有 user_id 关联列），模型却在「报错 → 换个参数 → 再报错」里空转了
**24 次 LLM 调用 / 136 秒**，最后把没法做的事埋在一段报告正文里。

既有约束为什么全部失效：

- `recursion_limit=100` 计的是**超步**（model=1 步、tools=1 步），一次「模型+工具」
  往返就是 2 步，24 次调用约 48 步 —— 根本没到 100；
- 审批模式下连墙钟超时都没有（`use_timeout=not approval_mode`）；
- `ToolCallLimitMiddleware` **只计调用次数、不比较参数**，而空转的特征恰恰是
  「参数相同或几乎相同地反复调用」—— 计次治不住它。

本模块提供 `RepeatToolCallGuardMiddleware`：在 `after_model` 钩子里对
`(tool_name, args)` 做指纹计数，同一指纹累计达到阈值就**抛异常终止**，
由 `task_node` 转成中文结构化结论。

为什么用「抛异常」而不是 `jump_to="end"`：
`exit_behavior="end"` 会注入一条**英文**的 AIMessage，而 `_extract_answer`
取的是 `messages[-1].content`，那句话会原样出现在用户面前。抛异常则由
`task_node` 统一产出中文结论（与 `ModelCallLimitExceededError` 同一条路径）。
"""
import json
import logging
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware

from agent.observability import metrics

logger = logging.getLogger(__name__)


class RepeatedToolCallError(Exception):
    """
    同一工具以相同参数被反复调用，Guard 判定为「空转」并终止任务。

    :param tool_name: 触发守卫的工具名
    :param args: 触发守卫的参数（已脱敏截断）
    :param count: 该指纹累计出现次数
    :param max_repeats: 阈值
    """

    def __init__(
        self,
        tool_name: str,
        args: str,
        count: int,
        max_repeats: int,
    ) -> None:
        self.tool_name = tool_name
        self.args = args
        self.count = count
        self.max_repeats = max_repeats
        super().__init__(
            f"工具 {tool_name} 以相同参数被调用 {count} 次（阈值 {max_repeats}）"
        )

    def user_message(self) -> str:
        """面向用户的中文结论（不暴露内部计数口径以外的敏感信息）。"""
        return (
            "任务已停止：检测到同一个工具在用相同的参数反复重试，"
            "继续下去不会得到新结果。\n"
            f"- 工具：{self.tool_name}\n"
            f"- 参数：{self.args}\n"
            f"- 已重试：{self.count} 次\n\n"
            "这通常意味着当前工具无法表达你要的操作（例如需要跨表关联、"
            "模糊匹配或排序取 TopN）。请把需求拆得更具体一些，"
            "或换一种可回答的问法（例如统计某张表的总行数、按某个字段分组计数）。"
        )


def tool_call_fingerprint(name: str, args: Any) -> str:
    """
    构造调用指纹：`工具名 + 规范化参数`。

    参数规范化是必要的 —— 同一个语义的调用，`filters` 的键顺序可能不同，
    直接比较 `repr` 会漏判；值是 list/dict 时更要按 key 排序后序列化。
    """
    try:
        canonical = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # 防御：任何不可序列化的入参都退化成 repr，不让守卫自己炸
        canonical = repr(args)
    return f"{name}|{canonical}"


def _iter_tool_calls(messages: Any):
    """
    从消息历史里逐条取出 `(工具名, 参数)`。

    中间件拿到的 state["messages"] 里既有 LangChain 消息对象，也可能（取决于
    上层如何构造 state）出现 dict 形态，两种都要兼容；取不到就跳过，
    **绝不能因为解析失败而阻断任务**（守卫是保护，不是新故障源）。
    """
    for msg in messages or []:
        calls = None
        if isinstance(msg, dict):
            calls = msg.get("tool_calls")
        else:
            calls = getattr(msg, "tool_calls", None)
        if not calls:
            continue
        for call in calls:
            if isinstance(call, dict):
                name = call.get("name") or call.get("tool") or ""
                args = call.get("args", {})
            else:
                name = getattr(call, "name", "") or ""
                args = getattr(call, "args", {})
            if name:
                yield name, args


class RepeatToolCallGuardMiddleware(AgentMiddleware):
    """
    同参重复调用守卫（§25 P0-3）。

    在 `after_model` 阶段扫描整条消息历史，统计每个 `(工具名, 参数)` 指纹的
    **累计**出现次数（不是「连续」—— 空转往往夹杂着别的小动作，只看连续会漏判）。
    达到阈值即抛 `RepeatedToolCallError`，在工具真正执行**之前**终止。
    """

    def __init__(self, max_repeats: int = 3) -> None:
        super().__init__()
        # <2 无意义（等于禁用了重试），做下限保护避免配错值把守卫变成「永不触发」
        self.max_repeats = max(2, int(max_repeats))

    def after_model(self, state: Any, runtime: Any) -> Optional[dict]:
        try:
            messages = state.get("messages") if hasattr(state, "get") else None
            counts: dict[str, int] = {}
            first_args: dict[str, str] = {}
            for name, args in _iter_tool_calls(messages):
                fp = tool_call_fingerprint(name, args)
                counts[fp] = counts.get(fp, 0) + 1
                first_args.setdefault(
                    fp, json.dumps(args, ensure_ascii=False, default=str)[:200]
                )
            for fp, count in counts.items():
                if count >= self.max_repeats:
                    name = fp.split("|", 1)[0]
                    metrics.incr("agent.task.repeat_call_blocked")
                    logger.warning(
                        f"[任务守卫] 同参重复调用达阈值，终止任务: tool={name} "
                        f"count={count}/{self.max_repeats} args={first_args.get(fp, '')}"
                    )
                    raise RepeatedToolCallError(
                        tool_name=name,
                        args=first_args.get(fp, ""),
                        count=count,
                        max_repeats=self.max_repeats,
                    )
        except RepeatedToolCallError:
            raise
        except Exception as e:
            # 守卫自身异常一律降级为「不守卫」：它只是优化项，
            # 绝不能反过来击穿主链路（项目韧性原则）。
            logger.warning(f"[任务守卫] 重复调用守卫执行失败，已跳过（非致命）: {e}")
        return None
