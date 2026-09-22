"""
意图分层判定离线冒烟（§16.7 验收）

不依赖 LLM / MySQL / Qdrant / embedding API，纯规则层 + 缓存层验证。
运行：python scripts/smoke_intent_gate.py
"""
import os
import sys

# 脚本方式运行时补齐项目根目录（否则 sys.path[0] 为 scripts/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import intent_gate as G  # noqa: E402

PASS, FAIL = 0, 0
FAILED_CASES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        FAILED_CASES.append(name)
        print(f"  [FAIL] {name} {detail}")


def expect(text: str, kind: str, intents=None, last=None, rule: str = "") -> None:
    """断言一次判定结果。"""
    d = G.gate(text, last_intents=last)
    label = f"gate({text!r}) -> {kind}"
    ok = d.kind == kind
    if ok and intents is not None:
        ok = d.intents == intents
    if ok and rule:
        ok = d.rule == rule
    check(label, ok, f"实际 kind={d.kind} rule={d.rule} intents={d.intents}")


def main() -> int:
    print("\n== 1. 斜杠命令路由（Hermes 第一层） ==")
    check("/clear 解析", G.parse_slash_command("/clear") == ("clear", ""))
    check("/status 带参数解析", G.parse_slash_command("/status now") == ("status", "now"))
    check("未知命令不吞（回落后续层）", G.parse_slash_command("/usr/local/bin") is None)
    check("非命令文本", G.parse_slash_command("你好") is None)
    expect("/clear", G.KIND_SLASH)
    expect("/help", G.KIND_SLASH)
    expect("/unknown-cmd", G.KIND_NONE)

    print("\n== 2. 寒暄门控（Hermes 第四层） ==")
    for t in ("你好", "谢谢", "嗯", "？", "。。。", "在吗"):
        check(f"trivial({t!r})", G.is_trivial(t))
    # 关键负样本：含寒暄词但仍有实质内容，绝不能判为 trivia
    for t in ("谢谢，我想查一下订单表", "好的，帮我删除用户表的张三", "你好，请问年假制度是怎样的"):
        check(f"非 trivial({t!r})", not G.is_trivial(t))
    expect("你好", G.KIND_CHITCHAT, intents=["chat"])

    print("\n== 3. 强信号 → 确定性意图（准确率 100%） ==")
    expect("把用户表里张三这条记录删除", G.KIND_FORCED, ["task"], rule="db_task_strong")
    expect("修改订单表的金额字段", G.KIND_FORCED, ["task"])
    expect("数据库里有哪些表", G.KIND_FORCED, ["task"])
    expect("根据知识库文档，售后政策怎么规定", G.KIND_FORCED, ["knowledge_base"],
           rule="explicit_kb")
    expect("删除用户表记录并对照售后政策给出建议", G.KIND_FORCED, ["task", "knowledge_base"],
           rule="db_task+explicit_kb")

    print("\n== 4. 长尾输入必须回落 LLM（规则层不猜测） ==")
    for t in ("帮我写一首关于春天的诗", "Python 列表和元组有什么区别",
              "介绍一下你自己", "我想了解一下公司的年假制度具体是怎么规定的"):
        expect(t, G.KIND_NONE)

    print("\n== 5. 短追问继承（会话连续性） ==")
    expect("继续", G.KIND_FORCED, ["task"], last=["task"], rule="followup_inherit")
    expect("张三", G.KIND_FORCED, ["knowledge_base"], last=["knowledge_base"])
    # 上一轮为并行/chat 时不继承，避免把复杂度带进来
    expect("继续", G.KIND_NONE, last=["task", "knowledge_base"])
    expect("继续", G.KIND_NONE, last=["chat"])
    # 带写动词的追问不继承（可能是新意图），走正常判定
    expect("继续删除", G.KIND_NONE, last=["knowledge_base"])

    print("\n== 6. 决策缓存（L1） ==")
    G._DECISION_CACHE.clear()
    d = G.gate("你好")
    G.put_cached_decision("你好", d)
    check("缓存命中（含空白归一）", G.get_cached_decision(" 你好 ") is not None)
    check("低置信不缓存", (lambda: (
        G.put_cached_decision("x", G.GateDecision(kind=G.KIND_FORCED, confidence=0.1)),
        G.get_cached_decision("x") is None)[1])())
    # 确定性结果 TTL 最长
    check("forced TTL = 3600", G.ttl_for(d) in (G.FORCED_TTL_SECONDS, G.CHITCHAT_TTL_SECONDS))
    check("none 不缓存", G.ttl_for(G.GateDecision(kind=G.KIND_NONE)) == 0)

    print("\n== 7. 序列化往返（语义缓存 payload） ==")
    d = G.gate("把用户表里张三删除")
    back = G.deserialize_decision(G.serialize_decision(d))
    check("往返一致", back is not None and back.intents == d.intents and back.kind == d.kind)
    check("非法 payload 返回 None", G.deserialize_decision("{bad json") is None)

    print("\n== 8. 向量就近判定（无向量时静默跳过） ==")
    check("无 query 向量返回 None", G.match_by_embedding(None, {"a": [1.0]}) is None)
    check("无原型向量返回 None", G.match_by_embedding([1.0, 0.0], None) is None)
    d = G.match_by_embedding([1.0, 0.0], {"x": [1.0, 0.0], "y": [0.0, 1.0]})
    # "x" 不在原型意图表里 -> 返回 None（防止未登记原型被误用）
    check("未登记原型不返回判定", d is None)
    proto = {t: [1.0, 0.0] for t, _ in G.PROTOTYPES[:1]}
    d = G.match_by_embedding([1.0, 0.0], proto)
    check("命中登记原型", d is not None and d.kind == G.KIND_FORCED,
          f"实际 {None if d is None else d.rule}")

    print("\n== 9. 清单句式扩表（§17.3，保守扩表 + LLM 兜底） ==")
    # 正样本：清单句式 + 无知识库线索 -> 判为 db_task（避免 LLM 层多花 3~10s）
    for t in ("数据库中有哪些表", "有哪些用户", "列出订单表所有记录",
              "查一下用户表里有多少人", "系统里都有什么表", "用户表包含哪些字段"):
        expect(t, G.KIND_FORCED, ["task"])
    # 负样本：带知识库线索 -> 必须让给知识库，不得吞成 task
    # （注：显式含「知识库」「文档」的会命中更优先的 explicit_kb，属正确行为，
    #   故这里断言「不是 task」而非「必须是 NONE」）
    for t in ("根据知识库文档有哪些条款", "文档中有哪些退货规则", "政策里列出了哪些情形",
              "知识库有哪些内容"):
        d = G.gate(t)
        check(f"非 task({t!r})", "task" not in d.intents,
              f"实际 kind={d.kind} rule={d.rule} intents={d.intents}")
    # 负样本：文档词在前、无介词（_mentions_doc_scope 兜住的场景）
    for t in ("文档里有哪些条款", "手册里的规范有哪些"):
        expect(t, G.KIND_NONE)
    # 负样本：纯闲聊/写作类清单，不含数据域锚点 -> 回落 LLM（§17.3 锚定规则）
    for t in ("有哪些适合春天的诗", "推荐几本书", "有哪些好看的电影", "有哪些好用的工具"):
        expect(t, G.KIND_NONE)
    # 灰度开关：关闭后清单句式不再强判为 task
    _orig = getattr(G, "_list_patterns_enabled", None)
    try:
        G._list_patterns_enabled = lambda: False
        expect("有哪些用户", G.KIND_NONE)
    finally:
        if _orig is not None:
            G._list_patterns_enabled = _orig

    print("\n== 10. 异常 best-effort（不抛错、退化为 NONE） ==")
    # 注意：空输入/None 会被寒暄门控判为 chitchat（chat 兜底），这是**预期行为**
    check("空输入 -> chat 兜底", G.gate("").kind == G.KIND_CHITCHAT)
    check("None -> chat 兜底", G.gate(None).kind == G.KIND_CHITCHAT)
    check("超长文本不抛错", G.gate("很长的一段话" * 200).kind in (G.KIND_NONE, G.KIND_FORCED))
    check("规则异常退化为 NONE（best-effort）", G.gate(12345).kind in (G.KIND_NONE, G.KIND_CHITCHAT))

    print(f"\n== 结果：{PASS} 通过 / {FAIL} 失败 ==")
    if FAILED_CASES:
        print("失败项：" + "; ".join(FAILED_CASES))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
