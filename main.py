from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import on_llm_request
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.api import llm_tool


COMPLETION_LISTEN_SECONDS = 10
FOLLOWUP_PROBABILITY = 0.5
MIN_TASK_SECONDS = 10

TOOL_USAGE_SYSTEM_PROMPT = (
    "\n[任务链工具使用规则："
    "当你准备承诺一个需要等待的动作时，必须先调用 chain_task 创建任务，"
    "再用角色语气自然回应。包括：泡茶、泡咖啡、做饭、拿东西、去宿舍/厨房/仓库、"
    "整理材料、查资料、找物品、带东西过来、等会儿给用户看等。"
    "这不只适用于用户命令；如果是你自己临时想做、主动提议、角色自然产生的下一步行动，"
    "只要会跨过当前回复、需要稍后回来告诉用户结果，也必须创建任务。"
    "如果你说“等会儿我去……”“我一会儿拿/带/整理/查……”“等用户同意后去做……”，"
    "或用户已经同意你刚才提出的延迟动作，都属于必须创建任务。"
    "创建时调用 action=create，tasks_json 必须是 JSON 数组字符串，"
    "例如 [{\"name\":\"拿拓片\",\"duration_minutes\":3}]。"
    "不要只在文字里承诺未来动作而不登记任务。"
    "只有纯粹闲聊、假设、能力说明，且你没有真的准备去做时，才不需要调用。]"
)


@dataclass
class ChainTask:
    name: str = ""
    description: str = ""
    duration_minutes: float | int = 10
    prompt: str = ""


@dataclass
class TaskChain:
    id: str = ""
    session_id: str = ""
    conversation_id: str = ""
    tasks: list[ChainTask] = field(default_factory=list)
    current_index: int = 0
    is_active: bool = True
    created_at: float = 0.0
    current_task_started_at: float = 0.0
    current_task_wake_at: float = 0.0
    completion_listen_started_at: float = 0.0
    completion_callback_at: float = 0.0
    callback_retry_count: int = 0

    def reset_completion_listener(self) -> None:
        self.completion_listen_started_at = 0.0
        self.completion_callback_at = 0.0
        self.callback_retry_count = 0

    @property
    def current_task(self) -> ChainTask | None:
        if 0 <= self.current_index < len(self.tasks):
            return self.tasks[self.current_index]
        return None

    @property
    def is_completed(self) -> bool:
        return self.current_index >= len(self.tasks) or not self.is_active

    def advance(self) -> ChainTask | None:
        self.current_index += 1
        return self.current_task if not self.is_completed else None


def _get_data_dir() -> str:
    try:
        from astrbot.api.star import StarTools
        return StarTools.get_data_dir("astrbot_plugin_taskchain_tool")
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "plugin_data", "astrbot_plugin_taskchain_tool",
        )


@register("TaskChainTool", "AstrBot", "让 AI 能安排并管理后台时间任务链", "1.0.0")
class TaskChainToolPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._chains: dict[str, TaskChain] = {}
        self._lock = asyncio.Lock()
        self._scheduler_task: asyncio.Task | None = None
        self._data_file = os.path.join(_get_data_dir(), "task_chains.json")
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        self._load_chains()

    @on_llm_request()
    async def _inject_chain_state(self, event: AstrMessageEvent, request: ProviderRequest) -> None:
        session_id = event.unified_msg_origin
        conversation_id = self._request_conversation_id(request)
        request.system_prompt += TOOL_USAGE_SYSTEM_PROMPT
        async with self._lock:
            changed = False
            for c in list(self._chains.values()):
                if c.session_id != session_id:
                    continue
                if c.conversation_id != conversation_id:
                    continue
                # 新对话/reset时清理当前会话任务；不会影响同一用户的其他会话。
                if request.conversation and (
                    not request.conversation.history
                    or request.conversation.history == "[]"
                ):
                    c.is_active = False
                    changed = True
                    continue
                if c.completion_callback_at:
                    task_name = c.current_task.name if c.current_task else "当前任务"
                    request.system_prompt += (
                        f"\n[当前任务链：{task_name}已完成。"
                        "用户在完成监听期内回复了你，请直接自然地给出完成后的结果，"
                        "不要再等待后台提醒，不要提任务链、工具或时间。]"
                    )
                    c.is_active = False
                    c.reset_completion_listener()
                    changed = True
                    continue
                if c.is_completed:
                    continue
                lines = []
                active_task_name = ""
                active_remain = 0
                for i, t in enumerate(c.tasks):
                    status = "进行中" if i == c.current_index else ("待开始" if i > c.current_index else "已完成")
                    remain = ""
                    if i == c.current_index and c.is_active and c.current_task:
                        secs = max(0, c.current_task_wake_at - time.time())
                        active_task_name = t.name
                        active_remain = int(secs)
                        remain = f"，还需{int(secs//60)}分{int(secs%60)}秒" if secs >= 60 else f"，还需{int(secs)}秒"
                    lines.append(f"  [{i+1}] {t.name}{remain} ({status})")
                request.system_prompt += (
                    "\n[当前任务链：\n"
                    + "\n".join(lines)
                    + (
                        f"\n当前仍在进行「{active_task_name}」，还没有完成。"
                        "用户普通回复时，当前状态已经在这里给出，除非用户明确问状态/剩多久，"
                        "不要再调用 chain_task list。"
                        "请直接结合用户回复自然回应，但必须保持任务仍在进行中。"
                        "禁止说快好啦、马上就好、已经倒进杯子、端上来、递给用户、可以喝/尝、完成。"
                        "不要把进度自行推进到最终步骤；偏好问题优先留给中途 interact，"
                        "创建任务后的第一轮不要主动问加奶/加糖/口味。"
                        if active_task_name and active_remain > 0
                        else ""
                    )
                    + "\n]"
                )
            if changed:
                self._save_chains()

    # ── 持久化 ──

    def _load_chains(self) -> None:
        try:
            with open(self._data_file, encoding="utf-8") as f:
                raw = json.load(f)
            now_t = time.time()
            for c in raw:
                if not isinstance(c, dict):
                    continue
                if now_t - c.get("current_task_wake_at", 0) > 300:
                    continue
                tasks = [self._parse_task(t) for t in c.pop("tasks", [])]
                tasks = [t for t in tasks if t]
                if not tasks:
                    continue
                allowed = set(TaskChain.__dataclass_fields__)
                chain_data = {k: v for k, v in c.items() if k in allowed}
                if not chain_data.get("id"):
                    continue
                current_index = int(chain_data.get("current_index", 0) or 0)
                if current_index < 0:
                    current_index = 0
                chain_data["current_index"] = min(current_index, len(tasks))
                self._chains[chain_data["id"]] = TaskChain(**chain_data, tasks=tasks)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            self._chains = {}

    def _save_chains(self) -> None:
        raw = []
        for c in self._chains.values():
            raw.append({
                "id": c.id, "session_id": c.session_id,
                "conversation_id": c.conversation_id,
                "current_index": c.current_index, "is_active": c.is_active,
                "created_at": c.created_at,
                "current_task_started_at": c.current_task_started_at,
                "current_task_wake_at": c.current_task_wake_at,
                "completion_listen_started_at": c.completion_listen_started_at,
                "completion_callback_at": c.completion_callback_at,
                "callback_retry_count": c.callback_retry_count,
                "tasks": [
                    {"name": t.name, "description": t.description,
                     "duration_minutes": t.duration_minutes, "prompt": t.prompt}
                    for t in c.tasks
                ],
            })
        with open(self._data_file, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def _request_conversation_id(self, request: ProviderRequest) -> str:
        conv = getattr(request, "conversation", None)
        cid = getattr(conv, "cid", "") if conv else ""
        if not isinstance(cid, (str, int)):
            return ""
        return str(cid or "")

    async def _current_conversation_id(self, session_id: str) -> str:
        conv_mgr = getattr(self.context, "conversation_manager", None)
        if not conv_mgr:
            return ""
        try:
            cid = await conv_mgr.get_curr_conversation_id(session_id)
            return str(cid or "")
        except Exception:
            return ""

    async def _get_recent_history(
        self,
        session_id: str,
        conversation_id: str = "",
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        conv_mgr = getattr(self.context, "conversation_manager", None)
        if not conv_mgr:
            return []
        try:
            conv = None
            if conversation_id:
                conv = await conv_mgr.get_conversation(session_id, conversation_id)
            if not conv:
                return []
            if not conv or not conv.history:
                return []
            history = json.loads(conv.history)
            return self._sanitize_provider_history(history, limit)
        except Exception:
            return []

    def _sanitize_provider_history(self, history: list[Any], limit: int) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")).strip())
                    elif isinstance(item, str):
                        parts.append(item.strip())
                content = "\n".join(part for part in parts if part)
            elif content is None:
                content = ""
            else:
                content = str(content)
            content = content.strip()
            if not content:
                continue
            cleaned.append({"role": str(role), "content": content})
        return cleaned[-limit:]

    def _looks_like_question(self, text: str) -> bool:
        question_marks = ("?", "？")
        question_words = (
            "要不要", "想不想", "需不需要", "可以吗", "行吗", "好吗",
            "加糖", "加奶", "怎么", "什么", "哪", "几", "是否", "是不是",
        )
        return text.strip().endswith(question_marks) or any(w in text for w in question_words)

    def _looks_like_completion_claim(self, text: str) -> bool:
        completion_words = (
            "好了", "做好了", "泡好了", "煮好了", "完成了", "弄好了",
            "来啦", "端着", "端来", "端上", "递给", "放在", "搁在",
            "倒进杯子", "倒入杯子", "倒进了杯子", "倒进杯里", "杯子里",
            "快好啦", "快好了", "马上就好", "马上就到",
            "趁热", "小心烫", "您的咖啡", "你的咖啡", "您的茶", "你的茶",
        )
        return any(w in text for w in completion_words)

    def _safe_in_progress_text(self, task_name: str) -> str:
        return f"我这边还在{task_name}，博士稍等一下。"

    def _looks_like_plain_completion(self, text: str, task_name: str) -> bool:
        normalized = text.strip().strip("。.!！~～")
        plain_patterns = (
            f"{task_name}好啦",
            f"{task_name}好了",
            f"{task_name}完成了",
            f"{task_name}弄好了",
        )
        return normalized in plain_patterns or (
            len(normalized) <= len(task_name) + 6
            and any(word in normalized for word in ("好啦", "好了", "完成了"))
        )

    def _fallback_completion_text(self, task_name: str) -> str:
        if any(word in task_name for word in ("咖啡", "拿铁")):
            return (
                "咖啡好了，博士。刚收尾完，香气还挺足的，"
                "我先放在你手边，小心烫。"
            )
        if any(word in task_name for word in ("茶", "泡茶")):
            return (
                "茶泡好了，博士。温度刚好还热着，"
                "我给你放旁边，慢慢喝。"
            )
        if any(word in task_name for word in ("查", "资料", "翻书", "找资料")):
            return (
                "我把资料翻完了，博士。刚才对了几处关键内容，"
                "有一条挺值得展开讲。"
            )
        if any(word in task_name for word in ("拿", "带", "取", "找")):
            return (
                f"我把{task_name}这事处理好了，博士。东西已经找着了，"
                "我这就给你看。"
            )
        return (
            f"{task_name}这边已经收尾了，博士。"
            "我刚确认了一遍，接下来可以继续说结果。"
        )

    def _parse_task(self, raw: Any) -> ChainTask | None:
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name", "")).strip()
        description = str(raw.get("description", "") or "")
        prompt = str(raw.get("prompt", "") or "")
        try:
            duration = float(raw.get("duration_minutes", 10))
        except (TypeError, ValueError):
            duration = 10
        if not math.isfinite(duration) or duration <= 0:
            duration = MIN_TASK_SECONDS / 60
        return ChainTask(
            name=name or "当前任务",
            description=description,
            duration_minutes=duration,
            prompt=prompt,
        )

    # ── LLM 工具 ──

    def _make_checkin(self, main: ChainTask) -> ChainTask | None:
        if main.duration_minutes < 2:
            return None
        duration = random.uniform(0.5, 1.0)
        if main.duration_minutes - duration < 0.5:
            return None
        return ChainTask(
            name=main.name,
            description=main.description,
            duration_minutes=duration,
            prompt="",
        )

    @llm_tool(name="chain_task")
    async def chain_task(
        self,
        event: AstrMessageEvent,
        action: str = "list",
        tasks_json: str = "[]",
        chain_id: str = "",
    ) -> str:
        """重要：角色做任何事都需要用此工具推进时间。只要你说了要去做一件事（泡茶/咖啡、拿东西、翻行李、烧水、查资料、找材料、等会儿带东西过来等），必须立即调用此工具。这是维持角色扮演沉浸感的核心机制。长任务会自动插入随机互动。create后按人设自然回应，不要说任务已完成。不要空参数调用；如果只是想查看状态，action用list。

        Args:
            action(string): create创建 list查看 cancel取消。默认list，空参数调用只会查看当前任务状态。
            tasks_json(string): action=create时必填，必须是JSON数组字符串。只放1个任务：[{"name":"做什么","duration_minutes":3}]
            chain_id(string): action=cancel时必填。
        """
        session_id = event.unified_msg_origin
        conversation_id = await self._current_conversation_id(session_id)
        action = (action or "list").strip().lower()

        async with self._lock:
            if action == "create":
                try:
                    tasks_data = json.loads(tasks_json)
                except json.JSONDecodeError:
                    return "错误：tasks_json 不是合法 JSON。"
                if not isinstance(tasks_data, list):
                    return "错误：tasks_json 必须是 JSON 数组。"
                parsed_tasks = [self._parse_task(t) for t in tasks_data]
                tasks_raw = [t for t in parsed_tasks if t]
                if not tasks_raw:
                    return "错误：任务列表不能为空。"

                for c in self._chains.values():
                    if (
                        c.session_id == session_id
                        and c.conversation_id == conversation_id
                        and not c.is_completed
                    ):
                        c.is_active = False
                        c.reset_completion_listener()

                for t in tasks_raw:
                    secs = max(t.duration_minutes * 60, MIN_TASK_SECONDS)
                    t.duration_minutes = secs / 60

                total_secs = sum(t.duration_minutes * 60 for t in tasks_raw)
                merged = ChainTask(
                    name=tasks_raw[0].name,
                    description=tasks_raw[0].description,
                    duration_minutes=total_secs / 60,
                    prompt=next((t.prompt for t in reversed(tasks_raw) if t.prompt), tasks_raw[0].prompt),
                )

                checkin = self._make_checkin(merged)
                main_secs = total_secs
                if checkin:
                    main_secs = max(MIN_TASK_SECONDS, total_secs - checkin.duration_minutes * 60)
                main = ChainTask(
                    name=merged.name,
                    description=merged.description,
                    duration_minutes=main_secs / 60,
                    prompt=merged.prompt,
                )
                tasks = [checkin, main] if checkin else [main]
                cid = uuid.uuid4().hex[:12]
                now_t = time.time()
                chain = TaskChain(
                    id=cid, session_id=session_id, conversation_id=conversation_id, tasks=tasks,
                    created_at=now_t, current_task_started_at=now_t,
                    current_task_wake_at=now_t + tasks[0].duration_minutes * 60,
                )
                self._chains[cid] = chain
                self._save_chains()
                main_dur = sum(t.duration_minutes * 60 for t in tasks_raw)
                done_time = datetime.fromtimestamp(now_t + main_dur).strftime("%H:%M:%S")
                return (
                    f"[任务已安排(id={cid})] "
                    f"预计{done_time}完成。当前状态：任务刚开始，仍在进行中。"
                    "本轮只允许自然回应“正在去做/正在准备/开始处理”，"
                    "不要询问加奶、加糖、口味、要不要配料等偏好；"
                    "偏好询问交给后续中途互动。"
                    "严禁说已经做好、端上来、递给用户、完成或可以品尝。"
                )

            elif action == "list":
                active = [c for c in self._chains.values()
                          if (
                              c.session_id == session_id
                              and c.conversation_id == conversation_id
                              and not c.is_completed
                          )]
                if not active:
                    return "当前没有活跃的任务链。"
                lines = []
                for c in active:
                    ct = c.current_task
                    if not ct:
                        continue
                    remain = max(0, c.current_task_wake_at - time.time())
                    lines.append(
                        f"  [{c.id}] 「{ct.name}」{ct.description} "
                        f"(剩余 {int(remain//60)} 分 {int(remain%60)} 秒) "
                        f"[{c.current_index}/{len(c.tasks)}]"
                    )
                if not lines:
                    return "当前没有活跃的任务链。"
                return "活跃任务链：\n" + "\n".join(lines)

            elif action == "cancel":
                chain = self._chains.get(chain_id)
                if chain and chain.session_id == session_id and chain.conversation_id == conversation_id:
                    chain.is_active = False
                    self._save_chains()
                    return f"任务链 {chain_id} 已取消。"
                return f"未找到任务链 {chain_id}。"

            elif action == "advance":
                chain = self._chains.get(chain_id)
                if not chain or chain.session_id != session_id or chain.conversation_id != conversation_id:
                    return f"未找到任务链 {chain_id}。"
                if chain.is_completed:
                    return f"任务链 {chain_id} 已经完成。"
                ct = chain.current_task
                nxt = chain.advance()
                if nxt:
                    chain.current_task_started_at = time.time()
                    chain.current_task_wake_at = time.time() + nxt.duration_minutes * 60
                    wake = datetime.fromtimestamp(chain.current_task_wake_at).strftime("%H:%M:%S")
                    self._save_chains()
                    return (
                        f"阶段「{ct.name}」已完成，进入「{nxt.name}」。\n"
                        f"{nxt.description}\n预计 {nxt.duration_minutes} 分钟（{wake}）后唤醒。"
                    )
                chain.is_active = False
                chain.reset_completion_listener()
                self._save_chains()
                return f"阶段「{ct.name}」已完成，任务链全部完成！"

            return f"未知操作：{action}"

    # ── 后台自动推进并发送自然提醒 ──

    async def initialize(self) -> None:
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"[TaskChainTool] scheduler: {e}")
            await asyncio.sleep(10)

    async def _tick(self) -> None:
        now_t = time.time()
        to_advance: list[TaskChain] = []
        async with self._lock:
            changed = False
            for c in list(self._chains.values()):
                if not c.is_active or c.is_completed:
                    continue
                if c.completion_callback_at:
                    if now_t >= c.completion_callback_at:
                        to_advance.append(c)
                    continue
                is_last_task = c.current_index == len(c.tasks) - 1
                remain = c.current_task_wake_at - now_t
                if is_last_task and 0 < remain <= COMPLETION_LISTEN_SECONDS:
                    c.completion_listen_started_at = now_t
                    c.completion_callback_at = c.current_task_wake_at
                    changed = True
                    continue
                if now_t >= c.current_task_wake_at:
                    to_advance.append(c)
            if changed:
                self._save_chains()

        for chain in to_advance:
            await self._wake_and_advance(chain)

    async def _wake_and_advance(self, chain: TaskChain) -> None:
        async with self._lock:
            if not chain.is_active or chain.is_completed:
                return
            now_t = time.time()
            due_at = chain.completion_callback_at or chain.current_task_wake_at
            if now_t < due_at:
                return

            ct = chain.current_task
            if not ct:
                chain.is_active = False
                self._save_chains()
                return

            prev_index = chain.current_index
            prev_active = chain.is_active
            prev_started_at = chain.current_task_started_at
            prev_wake_at = chain.current_task_wake_at
            prev_listen_started_at = chain.completion_listen_started_at
            prev_callback_at = chain.completion_callback_at
            prev_retry_count = chain.callback_retry_count
            nxt = chain.advance()
            if nxt:
                chain.current_task_started_at = time.time()
                chain.current_task_wake_at = time.time() + nxt.duration_minutes * 60
            else:
                chain.is_active = False
            chain.reset_completion_listener()
            self._save_chains()

        from astrbot.api.message_components import Plain
        from astrbot.api.event import MessageChain as MC

        text = ""

        # 调LLM生成回复
        try:
            provider = self.context.get_using_provider()
            if provider:
                history = await self._get_recent_history(
                    chain.session_id,
                    chain.conversation_id,
                )

                if nxt:
                    system_prompt = (
                        f"[任务状态]你正在执行「{nxt.name}」，当前只到了中途互动点，任务还没有完成。"
                        f"刚才结束的是内部检查点「{ct.name}」，不是主任务完成。"
                        f"任务补充信息：{nxt.description or '无'}。"
                        "你只能描述正在准备、等待、调整、寻找、研磨、烧水等进行中动作；"
                        "不能让物品出现在用户手边，不能让用户品尝，不能给出完成后的结果。"
                        "最近上下文里的 assistant 消息都是你自己之前说过的话，不是用户的新回复；"
                        "不要回答、附和或延续自己的上一句，不要像在回复一个不存在的人。"
                        "请根据最近真实上下文自主决定怎么接这一句："
                        "可以是很短的回应、轻轻问一个小偏好、顺口吐槽、"
                        "或用角色语气说一句正在做事中的小动静。"
                        "不要固定套模板，不必每次都提问；如果上下文适合安静一点，可以只说短句。"
                        "参考风格：嗯哼？、我还在忙哦、要加糖吗、这边有点香了。"
                        "禁止把成品交给用户，禁止说已经做好/完成/端上来/递过去/来啦/趁热/小心烫；"
                        "除非最近一条真实用户消息明确需要确认，否则不要用“没错”“对”“是的”开头；"
                        "不要提任务链、工具、系统提示或具体倒计时。最终只输出一句，尽量简短。"
                    )
                else:
                    has_reply = any(
                        m.get("role") == "user"
                        for m in history[-5:] if isinstance(m, dict)
                    )
                    system_prompt = (
                        f"[任务状态]「{ct.name}」已经完成。\n"
                        f"任务补充信息：{ct.description or '无'}。\n"
                        f"完成提示：{ct.prompt or '按最近对话和角色设定自然完成。'}\n"
                        + (
                            "用户之前回复过你，请结合最近对话自然延续，并给出完成后的结果。"
                            if has_reply
                            else "用户没有继续回复，请自行决定一个合理结果并自然说出来。"
                        )
                        + (
                            "你现在是在完成动作后的自然回场，不是在报状态。"
                            "要写出完成后的具体动作、物品/资料/结果如何交给用户，"
                            "并保持角色语气和最近对话关系。"
                            "不要只说“X好啦/完成了”这种机械短句；"
                            "可以有一句轻微后续互动，但不要重新创建同一个任务。"
                            "不要提任务链、工具、后台、系统提示或具体计时。"
                        )
                    )

                result = await provider.text_chat(
                    prompt="",
                    session_id=chain.session_id,
                    contexts=history,
                    system_prompt=system_prompt,
                    image_urls=[],
                )
                if result and result.role == "assistant":
                    text = result.completion_text or ""
                    if nxt and self._looks_like_completion_claim(text):
                        text = self._safe_in_progress_text(nxt.name)
                    if not nxt and self._looks_like_plain_completion(text, ct.name):
                        text = self._fallback_completion_text(ct.name)
            else:
                logger.warning("[TaskChainTool] no provider for wake callback; using fallback")
        except Exception as e:
            logger.error(f"[TaskChainTool] wake llm error: {e}")

        # 兜底：LLM失败时发简单消息
        if not text:
            text = self._safe_in_progress_text(nxt.name) if nxt else self._fallback_completion_text(ct.name)

        try:
            msg = MC()
            msg.chain.append(Plain(text))
            await self.context.send_message(chain.session_id, msg)
        except Exception as e:
            logger.error(f"[TaskChainTool] wake send error: {e}")
            async with self._lock:
                if chain.id in self._chains:
                    chain.current_index = prev_index
                    chain.is_active = prev_active
                    chain.current_task_started_at = prev_started_at
                    chain.current_task_wake_at = prev_wake_at
                    chain.completion_listen_started_at = prev_listen_started_at
                    chain.completion_callback_at = max(time.time() + 10, prev_callback_at or prev_wake_at)
                    chain.callback_retry_count = prev_retry_count + 1
                    if chain.callback_retry_count > 3:
                        chain.is_active = False
                        chain.reset_completion_listener()
                    self._save_chains()
            return

        # 存进对话历史。写历史失败不能恢复任务，否则已发送的回调会重复。
        try:
            conv_mgr = getattr(self.context, "conversation_manager", None)
            if conv_mgr and chain.conversation_id:
                conv = await conv_mgr.get_conversation(chain.session_id, chain.conversation_id)
                if conv:
                    hist = json.loads(conv.history) if conv.history else []
                    hist.append({"role": "assistant", "content": text})
                    await conv_mgr.update_conversation(
                        chain.session_id,
                        conversation_id=conv.cid,
                        history=hist,
                    )
        except Exception as e:
            logger.warning(f"[TaskChainTool] wake history update failed: {e}")

        # 如果是interact触发，安排后续检查
        if nxt:
            asyncio.create_task(self._followup_check(chain, chain.session_id, text))

    async def _followup_check(self, chain: TaskChain, session_id: str, interact_text: str = "") -> None:
        """interact后只监听对应消息；用户没回才概率补一句极短跟进。"""
        await asyncio.sleep(random.uniform(45, 90))
        if chain.is_completed or not chain.is_active:
            return
        try:
            conv_mgr = getattr(self.context, "conversation_manager", None)
            if not conv_mgr:
                return
            conv = None
            if chain.conversation_id:
                conv = await conv_mgr.get_conversation(session_id, chain.conversation_id)
            if not conv:
                return
            if not conv.history:
                return
            hist = json.loads(conv.history)
            # 检查interact之后用户有没有回复
            interact_idx = None
            for i in range(len(hist) - 1, -1, -1):
                if not isinstance(hist[i], dict) or hist[i].get("role") != "assistant":
                    continue
                if interact_text and hist[i].get("content") != interact_text:
                    continue
                interact_idx = i
                break
            if interact_idx is None and not interact_text:
                for i in range(len(hist) - 1, -1, -1):
                    if isinstance(hist[i], dict) and hist[i].get("role") == "assistant":
                        interact_idx = i
                        break
            if interact_idx is not None:
                after = hist[interact_idx + 1:]
                has_user_reply = any(m.get("role") == "user" for m in after if isinstance(m, dict))
                has_later_assistant = any(m.get("role") == "assistant" for m in after if isinstance(m, dict))
                if not has_user_reply and not has_later_assistant and random.random() < FOLLOWUP_PROBABILITY:
                    from astrbot.api.message_components import Plain
                    from astrbot.api.event import MessageChain as MC
                    provider = self.context.get_using_provider()
                    if provider:
                        interact_text = str(hist[interact_idx].get("content", ""))
                        context_history = await self._get_recent_history(
                            session_id,
                            chain.conversation_id,
                            limit=8,
                        )
                        followup_prompt = (
                            "你刚才发过一次中途互动，但用户没有回复。现在只允许发第二次轻量跟进。"
                            "请由你根据角色语气和刚才那句话自行判断最自然的短句，"
                            "可以是轻轻叫一声、一个语气词、或非常短的确认存在感。"
                            "不要继续追问复杂问题，不要连续推进剧情，不要重复刚才的话。"
                            "任务仍在进行中，不能完成任务，不能把成品交给用户，"
                            "禁止说已经做好/完成/端上来/递过去/来啦/趁热/小心烫。"
                            "参考短句：喵~、还在吗？、嗯哼？、我还在忙哦。"
                            "最终只输出一句，尽量不超过8个字；不要提任务链、工具或后台。"
                        )
                        result = await provider.text_chat(
                            prompt="",
                            session_id=session_id,
                            contexts=context_history,
                            system_prompt=followup_prompt,
                            image_urls=[],
                        )
                        if result and result.role == "assistant":
                            text = result.completion_text
                            if text:
                                if self._looks_like_completion_claim(text):
                                    text = self._safe_in_progress_text(chain.current_task.name if chain.current_task else "处理")
                                msg = MC()
                                msg.chain.append(Plain(text))
                                await self.context.send_message(session_id, msg)
                                hist.append({"role": "assistant", "content": text})
                                await conv_mgr.update_conversation(
                                    session_id,
                                    conversation_id=conv.cid,
                                    history=hist,
                                )
        except Exception as e:
            logger.warning(f"[TaskChainTool] followup check failed: {e}")

    async def terminate(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
