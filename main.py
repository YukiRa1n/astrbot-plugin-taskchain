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
    "当你要让角色实际去做一件需要等待、稍后才有结果的事，必须主动调用 chain_task。"
    "触发条件包括用户命令、用户约定、你自己顺着角色扮演构思出的行程，"
    "以及用户同意了你刚才提出的行动。"
    "只要角色产生了“我要去做某事/到点做某事/稍后告诉用户”的意图，并且会跨过当前回复，就视为后台任务。"
    "典型动作：泡茶、泡咖啡、做饭、出门散步、洗澡、睡觉/休息、拿东西、去宿舍/厨房/仓库、整理材料、查资料、找物品、稍后带东西过来、稍后给用户看。"
    "典型约定：用户说“一会出门告诉我”“十分钟后提醒我”“等你洗完/散步回来跟我说”“查完资料告诉我”，都要创建任务。"
    "如果用户只给了模糊时间，如“一会儿/等下/晚点”，请按上下文构思一个合理时长；简单动作短一些，出门、洗澡、整理、查资料等可以更久。"
    "如果你准备说“我去……”“我一会儿……”“等会儿给你……”“到时候提醒你……”“我拿来/查完/做好后……”，先创建任务。"
    "如果只是闲聊、假设、回忆、能力说明，且你并不真的要离开当前回复去做事，不要调用。"
    "创建任务用 action=create，tasks_json 是 JSON 数组字符串，例如 [{\"name\":\"拿拓片\",\"duration_minutes\":3}]。"
    "duration_minutes 按任务真实体感设置：简单动作可短，查资料、整理材料、出门取物等长时间状态任务可以设置更久。"
    "创建后只自然回应正在开始/正在准备，不要直接演到完成。]"
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
    system_prompt: str = ""
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
        self._session_system_prompts: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()
        self._scheduler_task: asyncio.Task | None = None
        self._data_file = os.path.join(_get_data_dir(), "task_chains.json")
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        self._load_chains()

    @on_llm_request()
    async def _inject_chain_state(self, event: AstrMessageEvent, request: ProviderRequest) -> None:
        session_id = event.unified_msg_origin
        conversation_id = self._request_conversation_id(request)
        base_system_prompt = request.system_prompt or ""
        request.system_prompt = base_system_prompt + TOOL_USAGE_SYSTEM_PROMPT
        async with self._lock:
            changed = False
            if conversation_id and base_system_prompt:
                self._session_system_prompts[(session_id, conversation_id)] = base_system_prompt
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
                        remain = f"，{self._coarse_remaining_text(secs)}"
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
                "system_prompt": c.system_prompt,
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

    def _chain_system_prompt(self, session_id: str, conversation_id: str) -> str:
        if not conversation_id:
            return ""
        return self._session_system_prompts.get((session_id, conversation_id), "")

    def _with_chain_system_prompt(self, chain: TaskChain, task_prompt: str) -> str:
        base = (chain.system_prompt or "").strip()
        if not base:
            return task_prompt
        return f"{base}\n\n{task_prompt}"

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

    def _coarse_remaining_text(self, secs: float) -> str:
        if secs <= 0:
            return "等待完成回调"
        if secs < 60:
            return "不到1分钟"
        minutes = max(1, int(round(secs / 60)))
        return f"约{minutes}分钟"

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
            "趁热", "小心烫",
        )
        return any(w in text for w in completion_words)

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
        if not self.config.get("interact_enabled", False):
            return None
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
        """安排或查看角色正在做的后台时间任务。

        使用时机：
        - 当你准备让角色实际去做一件需要等待的事时，主动调用 action=create。
        - 这包括用户直接要求、用户约定到点提醒，也包括你自己在对话里提出“我去拿/查/泡/做/整理/散步/洗澡，稍后回来”等行动。
        - 只要角色自己想去做某件事、构思了下一段行程，且结果不是当前回复内立即完成，就必须登记成任务。
        - 如果用户同意了你刚才提出的延迟行动，也要立刻 create，不要只用文字承诺。
        - 用户说“一会出门告诉我”“十分钟后提醒我”“等你回来告诉我”等，就是定时/延迟任务；请顺着上下文安排合理时长。
        - 纯闲聊、假设、回忆、能力说明，且没有真实后台动作时，不要调用。

        create 后的本轮回复：
        - 只能自然表达已经开始、正在准备、正在路上、正在处理或已经记下这个约定。
        - 不要说任务已完成，不要把结果提前交给用户，不要提前描述已经回来/已经做完。
        - 刚 create 后避免追问会改变任务定义的细节；需要偏好、路线、材料等互动时，交给后续中途互动。

        查看状态时使用 action=list；取消任务时使用 action=cancel。不要无意义重复调用。

        Args:
            action(string): create 创建任务；list 查看当前会话任务；cancel 取消任务。默认 list。
            tasks_json(string): action=create 时必填，JSON数组字符串。通常只放1个任务，如 [{"name":"出门散步","duration_minutes":30,"description":"用户让你一会出门时告诉他","prompt":"到时间后自然告诉用户准备出门或已经回到约定节点"}]。duration_minutes 按任务体感设置；泡茶/拿小物可短，洗澡、散步、查资料、整理、取物、外出等可以更久。
            chain_id(string): action=cancel 时必填。
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
                    id=cid,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    system_prompt=self._chain_system_prompt(session_id, conversation_id),
                    tasks=tasks,
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
                "注意：工具调用前你已经发出的文字、这个工具结果、接下来的最终回复属于同一轮对话，"
                "不是新的独立对话。最终回复要自然接上前一句，像同一段连续表达的后半句。"
                "如果你在调用工具前已经说过要去做、正在找路、正在准备或已经记下约定，"
                "不要重新打招呼、不要重复同义开头，只补充必要的新信息或轻轻收束。"
                "如果你还没对用户说过，则本轮只回应已经开始、正在准备、正在路上、正在处理或已经记下约定。"
                "不要在本轮追问会改变任务定义的细节；需要偏好、路线、材料等互动时，交给后续中途互动。"
                "严禁说已经做好、已经回来、已经到达、已经完成、已经把结果交给用户，或让用户提前体验结果。"
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
                    system_prompt = self._with_chain_system_prompt(chain, (
                        f"[任务状态]你正在执行「{nxt.name}」，现在只是中途互动点，任务还没有完成。"
                        f"刚才结束的是内部检查点「{ct.name}」，不是主任务完成。"
                        f"任务补充信息：{nxt.description or '无'}。"
                        "你只能描写正在准备、等待、调整、寻找、研磨、烧水等进行中动作；"
                        "不能让成品出现在用户手边，不能让用户品尝，不能给出完成后的结果。"
                        "最近上下文里的 assistant 消息都是你自己之前说过的话，不是用户的新回复；"
                        "不要回答、附和或延续自己的上一句，不要像在和不存在的人对话。"
                        "请根据真实上下文自主决定这一句："
                        "如果适合互动，可以轻问偏好、习惯或相关小话题；"
                        "如果不适合提问，可以只说一句做事中的小动静、轻吐槽或短回应。"
                        "不要固定套模板，不必每次提问；保持角色语气，简短自然。"
                        "禁止把成品交给用户，禁止说已经做好/完成/端上来/递过去/来啦/趁热/小心烫；"
                        "除非最近一条真实用户消息明确需要确认，否则不要用“没错”“对”“是的”开头；"
                        "不要提任务链、工具、系统提示或具体倒计时。最终只输出一句，尽量简短。"
                    ))
                else:
                    has_reply = any(
                        m.get("role") == "user"
                        for m in history[-5:] if isinstance(m, dict)
                    )
                    system_prompt = self._with_chain_system_prompt(chain, (
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
                            "请写出完成后的具体动作、物品/资料/结果如何呈现给用户，"
                            "并承接最近对话关系。"
                            "不要只说“X好啦/完成了”这种机械短句；也不要过度铺陈。"
                            "可以带一句轻微后续互动，但不要重新创建同一个任务。"
                            "不要提任务链、工具、后台、系统提示或具体计时。"
                        )
                    ))

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
                        logger.info(f"[TaskChainTool] interact completion claim blocked: {text[:120]}")
                        text = ""
                    if not nxt and self._looks_like_plain_completion(text, ct.name):
                        logger.info(f"[TaskChainTool] plain completion callback returned by provider: {text[:120]}")
            else:
                logger.warning("[TaskChainTool] no provider for wake callback")
        except Exception as e:
            logger.error(f"[TaskChainTool] wake llm error: {e}")

        if not text:
            if nxt:
                logger.info("[TaskChainTool] skip empty/invalid interact callback")
                return
            logger.warning("[TaskChainTool] empty completion callback; restore chain for retry")
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
        if not self.config.get("interact_enabled", False):
            return
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
                        followup_prompt = self._with_chain_system_prompt(chain, (
                            "你刚才发过一次中途互动，但用户没有回复。现在只允许发第二次轻量跟进。"
                            "请根据角色语气和刚才那句话自行判断最自然的短句，"
                            "可以是轻轻叫一声、一个语气词、一个做事中的小声反应，或非常短的确认存在感。"
                            "如果第一次已经问过用户问题，这次不要继续追问；"
                            "如果第一次只是自言自语，可以轻轻抛出一个很短的互动点。"
                            "不要连续推进剧情，不要重复刚才的话。"
                            "任务仍在进行中，不能完成任务，不能把成品交给用户，"
                            "禁止说已经做好/完成/端上来/递过去/来啦/趁热/小心烫。"
                            "最终只输出一句，尽量不超过8个字；不要提任务链、工具或后台。"
                        ))
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
                                    logger.info(f"[TaskChainTool] followup completion claim blocked: {text[:120]}")
                                    return
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
