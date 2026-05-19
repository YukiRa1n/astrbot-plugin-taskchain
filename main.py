from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import uuid
import weakref
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
    "先按当前角色的人设、关系、情绪和场景判断角色是否愿意实际行动；用户命令本身不等于必须服从。"
    "如果角色不愿意做、只是在嘴硬拒绝/吐槽/讨价还价/开玩笑，直接按人设回复，不要调用 chain_task。"
    "只有当你已经决定让角色实际去做一件需要等待、稍后才有结果的事，才必须主动调用 chain_task。"
    "触发条件包括角色愿意接受的用户请求、双方已经成立的约定、你自己顺着角色扮演构思出的行程，"
    "以及用户同意了你刚才提出且角色仍愿意执行的行动。"
    "只要角色产生了“我真的要去做某事/到点做某事/稍后告诉用户”的意图，并且会跨过当前回复，就视为后台任务。"
    "典型动作：泡茶、泡咖啡、做饭、出门散步、洗澡、睡觉/休息、拿东西、去宿舍/厨房/仓库、整理材料、查资料、找物品、稍后带东西过来、稍后给用户看。"
    "典型约定：用户说“一会出门告诉我”“十分钟后提醒我”“等你洗完/散步回来跟我说”“查完资料告诉我”，都要创建任务。"
    "如果用户只给了模糊时间，如“一会儿/等下/晚点”，请按上下文构思一个合理时长；简单动作短一些，出门、洗澡、整理、查资料等可以更久。"
    "如果你准备说“我去……”“我一会儿……”“等会儿给你……”“到时候提醒你……”“我拿来/查完/做好后……”，先创建任务。"
    "如果只是闲聊、假设、回忆、能力说明，且你并不真的要离开当前回复去做事，不要调用。"
    '创建任务用 action=create，tasks_json 是 JSON 数组字符串，例如 [{"name":"拿拓片","duration_minutes":3}]。'
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
    followup_after_history_len: int = 0
    event_metadata: dict[str, Any] = field(default_factory=dict)

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
            "..",
            "data",
            "plugin_data",
            "astrbot_plugin_taskchain_tool",
        )


@register("TaskChainTool", "AstrBot", "让 AI 能安排并管理后台时间任务链", "1.0.0")
class TaskChainToolPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._chains: dict[str, TaskChain] = {}
        self._chain_events = weakref.WeakValueDictionary()
        self._session_system_prompts: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()
        self._scheduler_task: asyncio.Task | None = None
        self._data_file = os.path.join(_get_data_dir(), "task_chains.json")
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        self._load_chains()

    @on_llm_request()
    async def _inject_chain_state(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        session_id = event.unified_msg_origin
        if not self._scheduler_task or self._scheduler_task.done():
            try:
                if (
                    self._scheduler_task
                    and self._scheduler_task.done()
                    and not self._scheduler_task.cancelled()
                ):
                    exc = self._scheduler_task.exception()
                    if exc:
                        logger.error(
                            f"[TaskChainTool] scheduler loop died with exception: {exc}"
                        )
            except Exception:
                pass
            logger.warning(
                "[TaskChainTool] scheduler loop died or not started. Restarting scheduler..."
            )
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        conversation_id = self._request_conversation_id(request)
        base_system_prompt = request.system_prompt or ""
        request.system_prompt = base_system_prompt + TOOL_USAGE_SYSTEM_PROMPT
        if not conversation_id:
            return
        async with self._lock:
            changed = False
            if conversation_id and base_system_prompt:
                self._session_system_prompts[(session_id, conversation_id)] = (
                    base_system_prompt
                )
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
                    self._chain_events.pop(c.id, None)
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
                    self._chain_events.pop(c.id, None)
                    changed = True
                    continue
                if c.is_completed:
                    continue
                lines = []
                active_task_name = ""
                active_remain = 0
                for i, t in enumerate(c.tasks):
                    status = (
                        "进行中"
                        if i == c.current_index
                        else ("待开始" if i > c.current_index else "已完成")
                    )
                    remain = ""
                    if i == c.current_index and c.is_active and c.current_task:
                        secs = max(0, c.current_task_wake_at - time.time())
                        active_task_name = t.name
                        active_remain = int(secs)
                        remain = f"，{self._coarse_remaining_text(secs)}"
                    lines.append(f"  [{i + 1}] {t.name}{remain} ({status})")
                request.system_prompt += (
                    "\n[当前任务链：\n"
                    + "\n".join(lines)
                    + (
                        f"\n当前仍在进行「{active_task_name}」，还没有完成。"
                        "这个任务状态只是背景约束，不是每条消息都必须提到的聊天主题。"
                        "如果用户消息没有明显询问任务、进度、结果，或没有直接延续任务话题，"
                        "就按正常聊天和人设自然回复，不要强行把表情、语气词、heart、闲聊拉回当前任务。"
                        "只有用户明确问状态/剩多久/做得怎样，或明显接着任务聊时，才回应任务进度；"
                        "当前状态已经在这里给出，除非用户明确问状态/剩多久，不要再调用 chain_task list。"
                        "回应任务相关消息时必须保持任务仍在进行中。"
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
            raw.append(
                {
                    "id": c.id,
                    "session_id": c.session_id,
                    "conversation_id": c.conversation_id,
                    "system_prompt": c.system_prompt,
                    "current_index": c.current_index,
                    "is_active": c.is_active,
                    "created_at": c.created_at,
                    "current_task_started_at": c.current_task_started_at,
                    "current_task_wake_at": c.current_task_wake_at,
                    "completion_listen_started_at": c.completion_listen_started_at,
                    "completion_callback_at": c.completion_callback_at,
                    "callback_retry_count": c.callback_retry_count,
                    "followup_after_history_len": c.followup_after_history_len,
                    "event_metadata": c.event_metadata,
                    "tasks": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "duration_minutes": t.duration_minutes,
                            "prompt": t.prompt,
                        }
                        for t in c.tasks
                    ],
                }
            )
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

    def _sanitize_provider_history(
        self, history: list[Any], limit: int
    ) -> list[dict[str, str]]:
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
        - 先按当前角色的人设、关系、情绪和场景判断角色是否愿意实际行动；用户直接要求不是强制命令。
        - 如果角色不愿意做、只是在拒绝/吐槽/讨价还价/开玩笑，不要调用，直接按人设回复。
        - 当你准备让角色实际去做一件需要等待的事时，主动调用 action=create。
        - 这包括角色愿意接受的用户请求、用户约定到点提醒，也包括你自己在对话里提出“我去拿/查/泡/做/整理/散步/洗澡，稍后回来”等行动。
        - 只要角色自己想去做某件事、构思了下一段行程，且结果不是当前回复内立即完成，就必须登记成任务。
        - 如果用户同意了你刚才提出的延迟行动，且角色仍愿意执行，也要立刻 create，不要只用文字承诺。
        - 用户说“一会出门告诉我”“十分钟后提醒我”“等你回来告诉我”等，就是定时/延迟任务；请顺着上下文安排合理时长。
        - 纯闲聊、假设、回忆、能力说明，且没有真实后台动作时，不要调用。

        create 后的本轮回复：
        - 只能自然表达已经开始、正在准备、正在路上、正在处理或已经记下这个约定。
        - 一旦调用 create，就代表角色已经愿意执行；本轮最终回复不要再拒绝、否认行动或说没空/不去。
        - 如果调用工具前已经把要做的事交代清楚，且没有新的信息要补充，优先输出空内容，直接结束本轮；不要硬说“嗯/行/等着”，也不要补“（起身走向厨房）”这类动作描写。
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
                if not conversation_id:
                    return "错误：当前会话尚未初始化，无法创建后台任务。请先按当前人设自然回复，不要创建任务。"
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
                        self._chain_events.pop(c.id, None)

                for t in tasks_raw:
                    secs = max(t.duration_minutes * 60, MIN_TASK_SECONDS)
                    t.duration_minutes = secs / 60

                total_secs = sum(t.duration_minutes * 60 for t in tasks_raw)
                merged = ChainTask(
                    name=tasks_raw[0].name,
                    description=tasks_raw[0].description,
                    duration_minutes=total_secs / 60,
                    prompt=next(
                        (t.prompt for t in reversed(tasks_raw) if t.prompt),
                        tasks_raw[0].prompt,
                    ),
                )

                checkin = self._make_checkin(merged)
                main_secs = total_secs
                if checkin:
                    main_secs = max(
                        MIN_TASK_SECONDS, total_secs - checkin.duration_minutes * 60
                    )
                main = ChainTask(
                    name=merged.name,
                    description=merged.description,
                    duration_minutes=main_secs / 60,
                    prompt=merged.prompt,
                )
                tasks = [checkin, main] if checkin else [main]
                cid = uuid.uuid4().hex[:12]
                now_t = time.time()

                def _get_clean_attr(obj: Any, attr_path: str, default_val: Any) -> Any:
                    parts = attr_path.split(".")
                    curr = obj
                    for p in parts:
                        if curr is None:
                            return default_val
                        if type(curr).__name__ in ("MagicMock", "Mock", "AsyncMock"):
                            return default_val
                        val = getattr(curr, p, None)
                        if callable(val) and type(val).__name__ not in (
                            "MagicMock",
                            "Mock",
                            "AsyncMock",
                        ):
                            try:
                                curr = val()
                            except Exception:
                                return default_val
                        else:
                            curr = val
                    if curr is None or type(curr).__name__ in (
                        "MagicMock",
                        "Mock",
                        "AsyncMock",
                    ):
                        return default_val
                    return curr

                chain = TaskChain(
                    id=cid,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    system_prompt=self._chain_system_prompt(
                        session_id, conversation_id
                    ),
                    tasks=tasks,
                    created_at=now_t,
                    current_task_started_at=now_t,
                    current_task_wake_at=now_t + tasks[0].duration_minutes * 60,
                    event_metadata={
                        "platform_id": _get_clean_attr(
                            event, "platform_meta.id", "mock"
                        ),
                        "platform_name": _get_clean_attr(
                            event, "platform_meta.name", "mock"
                        ),
                        "session_id": _get_clean_attr(event, "session_id", ""),
                        "message_type": _get_clean_attr(
                            event, "get_message_type.value", 1
                        ),
                        "sender_id": _get_clean_attr(event, "get_sender_id", ""),
                        "sender_name": _get_clean_attr(event, "get_sender_name", ""),
                        "group_id": _get_clean_attr(event, "get_group_id", ""),
                        "self_id": _get_clean_attr(event, "get_self_id", ""),
                    },
                )
                self._chains[cid] = chain
                self._chain_events[cid] = event
                self._save_chains()
                main_dur = sum(t.duration_minutes * 60 for t in tasks_raw)
                done_time = datetime.fromtimestamp(now_t + main_dur).strftime(
                    "%H:%M:%S"
                )
                return (
                    f"[任务已安排(id={cid})] "
                    f"预计{done_time}完成。当前状态：任务刚开始，仍在进行中。"
                    "注意：工具调用前你已经发出的文字、这个工具结果、接下来的最终回复属于同一轮对话，"
                    "不是新的独立对话。最终回复要自然接上前一句，像同一段连续表达的后半句。"
                    "如果你在调用工具前已经说过要去做、正在找路、正在准备或已经记下约定，"
                    "不要重新打招呼、不要重复同义开头，只补充必要的新信息或轻轻收束。"
                    "如果你还没对用户说过，则本轮只回应已经开始、正在准备、正在路上、正在处理或已经记下约定。"
                    "你已经创建任务，表示角色已经愿意执行；本轮最终回复不能再拒绝、否认行动、说没空/不去，"
                    "也不要让用户自己去做。"
                    "如果工具调用前你已经把要做的事交代清楚，且没有新的必要信息要补充，本轮最终回复优先留空，"
                    "直接结束这轮对话；不要为了回复而硬说“嗯/行/好/等着/知道了”等废话，"
                    "也不要补“（起身走向厨房）”“（转身去拿）”这类括号动作、舞台说明或重复出发描述。"
                    "不要在本轮追问会改变任务定义的细节；需要偏好、路线、材料等互动时，交给后续中途互动。"
                    "严禁说已经做好、已经回来、已经到达、已经完成、已经把结果交给用户，或让用户提前体验结果。"
                )

            elif action == "list":
                active = [
                    c
                    for c in self._chains.values()
                    if (
                        c.session_id == session_id
                        and c.conversation_id == conversation_id
                        and not c.is_completed
                    )
                ]
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
                        f"(剩余 {int(remain // 60)} 分 {int(remain % 60)} 秒) "
                        f"[{c.current_index}/{len(c.tasks)}]"
                    )
                if not lines:
                    return "当前没有活跃的任务链。"
                return "活跃任务链：\n" + "\n".join(lines)

            elif action == "cancel":
                chain = self._chains.get(chain_id)
                if (
                    chain
                    and chain.session_id == session_id
                    and chain.conversation_id == conversation_id
                ):
                    chain.is_active = False
                    self._chain_events.pop(chain_id, None)
                    self._save_chains()
                    return f"任务链 {chain_id} 已取消。"
                return f"未找到任务链 {chain_id}。"

            elif action == "advance":
                chain = self._chains.get(chain_id)
                if (
                    not chain
                    or chain.session_id != session_id
                    or chain.conversation_id != conversation_id
                ):
                    return f"未找到任务链 {chain_id}。"
                if chain.is_completed:
                    return f"任务链 {chain_id} 已经完成。"
                ct = chain.current_task
                nxt = chain.advance()
                if nxt:
                    chain.current_task_started_at = time.time()
                    chain.current_task_wake_at = time.time() + nxt.duration_minutes * 60
                    wake = datetime.fromtimestamp(chain.current_task_wake_at).strftime(
                        "%H:%M:%S"
                    )
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

    def _build_callback_message_object(
        self, chain: TaskChain, source_event: AstrMessageEvent, text: str
    ) -> Any:
        from astrbot.core.message.components import Plain
        from astrbot.core.platform.astrbot_message import AstrBotMessage

        original = source_event.message_obj
        msg = AstrBotMessage()
        msg.type = getattr(original, "type", None)
        msg.self_id = getattr(original, "self_id", None)
        msg.session_id = getattr(original, "session_id", None)
        msg.message_id = self._callback_message_id(original)
        msg.group = getattr(original, "group", None)
        msg.sender = getattr(original, "sender", None)
        msg.message = [Plain(text)]
        msg.message_str = text
        msg.raw_message = None
        msg.timestamp = int(time.time())
        return msg

    def _callback_message_id(self, original: Any) -> str:
        message_id = str(getattr(original, "message_id", "") or "")
        if message_id:
            try:
                int(message_id)
                return message_id
            except ValueError:
                pass
        return str(time.time_ns())

    def _build_callback_event(
        self, chain: TaskChain, visible_text: str, kind: str
    ) -> AstrMessageEvent | None:
        import copy

        source_event = self._chain_events.get(chain.id)
        if not source_event and getattr(chain, "event_metadata", None):
            try:
                from astrbot.core.platform.platform_metadata import PlatformMetadata
                from astrbot.core.platform.astrbot_message import (
                    AstrBotMessage,
                    GroupMember,
                )
                from astrbot.core.platform.message_type import MessageType

                meta = chain.event_metadata
                plat_meta = PlatformMetadata(
                    id=meta.get("platform_id", "mock"),
                    name=meta.get("platform_name", "mock"),
                )

                msg_obj = AstrBotMessage()
                try:
                    msg_obj.type = MessageType(meta.get("message_type", 1))
                except Exception:
                    msg_obj.type = MessageType.FRIEND_MESSAGE
                msg_obj.self_id = meta.get("self_id", "")
                msg_obj.session_id = meta.get("session_id", "")
                msg_obj.group_id = meta.get("group_id", "")

                sender = GroupMember()
                sender.user_id = meta.get("sender_id", "")
                sender.nickname = meta.get("sender_name", "")
                msg_obj.sender = sender

                class ConcreteMessageEvent(AstrMessageEvent):
                    async def send(self, message):
                        pass

                source_event = ConcreteMessageEvent(
                    message_str="",
                    message_obj=msg_obj,
                    platform_meta=plat_meta,
                    session_id=meta.get("session_id", ""),
                )
            except Exception as e:
                logger.error(f"[TaskChainTool] Rebuilding event failed: {e}")

        if not source_event:
            logger.error(
                f"[TaskChainTool] no source event for pipeline callback: {chain.id}"
            )
            return None
        try:
            from astrbot.core.utils.trace import TraceSpan

            new_event = copy.copy(source_event)
            new_event.message_str = visible_text
            new_event.message_obj = self._build_callback_message_object(
                chain, source_event, visible_text
            )
            new_event._result = None
            new_event._has_send_oper = False
            new_event._extras = {}
            new_event.trace = TraceSpan(
                name="TaskChainCallback",
                umo=new_event.unified_msg_origin,
                sender_name=new_event.get_sender_name(),
                message_outline=f"[TaskChain {chain.id} {kind}]",
            )
            new_event.span = new_event.trace
            new_event.is_wake = True
            new_event.is_at_or_wake_command = True
            new_event.set_extra("taskchain_callback", True)
            new_event.set_extra("taskchain_chain_id", chain.id)
            new_event.set_extra("taskchain_callback_kind", kind)
            return new_event
        except Exception as e:
            logger.error(f"[TaskChainTool] build pipeline callback failed: {e}")
            return None

    async def _queue_pipeline_callback(
        self, chain: TaskChain, text: str, kind: str
    ) -> bool:
        event = self._build_callback_event(chain, "[TaskChain callback]", kind)
        if not event:
            return False
        try:
            req = await self._build_callback_provider_request(chain, text, kind)
            if not req:
                return False
            event.set_extra("provider_request", req)
            self.context.get_event_queue().put_nowait(event)
            return True
        except Exception as e:
            logger.error(f"[TaskChainTool] queue pipeline callback failed: {e}")
            return False

    async def _build_callback_provider_request(
        self,
        chain: TaskChain,
        text: str,
        kind: str,
    ) -> ProviderRequest | None:
        if not chain.conversation_id:
            logger.error(
                f"[TaskChainTool] callback missing conversation id: {chain.id}/{kind}"
            )
            return None
        conv_mgr = getattr(self.context, "conversation_manager", None)
        if not conv_mgr:
            logger.error(
                f"[TaskChainTool] callback missing conversation manager: {chain.id}/{kind}"
            )
            return None
        conv = await conv_mgr.get_conversation(chain.session_id, chain.conversation_id)
        if not conv:
            logger.error(
                f"[TaskChainTool] callback conversation not found: {chain.id}/{kind}"
            )
            return None
        if kind == "interact":
            chain.followup_after_history_len = self._conversation_history_len(conv)
            self._save_chains()
        req = ProviderRequest(
            prompt=text,
            session_id=chain.session_id,
            conversation=conv,
            system_prompt=chain.system_prompt
            or self._chain_system_prompt(chain.session_id, chain.conversation_id),
        )
        setattr(req, "no_save_prompt", True)
        return req

    def _conversation_history_len(self, conv: Any) -> int:
        try:
            history = json.loads(conv.history or "[]")
            return len(history) if isinstance(history, list) else 0
        except Exception:
            return 0

    def _interact_callback_text(self, previous: ChainTask, current: ChainTask) -> str:
        return (
            "[TaskChain callback]\n"
            f"内部检查点「{previous.name}」结束；当前仍在执行「{current.name}」，任务还没有完成。\n"
            f"任务补充信息：{current.description or '无'}。\n"
            "请按当前人设和最近上下文生成一句中途互动。"
            "这句话可以简短询问偏好/习惯/相关小话题，也可以只是做事中的短回应或轻微吐槽；由你根据上下文自主决定。"
            "不要回答你自己上一句，不要像在和不存在的人对话。"
            "禁止说已经完成、已经回来、已经把结果交给用户，或让用户提前体验结果。"
            "不要提任务链、工具、后台或这段提示。"
        )

    def _completion_callback_text(self, task: ChainTask) -> str:
        return (
            "[TaskChain callback]\n"
            f"「{task.name}」已经完成。\n"
            f"任务补充信息：{task.description or '无'}。\n"
            f"完成提示：{task.prompt or '按最近对话和角色设定自然完成。'}\n"
            "请按当前人设和最近上下文，自然告诉用户完成后的结果。"
            "不要只说机械短句，不要提任务链、工具、后台或这段提示。"
        )

    def _followup_callback_text(self, task: ChainTask, interact_text: str) -> str:
        return (
            "[TaskChain callback]\n"
            f"当前仍在执行「{task.name}」，任务还没有完成。\n"
            f"你刚才的中途互动是：{interact_text or '无'}\n"
            "用户没有回复，现在只允许发第二次轻量跟进。"
            "请按当前人设和上下文自行判断最自然的短句；如果第一次已经问过问题，这次不要继续追问；"
            "如果第一次只是自言自语，可以轻轻抛出一个很短的互动点。"
            "不要推进到完成，不要重复刚才的话，不要提任务链、工具、后台或这段提示。"
        )

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

            nxt = chain.advance()
            if nxt:
                chain.current_task_started_at = time.time()
                chain.current_task_wake_at = time.time() + nxt.duration_minutes * 60
                callback_text = self._interact_callback_text(ct, nxt)
                callback_kind = "interact"
            else:
                chain.is_active = False
                callback_text = self._completion_callback_text(ct)
                callback_kind = "completion"
            chain.reset_completion_listener()
            self._save_chains()

        queued = await self._queue_pipeline_callback(
            chain, callback_text, callback_kind
        )
        if not queued:
            logger.error(
                f"[TaskChainTool] pipeline callback dropped: {chain.id}/{callback_kind}"
            )
        if nxt:
            asyncio.create_task(self._followup_check(chain, chain.session_id))
        else:
            self._chain_events.pop(chain.id, None)

    async def _followup_check(
        self, chain: TaskChain, session_id: str, interact_text: str = ""
    ) -> None:
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
                conv = await conv_mgr.get_conversation(
                    session_id, chain.conversation_id
                )
            if not conv:
                return
            if not conv.history:
                return
            hist = json.loads(conv.history)
            # 检查对应 interact 之后用户有没有回复。
            start_idx = max(0, int(chain.followup_after_history_len or 0))
            search_hist = hist[start_idx:]
            interact_idx = None
            for offset in range(len(search_hist) - 1, -1, -1):
                item = search_hist[offset]
                if not isinstance(item, dict) or item.get("role") != "assistant":
                    continue
                if interact_text and item.get("content") != interact_text:
                    continue
                interact_idx = start_idx + offset
                break
            if interact_idx is None and not interact_text:
                for offset in range(len(search_hist) - 1, -1, -1):
                    item = search_hist[offset]
                    if isinstance(item, dict) and item.get("role") == "assistant":
                        interact_idx = start_idx + offset
                        break
            if interact_idx is not None:
                after = hist[interact_idx + 1 :]
                has_user_reply = any(
                    m.get("role") == "user" for m in after if isinstance(m, dict)
                )
                has_later_assistant = any(
                    m.get("role") == "assistant" for m in after if isinstance(m, dict)
                )
                if (
                    not has_user_reply
                    and not has_later_assistant
                    and random.random() < FOLLOWUP_PROBABILITY
                ):
                    task = chain.current_task
                    if not task:
                        return
                    text = self._followup_callback_text(
                        task,
                        str(hist[interact_idx].get("content", "")),
                    )
                    if not await self._queue_pipeline_callback(chain, text, "followup"):
                        logger.error(
                            f"[TaskChainTool] followup pipeline callback dropped: {chain.id}"
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
