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

from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import (
    after_message_sent,
    on_decorating_result,
    on_llm_request,
)
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.platform.message_session import MessageSession

COMPLETION_LISTEN_SECONDS = 10
FOLLOWUP_PROBABILITY = 0.5
MIN_TASK_SECONDS = 10
CALLBACK_ACK_TIMEOUT_SECONDS = 120
MAX_CALLBACK_PIPELINE_ATTEMPTS = 3
PROMPT_CACHE_TTL_SECONDS = 3600

_PRE_TOOL_REPLY_KEY = "taskchain_pre_tool_reply"
_CREATE_SUCCEEDED_KEY = "taskchain_create_succeeded"
_SUPPRESS_POST_REPLY_KEY = "taskchain_suppress_post_reply"
_CALLBACK_TOKEN_KEY = "taskchain_callback_token"
_SINGLE_REPLY_MODE_ENABLED = True

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
    "同一轮只能向用户展示一段自然语言：更推荐先静默调用工具，再在工具成功后完整回复一次；"
    "如果调用工具前已经输出过自然回复，工具成功后必须结束本轮，不要再输出确认、同义改写或动作描写。"
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
    callback_retry_at: float = 0.0
    pending_callback_kind: str = ""
    pending_callback_token: str = ""
    followup_after_history_len: int = 0
    finished_at: float = 0.0
    event_metadata: dict[str, Any] = field(default_factory=dict)

    def reset_completion_listener(self) -> None:
        self.completion_listen_started_at = 0.0
        self.completion_callback_at = 0.0

    def reset_callback_delivery(self) -> None:
        self.callback_retry_count = 0
        self.callback_retry_at = 0.0
        self.pending_callback_kind = ""
        self.pending_callback_token = ""

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


class _PrepareSingleReplyFilter(filter.CustomFilter):
    """在唤醒判定阶段关闭流式输出，但不激活消息处理器。"""

    def filter(self, event: AstrMessageEvent, _config: Any) -> bool:
        if _SINGLE_REPLY_MODE_ENABLED:
            event.set_extra("enable_streaming", False)
        # 只利用过滤阶段准备事件；False 可避免插件因此唤醒所有群消息。
        return False


class _ReliableCronMessageEvent(CronMessageEvent):
    """让主动发送失败可被确认钩子识别。"""

    async def send(self, message) -> None:
        if message is None:
            return
        sent = await self.context_obj.send_message(self.session, message)
        if not sent:
            self.set_extra("taskchain_delivery_failed", True)
            raise RuntimeError(f"cannot send callback to session {self.session}")
        await AstrMessageEvent.send(self, message)


class TaskChainToolPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        global _SINGLE_REPLY_MODE_ENABLED

        super().__init__(context)
        self.config = config or {}
        _SINGLE_REPLY_MODE_ENABLED = self.config.get("single_reply_mode", True)
        self._chains: dict[str, TaskChain] = {}
        self._chain_events: dict[str, Any] = {}
        self._session_system_prompts: dict[tuple[str, str], str] = {}
        self._session_prompt_seen_at: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()
        self._scheduler_task: asyncio.Task | None = None
        self._followup_tasks: set[asyncio.Task] = set()
        self._session_index: dict[str, list[str]] = {}
        self._data_file = os.path.join(_get_data_dir(), "task_chains.json")
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        self._load_chains()

    def _index_chain(self, chain: TaskChain) -> None:
        ids = self._session_index.setdefault(chain.session_id, [])
        if chain.id not in ids:
            ids.append(chain.id)

    def _deindex_chain(self, chain: TaskChain) -> None:
        ids = self._session_index.get(chain.session_id)
        if not ids:
            return
        self._session_index[chain.session_id] = [
            chain_id for chain_id in ids if chain_id != chain.id
        ]
        if not self._session_index[chain.session_id]:
            self._session_index.pop(chain.session_id, None)

    def _deactivate_chain(self, chain: TaskChain, *, now: float | None = None) -> None:
        chain.is_active = False
        chain.finished_at = now or time.time()
        chain.reset_completion_listener()
        chain.reset_callback_delivery()
        self._chain_events.pop(chain.id, None)
        self._deindex_chain(chain)

    def _active_chains_for_session(self, session_id: str) -> list[TaskChain]:
        """返回会话的活动任务，并修复可能残留的索引项。

        调用方必须持有 ``self._lock``。索引只保存活动任务；当旧版本数据或
        异常流程导致索引缺失时，才回退到全量扫描进行一次自愈。
        """
        indexed_ids = self._session_index.get(session_id, [])
        active = [
            chain
            for chain_id in dict.fromkeys(indexed_ids)
            if (chain := self._chains.get(chain_id))
            and chain.session_id == session_id
            and chain.is_active
            and not chain.is_completed
        ]
        valid_ids = [chain.id for chain in active]

        if indexed_ids:
            if valid_ids != indexed_ids:
                if valid_ids:
                    self._session_index[session_id] = valid_ids
                else:
                    self._session_index.pop(session_id, None)
            if active:
                return active

        active = [
            chain
            for chain in self._chains.values()
            if chain.session_id == session_id
            and chain.is_active
            and not chain.is_completed
        ]
        if active:
            self._session_index[session_id] = [chain.id for chain in active]
        return active

    @filter.custom_filter(_PrepareSingleReplyFilter, priority=-100000)
    async def _prepare_single_reply(self, _event: AstrMessageEvent) -> None:
        """该处理器不会执行；其过滤器负责在管道构建前准备事件。"""

    @on_decorating_result(priority=-100001)
    async def _discard_stale_callback_reply(self, event: AstrMessageEvent) -> None:
        """发送前再次校验回调令牌，避免超时重试产生重复消息。"""
        if event.get_extra("taskchain_callback", False) is not True:
            return
        token = str(event.get_extra(_CALLBACK_TOKEN_KEY, "") or "")
        if not token:
            return
        chain_id = str(event.get_extra("taskchain_chain_id", "") or "")
        async with self._lock:
            chain = self._chains.get(chain_id)
            is_current = bool(
                chain and chain.is_active and chain.pending_callback_token == token
            )
        if is_current:
            return

        result = event.get_result()
        if result:
            result.chain.clear()
        logger.info(
            f"[TaskChainTool] stale callback reply discarded: {chain_id}/{token}"
        )

    @on_decorating_result(priority=-100000)
    async def _consolidate_create_reply(self, event: AstrMessageEvent) -> None:
        """同一轮 create 只保留工具前或工具后的第一段可见回复。"""
        if not self.config.get("single_reply_mode", True):
            return
        if event.get_extra("taskchain_callback", False) is True:
            return
        result = event.get_result()
        if not result or not result.is_llm_result():
            return

        try:
            from astrbot.api.event import MessageChain

            text = MessageChain(chain=list(result.chain)).get_plain_text().strip()
        except Exception:
            text = ""

        if event.get_extra(_CREATE_SUCCEEDED_KEY, False) is True:
            if event.get_extra(_SUPPRESS_POST_REPLY_KEY, False) is True:
                result.chain.clear()
                logger.debug("[TaskChainTool] suppressed duplicate post-create reply")
            return

        if text:
            event.set_extra(_PRE_TOOL_REPLY_KEY, text)

    @on_llm_request()
    async def _inject_chain_state(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        if event.get_extra("taskchain_callback", False) is True:
            chain_id = str(event.get_extra("taskchain_chain_id", "") or "")
            token = str(event.get_extra(_CALLBACK_TOKEN_KEY, "") or "")
            async with self._lock:
                chain = self._chains.get(chain_id)
                is_current = bool(
                    chain and chain.is_active and chain.pending_callback_token == token
                )
            if token and not is_current:
                logger.info(
                    f"[TaskChainTool] stale callback discarded: {chain_id}/{token}"
                )
                event.stop_event()
            return

        session_id = event.unified_msg_origin
        if not self._scheduler_task or self._scheduler_task.done():
            try:
                if (
                    self._scheduler_task
                    and not self._scheduler_task.cancelled()
                    and (exc := self._scheduler_task.exception())
                ):
                    logger.error(
                        f"[TaskChainTool] scheduler loop died with exception: {exc}"
                    )
            except (asyncio.CancelledError, Exception):
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
                prompt_key = (session_id, conversation_id)
                self._session_system_prompts[prompt_key] = base_system_prompt
                self._session_prompt_seen_at[prompt_key] = time.time()
            for c in self._active_chains_for_session(session_id):
                if c.conversation_id != conversation_id:
                    continue
                # 新对话/reset时清理当前会话任务；不会影响同一用户的其他会话。
                if request.conversation and (
                    not request.conversation.history
                    or request.conversation.history == "[]"
                ):
                    self._deactivate_chain(c)
                    changed = True
                    continue
                is_last_task = c.current_index == len(c.tasks) - 1
                if is_last_task and (
                    c.completion_callback_at or time.time() >= c.current_task_wake_at
                ):
                    task_name = c.current_task.name if c.current_task else "当前任务"
                    request.system_prompt += (
                        f"\n[当前任务链：{task_name}已完成。"
                        "用户在完成监听期内回复了你，请直接自然地给出完成后的结果，"
                        "不要再等待后台提醒，不要提任务链、工具或时间。]"
                    )
                    self._deactivate_chain(c)
                    changed = True
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
                await self._save_chains_async()

    # ── 持久化 ──

    def _load_chains(self) -> None:
        try:
            with open(self._data_file, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"[TaskChainTool] load chains failed: {exc}")
            self._chains = {}
            self._session_index = {}
            return

        if not isinstance(raw, list):
            logger.error("[TaskChainTool] invalid persistence root: expected list")
            return

        now_t = time.time()
        allowed = set(TaskChain.__dataclass_fields__)
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                record = dict(item)
                tasks = [self._parse_task(t) for t in record.pop("tasks", [])]
                tasks = [task for task in tasks if task]
                if not tasks:
                    continue
                chain_data = {k: v for k, v in record.items() if k in allowed}
                if not chain_data.get("id"):
                    continue
                current_index = max(0, int(chain_data.get("current_index", 0) or 0))
                chain_data["current_index"] = min(current_index, len(tasks))
                loaded = TaskChain(**chain_data, tasks=tasks)

                # 活动任务即使已逾期也要恢复，由调度器补偿执行。
                expiry_base = loaded.finished_at or max(
                    loaded.current_task_wake_at,
                    loaded.created_at,
                )
                if not loaded.is_active and expiry_base and now_t - expiry_base > 600:
                    continue
                if loaded.is_active and loaded.pending_callback_kind:
                    loaded.callback_retry_at = min(
                        loaded.callback_retry_at or now_t,
                        now_t,
                    )

                self._chains[loaded.id] = loaded
                if loaded.is_active and not loaded.is_completed:
                    self._index_chain(loaded)
            except (TypeError, ValueError) as exc:
                logger.warning(f"[TaskChainTool] skipped invalid chain record: {exc}")

    def _serialize_chains(self) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
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
                    "callback_retry_at": c.callback_retry_at,
                    "pending_callback_kind": c.pending_callback_kind,
                    "pending_callback_token": c.pending_callback_token,
                    "followup_after_history_len": c.followup_after_history_len,
                    "finished_at": c.finished_at,
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
        return raw

    def _write_chains(self, raw: list[dict[str, Any]]) -> None:
        tmp_path = f"{self._data_file}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._data_file)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def _save_chains(self) -> None:
        """同步保存入口，主要用于初始化和测试。"""
        try:
            self._write_chains(self._serialize_chains())
        except (OSError, TypeError, ValueError) as exc:
            logger.error(f"[TaskChainTool] failed to save chains: {exc}")

    async def _save_chains_async(self) -> None:
        """锁内生成不可变快照，在线程中完成原子落盘。"""
        raw = self._serialize_chains()
        try:
            await asyncio.to_thread(self._write_chains, raw)
        except (OSError, TypeError, ValueError) as exc:
            logger.error(f"[TaskChainTool] failed to save chains: {exc}")

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
        if duration > 1440:
            duration = 1440
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

    def _parse_create_tasks(self, tasks_json: str) -> tuple[list[ChainTask], str]:
        if not isinstance(tasks_json, str):
            return [], "错误：tasks_json 不是合法 JSON。"
        if len(tasks_json) > 65536:
            return [], "错误：tasks_json 过大（>64KB）。"
        try:
            tasks_data = json.loads(tasks_json)
        except json.JSONDecodeError:
            return [], "错误：tasks_json 不是合法 JSON。"
        if not isinstance(tasks_data, list):
            return [], "错误：tasks_json 必须是 JSON 数组。"
        if len(tasks_data) > 50:
            return [], "错误：任务列表过长，最多 50 个。"

        tasks = [task for raw in tasks_data if (task := self._parse_task(raw))]
        if not tasks:
            return [], "错误：任务列表不能为空。"
        for task in tasks:
            task.duration_minutes = (
                max(task.duration_minutes * 60, MIN_TASK_SECONDS) / 60
            )
        return tasks, ""

    def _build_scheduled_tasks(
        self, source_tasks: list[ChainTask]
    ) -> tuple[list[ChainTask], float]:
        total_secs = sum(task.duration_minutes * 60 for task in source_tasks)
        merged = ChainTask(
            name=source_tasks[0].name,
            description=source_tasks[0].description,
            duration_minutes=total_secs / 60,
            prompt=next(
                (task.prompt for task in reversed(source_tasks) if task.prompt),
                source_tasks[0].prompt,
            ),
        )
        checkin = self._make_checkin(merged)
        main_secs = total_secs
        if checkin:
            main_secs = max(
                MIN_TASK_SECONDS,
                total_secs - checkin.duration_minutes * 60,
            )
        main = ChainTask(
            name=merged.name,
            description=merged.description,
            duration_minutes=main_secs / 60,
            prompt=merged.prompt,
        )
        return ([checkin, main] if checkin else [main]), total_secs

    @staticmethod
    def _event_attr(event: AstrMessageEvent, attr_path: str, default: Any) -> Any:
        current: Any = event
        for part in attr_path.split("."):
            if current is None or type(current).__module__ == "unittest.mock":
                return default
            value = getattr(current, part, None)
            if callable(value) and type(value).__module__ != "unittest.mock":
                try:
                    current = value()
                except Exception:
                    return default
            else:
                current = value
        if current is None or type(current).__module__ == "unittest.mock":
            return default
        return current

    def _event_metadata(self, event: AstrMessageEvent) -> dict[str, Any]:
        return {
            "platform_id": self._event_attr(event, "platform_meta.id", "mock"),
            "platform_name": self._event_attr(event, "platform_meta.name", "mock"),
            "session_id": self._event_attr(event, "session_id", ""),
            "message_type": self._event_attr(event, "get_message_type.value", 1),
            "sender_id": self._event_attr(event, "get_sender_id", ""),
            "sender_name": self._event_attr(event, "get_sender_name", ""),
            "group_id": self._event_attr(event, "get_group_id", ""),
            "self_id": self._event_attr(event, "get_self_id", ""),
        }

    async def _create_chain_locked(
        self,
        event: AstrMessageEvent,
        session_id: str,
        conversation_id: str,
        tasks_json: str,
    ) -> str:
        if not conversation_id:
            return "错误：当前会话尚未初始化，无法创建后台任务。请先按当前人设自然回复，不要创建任务。"

        source_tasks, error = self._parse_create_tasks(tasks_json)
        if error:
            return error
        for chain in self._active_chains_for_session(session_id):
            if chain.conversation_id == conversation_id:
                self._deactivate_chain(chain)

        tasks, total_secs = self._build_scheduled_tasks(source_tasks)
        chain_id = uuid.uuid4().hex[:12]
        now_t = time.time()
        chain = TaskChain(
            id=chain_id,
            session_id=session_id,
            conversation_id=conversation_id,
            system_prompt=self._chain_system_prompt(session_id, conversation_id),
            tasks=tasks,
            created_at=now_t,
            current_task_started_at=now_t,
            current_task_wake_at=now_t + tasks[0].duration_minutes * 60,
            event_metadata=self._event_metadata(event),
        )
        self._chains[chain_id] = chain
        self._index_chain(chain)
        self._chain_events[chain_id] = event
        await self._save_chains_async()

        pre_tool_reply = event.get_extra(_PRE_TOOL_REPLY_KEY, "")
        if not isinstance(pre_tool_reply, str):
            pre_tool_reply = ""
        suppress_post_reply = bool(
            self.config.get("single_reply_mode", True) and pre_tool_reply.strip()
        )
        event.set_extra(_CREATE_SUCCEEDED_KEY, True)
        event.set_extra(_SUPPRESS_POST_REPLY_KEY, suppress_post_reply)

        done_time = datetime.fromtimestamp(now_t + total_secs).strftime("%H:%M:%S")
        reply_instruction = (
            "你在工具调用前的自然回复已经展示给用户。"
            "本轮最终输出必须为空，不要再补充确认、动作描写或同义句。"
            if suppress_post_reply
            else (
                "你在工具调用前没有向用户展示自然回复。"
                "本轮最终只输出一段简短、完整的人设回复，表达已经开始行动或记下约定。"
            )
        )
        return (
            f"[任务已安排(id={chain_id})] "
            f"预计{done_time}完成。当前状态：任务刚开始，仍在进行中。"
            + reply_instruction
            + "你已经创建任务，表示角色已经愿意执行；本轮最终回复不能再拒绝、否认行动、说没空/不去，"
            "也不要让用户自己去做。"
            "不要在本轮追问会改变任务定义的细节；需要偏好、路线、材料等互动时，交给后续中途互动。"
            "严禁说已经做好、已经回来、已经到达、已经完成、已经把结果交给用户，或让用户提前体验结果。"
        )

    def _list_chains_locked(self, session_id: str, conversation_id: str) -> str:
        active = [
            chain
            for chain in self._active_chains_for_session(session_id)
            if chain.conversation_id == conversation_id
        ]
        lines = []
        for chain in active:
            task = chain.current_task
            if not task:
                continue
            remain = max(0, chain.current_task_wake_at - time.time())
            lines.append(
                f"  [{chain.id}] 「{task.name}」{task.description} "
                f"(剩余 {int(remain // 60)} 分 {int(remain % 60)} 秒) "
                f"[{chain.current_index}/{len(chain.tasks)}]"
            )
        return (
            "活跃任务链：\n" + "\n".join(lines) if lines else "当前没有活跃的任务链。"
        )

    async def _cancel_chain_locked(
        self, session_id: str, conversation_id: str, chain_id: str
    ) -> str:
        chain = self._chains.get(chain_id)
        if (
            not chain
            or chain.session_id != session_id
            or chain.conversation_id != conversation_id
        ):
            return f"未找到任务链 {chain_id}。"
        self._deactivate_chain(chain)
        await self._save_chains_async()
        return f"任务链 {chain_id} 已取消。"

    async def _advance_chain_locked(
        self, session_id: str, conversation_id: str, chain_id: str
    ) -> str:
        chain = self._chains.get(chain_id)
        if (
            not chain
            or chain.session_id != session_id
            or chain.conversation_id != conversation_id
        ):
            return f"未找到任务链 {chain_id}。"
        if chain.is_completed:
            return f"任务链 {chain_id} 已经完成。"

        current = chain.current_task
        next_task = chain.advance()
        if next_task:
            now_t = time.time()
            chain.current_task_started_at = now_t
            chain.current_task_wake_at = now_t + next_task.duration_minutes * 60
            chain.reset_completion_listener()
            chain.reset_callback_delivery()
            wake = datetime.fromtimestamp(chain.current_task_wake_at).strftime(
                "%H:%M:%S"
            )
            await self._save_chains_async()
            return (
                f"阶段「{current.name}」已完成，进入「{next_task.name}」。\n"
                f"{next_task.description}\n预计 {next_task.duration_minutes} 分钟（{wake}）后唤醒。"
            )
        self._deactivate_chain(chain)
        await self._save_chains_async()
        return f"阶段「{current.name}」已完成，任务链全部完成！"

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
                return await self._create_chain_locked(
                    event,
                    session_id,
                    conversation_id,
                    tasks_json,
                )
            if action == "list":
                return self._list_chains_locked(session_id, conversation_id)
            if action == "cancel":
                return await self._cancel_chain_locked(
                    session_id,
                    conversation_id,
                    chain_id,
                )
            if action == "advance":
                return await self._advance_chain_locked(
                    session_id,
                    conversation_id,
                    chain_id,
                )
            return f"未知操作：{action}"

    # ── 后台自动推进并发送自然提醒 ──

    async def initialize(self) -> None:
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"[TaskChainTool] scheduler: {e}", exc_info=True)
            await asyncio.sleep(10)

    def _cleanup_expired_chains(self) -> bool:
        """Remove completed/inactive chains older than 10 minutes."""
        now_t = time.time()
        expired = [
            cid
            for cid, c in self._chains.items()
            if c.is_completed
            and now_t - (c.finished_at or max(c.current_task_wake_at, c.created_at))
            > 600
        ]
        for cid in expired:
            chain = self._chains.pop(cid, None)
            if chain:
                self._chain_events.pop(cid, None)
                self._deindex_chain(chain)
        if expired:
            logger.debug(f"[TaskChainTool] Cleaned up {len(expired)} expired chains")

        active_sessions = {
            (c.session_id, c.conversation_id)
            for c in self._chains.values()
            if c.is_active and c.conversation_id
        }
        stale_keys = [
            k
            for k in self._session_system_prompts
            if k not in active_sessions
            and now_t - self._session_prompt_seen_at.get(k, 0)
            > PROMPT_CACHE_TTL_SECONDS
        ]
        for k in stale_keys:
            del self._session_system_prompts[k]
            self._session_prompt_seen_at.pop(k, None)
        return bool(expired)

    async def _tick(self) -> None:
        now_t = time.time()
        to_advance: list[TaskChain] = []
        async with self._lock:
            changed = self._cleanup_expired_chains()
            for c in list(self._chains.values()):
                if not c.is_active or c.is_completed:
                    continue
                if c.pending_callback_kind:
                    if now_t >= c.callback_retry_at:
                        to_advance.append(c)
                    continue
                if c.completion_callback_at:
                    if now_t >= c.completion_callback_at:
                        to_advance.append(c)
                    continue
                is_last_task = c.current_index == len(c.tasks) - 1
                remain = c.current_task_wake_at - now_t
                if is_last_task and 0 < remain <= COMPLETION_LISTEN_SECONDS:
                    c.completion_listen_started_at = now_t
                    c.completion_callback_at = (
                        c.current_task_wake_at + COMPLETION_LISTEN_SECONDS
                    )
                    changed = True
                    continue
                if now_t >= c.current_task_wake_at:
                    to_advance.append(c)
            if changed:
                await self._save_chains_async()

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
        self,
        chain: TaskChain,
        visible_text: str,
        kind: str,
        token: str = "",
    ) -> AstrMessageEvent | None:
        import copy

        try:
            from astrbot.core.utils.trace import TraceSpan

            try:
                session = MessageSession.from_str(chain.session_id)
                new_event = _ReliableCronMessageEvent(
                    context=self.context,
                    session=session,
                    message=visible_text,
                    sender_name="TaskChain",
                    message_type=session.message_type,
                )
            except (TypeError, ValueError):
                # 兼容旧测试数据和不规范的历史 UMO；正常生产路径使用 Cron 事件。
                source_event = self._chain_events.get(chain.id)
                if not source_event:
                    logger.error(
                        f"[TaskChainTool] invalid callback session: {chain.session_id}"
                    )
                    return None
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
            if token:
                new_event.set_extra(_CALLBACK_TOKEN_KEY, token)
            return new_event
        except Exception as e:
            logger.error(f"[TaskChainTool] build pipeline callback failed: {e}")
            return None

    async def _queue_pipeline_callback(
        self,
        chain: TaskChain,
        text: str,
        kind: str,
        token: str = "",
    ) -> bool:
        event = self._build_callback_event(
            chain,
            "[TaskChain callback]",
            kind,
            token,
        )
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
            history_len = self._conversation_history_len(conv)
            async with self._lock:
                current = self._chains.get(chain.id)
                if current and current.is_active:
                    current.followup_after_history_len = history_len
                    await self._save_chains_async()
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

    def _finalize_callback_locked(self, chain: TaskChain, kind: str) -> bool:
        """确认回调送达后推进状态；返回是否需要安排中途跟进。"""
        current = chain.current_task
        if not current:
            self._deactivate_chain(chain)
            return False

        if kind == "interact":
            scheduled_start = chain.current_task_wake_at
            nxt = chain.advance()
            if not nxt:
                self._deactivate_chain(chain)
                return False
            chain.current_task_started_at = scheduled_start
            chain.current_task_wake_at = scheduled_start + nxt.duration_minutes * 60
            chain.reset_completion_listener()
            chain.reset_callback_delivery()
            return True

        chain.advance()
        self._deactivate_chain(chain)
        return False

    def _schedule_followup(self, chain: TaskChain, interact_text: str) -> None:
        task = asyncio.create_task(
            self._followup_check(
                chain,
                chain.session_id,
                interact_text=interact_text,
            )
        )
        self._followup_tasks.add(task)
        task.add_done_callback(self._followup_tasks.discard)

    @after_message_sent(priority=-100000)
    async def _ack_callback_delivery(self, event: AstrMessageEvent) -> None:
        token = str(event.get_extra(_CALLBACK_TOKEN_KEY, "") or "")
        if not token:
            return
        if event.get_extra("taskchain_delivery_failed", False) is True:
            return
        if not getattr(event, "_has_send_oper", False):
            return

        chain_id = str(event.get_extra("taskchain_chain_id", "") or "")
        should_followup = False
        interact_text = ""
        chain: TaskChain | None = None
        async with self._lock:
            chain = self._chains.get(chain_id)
            if (
                not chain
                or not chain.is_active
                or chain.pending_callback_token != token
            ):
                return
            kind = chain.pending_callback_kind
            if kind == "interact" and chain.current_task:
                current = chain.current_task
                next_index = chain.current_index + 1
                if next_index < len(chain.tasks):
                    interact_text = self._interact_callback_text(
                        current,
                        chain.tasks[next_index],
                    )
            should_followup = self._finalize_callback_locked(chain, kind)
            await self._save_chains_async()

        if should_followup and chain:
            self._schedule_followup(chain, interact_text)

    async def _deliver_completion_fallback(
        self,
        chain: TaskChain,
        token: str,
    ) -> None:
        task = chain.current_task
        if not task:
            return
        try:
            from astrbot.api.event import MessageChain

            sent = await self.context.send_message(
                chain.session_id,
                MessageChain().message(f"「{task.name}」已经完成了。"),
            )
        except Exception as exc:
            logger.error(
                f"[TaskChainTool] completion fallback failed: {exc}",
                exc_info=True,
            )
            return
        if not sent:
            logger.error(
                f"[TaskChainTool] completion fallback not delivered: {chain.id}"
            )
            return

        async with self._lock:
            current = self._chains.get(chain.id)
            if (
                current
                and current.is_active
                and current.pending_callback_token == token
            ):
                self._finalize_callback_locked(current, "completion")
                await self._save_chains_async()

    async def _wake_and_advance(self, chain: TaskChain) -> None:
        callback_text = ""
        callback_kind = ""
        callback_token = ""
        use_fallback = False

        async with self._lock:
            chain = self._chains.get(chain.id, chain)
            if not chain.is_active or chain.is_completed:
                return
            now_t = time.time()
            ct = chain.current_task
            if not ct:
                self._deactivate_chain(chain)
                await self._save_chains_async()
                return

            if chain.pending_callback_kind:
                if now_t < chain.callback_retry_at:
                    return
                callback_kind = chain.pending_callback_kind
            else:
                due_at = chain.completion_callback_at or chain.current_task_wake_at
                if now_t < due_at:
                    return
                callback_kind = (
                    "interact"
                    if chain.current_index < len(chain.tasks) - 1
                    else "completion"
                )
                chain.pending_callback_kind = callback_kind
                chain.callback_retry_count = 0

            if callback_kind == "interact":
                next_index = chain.current_index + 1
                if next_index >= len(chain.tasks):
                    self._deactivate_chain(chain)
                    await self._save_chains_async()
                    return
                callback_text = self._interact_callback_text(
                    ct,
                    chain.tasks[next_index],
                )
            else:
                callback_text = self._completion_callback_text(ct)

            callback_token = uuid.uuid4().hex
            chain.pending_callback_token = callback_token
            chain.callback_retry_at = now_t + CALLBACK_ACK_TIMEOUT_SECONDS
            use_fallback = chain.callback_retry_count >= MAX_CALLBACK_PIPELINE_ATTEMPTS
            if not use_fallback:
                chain.callback_retry_count += 1
            await self._save_chains_async()

        if use_fallback:
            if callback_kind == "interact":
                async with self._lock:
                    current = self._chains.get(chain.id)
                    if current and current.pending_callback_token == callback_token:
                        self._finalize_callback_locked(current, "interact")
                        await self._save_chains_async()
                return
            await self._deliver_completion_fallback(chain, callback_token)
            return

        queued = await self._queue_pipeline_callback(
            chain,
            callback_text,
            callback_kind,
            callback_token,
        )
        if not queued:
            logger.warning(
                f"[TaskChainTool] pipeline callback pending retry: "
                f"{chain.id}/{callback_kind}"
            )
            async with self._lock:
                current = self._chains.get(chain.id)
                if current and current.pending_callback_token == callback_token:
                    current.callback_retry_at = time.time() + 10
                    await self._save_chains_async()

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
            # Find the first assistant message after the interact point
            interact_idx = None
            for offset, item in enumerate(search_hist):
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
        for task in list(self._followup_tasks):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
