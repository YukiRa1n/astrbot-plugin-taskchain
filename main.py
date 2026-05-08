from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import llm_tool


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
    tasks: list[ChainTask] = field(default_factory=list)
    current_index: int = 0
    is_active: bool = True
    created_at: float = 0.0
    current_task_started_at: float = 0.0
    current_task_wake_at: float = 0.0

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

    # ── 持久化 ──

    def _load_chains(self) -> None:
        try:
            with open(self._data_file, encoding="utf-8") as f:
                raw = json.load(f)
            for c in raw:
                tasks = [ChainTask(**t) for t in c.pop("tasks", [])]
                self._chains[c["id"]] = TaskChain(**c, tasks=tasks)
        except (FileNotFoundError, json.JSONDecodeError):
            self._chains = {}

    def _save_chains(self) -> None:
        raw = []
        for c in self._chains.values():
            raw.append({
                "id": c.id, "session_id": c.session_id,
                "current_index": c.current_index, "is_active": c.is_active,
                "created_at": c.created_at,
                "current_task_started_at": c.current_task_started_at,
                "current_task_wake_at": c.current_task_wake_at,
                "tasks": [
                    {"name": t.name, "description": t.description,
                     "duration_minutes": t.duration_minutes, "prompt": t.prompt}
                    for t in c.tasks
                ],
            })
        with open(self._data_file, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    # ── LLM 工具 ──

    def _make_checkin(self, main: ChainTask) -> ChainTask | None:
        """如果主任务较长，生成一个前置互动任务"""
        if main.duration_minutes < 2:
            return None
        checkin_prompt = (
            f"自然地询问一下用户关于「{main.name}」的细节或偏好，"
            f"让对话更生动。不要提任务或时间。"
        )
        return ChainTask(
            name=main.name,
            description=main.description,
            duration_minutes=0.5,
            prompt=checkin_prompt,
        )

    @llm_tool(name="chain_task")
    async def chain_task(
        self,
        event: AstrMessageEvent,
        action: str,
        tasks_json: str = "[]",
        chain_id: str = "",
    ) -> str:
        """角色扮演用时间推进器。用户说一会去做什么时，估算时长设任务，到点自动推进。调用即进入状态，不告知用户不提工具不提时间。较长的任务会自动插一个互动环节（如泡咖啡时问想喝什么口味）。与用户对话时建议用 list 查看当前任务状态来自然衔接。

        Args:
            action(string): 操作类型 create(创建) list(查看) cancel(取消) advance(手动推进)。
            tasks_json(string): action=create 时必填，JSON数组，放1个主任务：[{"name":"干什么","description":"描述","duration_minutes":1,"prompt":"到点后的状态提示"}]。duration_minutes支持小数(0.17=10秒)。
            chain_id(string): action=cancel/advance 时必填。
        """
        session_id = event.unified_msg_origin
        _MIN_SECONDS = 10

        async with self._lock:
            if action == "create":
                tasks_data = json.loads(tasks_json)
                tasks_raw = [ChainTask(**t) for t in tasks_data]
                if not tasks_raw:
                    return "错误：任务列表不能为空。"
                for t in tasks_raw:
                    secs = max(t.duration_minutes * 60, _MIN_SECONDS)
                    t.duration_minutes = secs / 60

                total_secs = sum(t.duration_minutes * 60 for t in tasks_raw)
                prompts = [t.prompt for t in tasks_raw if t.prompt]
                merged = ChainTask(
                    name=tasks_raw[0].name,
                    description=tasks_raw[0].description,
                    duration_minutes=total_secs / 60,
                    prompt=prompts[-1] if prompts else tasks_raw[0].prompt,
                )

                # 长任务前插一个互动任务
                checkin = self._make_checkin(merged)
                main = ChainTask(
                    name=merged.name,
                    description=merged.description,
                    duration_minutes=total_secs / 60,
                    prompt=merged.prompt,
                )
                tasks = [checkin, main] if checkin else [main]
                cid = uuid.uuid4().hex[:12]
                now_t = time.time()
                chain = TaskChain(
                    id=cid, session_id=session_id, tasks=tasks,
                    created_at=now_t, current_task_started_at=now_t,
                    current_task_wake_at=now_t + tasks[0].duration_minutes * 60,
                )
                self._chains[cid] = chain
                self._save_chains()
                first = tasks[0]
                wake = datetime.fromtimestamp(chain.current_task_wake_at).strftime("%H:%M:%S")
                return (
                    f"任务链已创建 (id={cid})。\n"
                    f"「{first.name}」{first.description}\n"
                    f"预计 {first.duration_minutes:.1f} 分钟（{wake}）后唤醒。"
                )

            elif action == "list":
                active = [c for c in self._chains.values()
                          if c.session_id == session_id and not c.is_completed]
                if not active:
                    return "当前没有活跃的任务链。"
                lines = []
                for c in active:
                    ct = c.current_task
                    remain = max(0, c.current_task_wake_at - time.time())
                    lines.append(
                        f"  [{c.id}] 「{ct.name}」{ct.description} "
                        f"(剩余 {int(remain//60)} 分 {int(remain%60)} 秒) "
                        f"[{c.current_index}/{len(c.tasks)}]"
                    )
                return "活跃任务链：\n" + "\n".join(lines)

            elif action == "cancel":
                if chain_id in self._chains:
                    self._chains[chain_id].is_active = False
                    self._save_chains()
                    return f"任务链 {chain_id} 已取消。"
                return f"未找到任务链 {chain_id}。"

            elif action == "advance":
                if chain_id not in self._chains:
                    return f"未找到任务链 {chain_id}。"
                chain = self._chains[chain_id]
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
                self._save_chains()
                return f"阶段「{ct.name}」已完成，任务链全部完成！"

            return f"未知操作：{action}"

    # ── 后台自动推进（仅推进链，不发送消息）──

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
            for c in list(self._chains.values()):
                if not c.is_active or c.is_completed:
                    continue
                if now_t >= c.current_task_wake_at:
                    to_advance.append(c)

        for chain in to_advance:
            await self._wake_and_advance(chain)

    async def _wake_and_advance(self, chain: TaskChain) -> None:
        nxt = chain.advance()
        async with self._lock:
            if nxt:
                chain.current_task_started_at = time.time()
                chain.current_task_wake_at = time.time() + nxt.duration_minutes * 60
            else:
                chain.is_active = False
            self._save_chains()

    async def terminate(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
