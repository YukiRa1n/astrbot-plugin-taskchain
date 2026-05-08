"""TaskChain 插件测试 - 单元测试 + Mock 测试。

测试覆盖：
1. ChainTask / TaskChain 数据结构和状态机
2. chain_task LLM 工具的 create/list/cancel/advance 操作
3. 调度器 _tick 逻辑
4. 持久化加载/保存
5. 端到端场景模拟（Mock LLM）
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 将项目根目录添加到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("ASTRBOT_TEST_MODE", "true")

# ── 导入被测模块 ──
from astrbot.builtin_stars.taskchain.main import (
    ChainTask,
    TaskChain,
    TaskChainPlugin,
)


# ============================================================
# 1. 数据结构单元测试
# ============================================================


class TestChainTask:
    """测试 ChainTask 数据类。"""

    def test_default_values(self):
        task = ChainTask()
        assert task.name == ""
        assert task.description == ""
        assert task.duration_minutes == 10
        assert task.prompt == ""

    def test_custom_values(self):
        task = ChainTask(
            name="go_out",
            description="出门散步",
            duration_minutes=15,
            prompt="你到了外面，空气很好",
        )
        assert task.name == "go_out"
        assert task.description == "出门散步"
        assert task.duration_minutes == 15
        assert task.prompt == "你到了外面，空气很好"

    def test_to_dict_roundtrip(self):
        task = ChainTask(name="eat", description="吃饭", duration_minutes=30, prompt="吃完了")
        d = {
            "name": task.name,
            "description": task.description,
            "duration_minutes": task.duration_minutes,
            "prompt": task.prompt,
        }
        restored = ChainTask(**d)
        assert restored.name == task.name
        assert restored.description == task.description
        assert restored.duration_minutes == task.duration_minutes
        assert restored.prompt == task.prompt


class TestTaskChain:
    """测试 TaskChain 状态机。"""

    def _make_chain(self, tasks=None):
        if tasks is None:
            tasks = [
                ChainTask(name="step1", description="第一阶段", duration_minutes=10, prompt="完成1"),
                ChainTask(name="step2", description="第二阶段", duration_minutes=20, prompt="完成2"),
                ChainTask(name="step3", description="第三阶段", duration_minutes=5, prompt="完成3"),
            ]
        now_t = time.time()
        return TaskChain(
            id="test-chain-001",
            session_id="test_session",
            tasks=tasks,
            created_at=now_t,
            current_task_started_at=now_t,
            current_task_wake_at=now_t + tasks[0].duration_minutes * 60,
        )

    def test_initial_state(self):
        chain = self._make_chain()
        assert chain.current_index == 0
        assert chain.is_active is True
        assert chain.is_completed is False
        assert chain.current_task.name == "step1"

    def test_current_task_none_when_done(self):
        chain = self._make_chain()
        chain.current_index = 3  # past end
        assert chain.current_task is None
        assert chain.is_completed is True

    def test_advance(self):
        chain = self._make_chain()
        # step1 -> step2
        nxt = chain.advance()
        assert nxt is not None
        assert nxt.name == "step2"
        assert chain.current_index == 1

        # step2 -> step3
        nxt = chain.advance()
        assert nxt is not None
        assert nxt.name == "step3"
        assert chain.current_index == 2

        # step3 -> done
        nxt = chain.advance()
        assert nxt is None
        assert chain.is_completed is True

    def test_advance_past_end(self):
        chain = self._make_chain()
        chain.current_index = 2  # last task
        nxt = chain.advance()
        assert nxt is None
        assert chain.is_completed is True

    def test_deactivate(self):
        chain = self._make_chain()
        chain.is_active = False
        assert chain.is_completed is True

    def test_single_task_chain(self):
        tasks = [ChainTask(name="only", description="唯一任务", duration_minutes=5, prompt="done")]
        chain = self._make_chain(tasks)
        assert chain.current_task.name == "only"
        nxt = chain.advance()
        assert nxt is None
        assert chain.is_completed is True

    def test_wake_time_calculation(self):
        tasks = [
            ChainTask(name="t1", description="", duration_minutes=10, prompt=""),
            ChainTask(name="t2", description="", duration_minutes=30, prompt=""),
        ]
        now_t = 1000.0
        chain = TaskChain(
            id="x", session_id="s", tasks=tasks,
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t + 10 * 60,
        )
        assert chain.current_task_wake_at == 1600.0
        # After advance
        nxt = chain.advance()
        assert nxt.name == "t2"
        chain.current_task_started_at = now_t + 10 * 60
        chain.current_task_wake_at = chain.current_task_started_at + 30 * 60
        assert chain.current_task_wake_at == now_t + 10 * 60 + 30 * 60


# ============================================================
# 2. chain_task LLM 工具 Mock 测试
# ============================================================


class TestChainTaskTool:
    """测试 chain_task 工具的 create/list/cancel/advance 操作。"""

    @pytest.fixture
    def mock_event(self):
        event = MagicMock()
        event.unified_msg_origin = "test_umo_123"
        return event

    @pytest.fixture
    def plugin(self, tmp_path):
        data_dir = tmp_path / "taskchain_data"
        data_dir.mkdir()
        data_file = data_dir / "task_chains.json"

        ctx = MagicMock()
        plugin = TaskChainPlugin(context=ctx, config={})
        plugin._data_file = str(data_file)
        plugin._chains = {}
        plugin._stop = False
        return plugin

    @pytest.mark.asyncio
    async def test_create_chain(self, plugin, mock_event):
        tasks_json = json.dumps([
            {"name": "go_out", "description": "出门", "duration_minutes": 10, "prompt": "到了外面"},
            {"name": "eat", "description": "吃饭", "duration_minutes": 30, "prompt": "吃完了"},
        ])
        result = await plugin.chain_task(mock_event, action="create", tasks_json=tasks_json)
        assert "任务链已创建" in result
        assert "id=" in result
        assert "go_out" in result
        assert len(plugin._chains) == 1

    @pytest.mark.asyncio
    async def test_create_chain_empty_tasks(self, plugin, mock_event):
        result = await plugin.chain_task(mock_event, action="create", tasks_json="[]")
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_create_chain_invalid_json(self, plugin, mock_event):
        with pytest.raises(json.JSONDecodeError):
            await plugin.chain_task(mock_event, action="create", tasks_json="not json")

    @pytest.mark.asyncio
    async def test_list_empty(self, plugin, mock_event):
        result = await plugin.chain_task(mock_event, action="list")
        assert "没有活跃的任务链" in result

    @pytest.mark.asyncio
    async def test_list_after_create(self, plugin, mock_event):
        tasks_json = json.dumps([
            {"name": "t1", "description": "任务1", "duration_minutes": 10, "prompt": ""},
        ])
        await plugin.chain_task(mock_event, action="create", tasks_json=tasks_json)
        result = await plugin.chain_task(mock_event, action="list")
        assert "活跃任务链" in result
        assert "t1" in result

    @pytest.mark.asyncio
    async def test_cancel(self, plugin, mock_event):
        tasks_json = json.dumps([
            {"name": "t1", "description": "任务1", "duration_minutes": 10, "prompt": ""},
        ])
        create_result = await plugin.chain_task(mock_event, action="create", tasks_json=tasks_json)
        chain_id = create_result.split("id=")[1].split(")")[0]

        result = await plugin.chain_task(mock_event, action="cancel", chain_id=chain_id)
        assert "已取消" in result
        assert plugin._chains[chain_id].is_active is False

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, plugin, mock_event):
        result = await plugin.chain_task(mock_event, action="cancel", chain_id="nonexistent")
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_advance(self, plugin, mock_event):
        # 直接创建2个未合并的任务到 _chains，测试 advance 功能
        tasks_raw = [
            ChainTask(name="step1", description="第一阶段", duration_minutes=10, prompt="完成1"),
            ChainTask(name="step2", description="第二阶段", duration_minutes=20, prompt="完成2"),
        ]
        chain = TaskChain(
            id="advance-test", session_id="test",
            tasks=tasks_raw, created_at=0,
            current_task_started_at=0,
            current_task_wake_at=600,
        )
        plugin._chains["advance-test"] = chain

        result = await plugin.chain_task(mock_event, action="advance", chain_id="advance-test")
        assert "step1" in result
        assert "已完成" in result
        assert "step2" in result
        assert "进入" in result

    @pytest.mark.asyncio
    async def test_advance_to_completion(self, plugin, mock_event):
        tasks_json = json.dumps([
            {"name": "only", "description": "唯一任务", "duration_minutes": 5, "prompt": "done"},
        ])
        create_result = await plugin.chain_task(mock_event, action="create", tasks_json=tasks_json)
        chain_id = create_result.split("id=")[1].split(")")[0]

        result = await plugin.chain_task(mock_event, action="advance", chain_id=chain_id)
        assert "全部完成" in result
        assert plugin._chains[chain_id].is_active is False

    @pytest.mark.asyncio
    async def test_advance_already_completed(self, plugin, mock_event):
        tasks_json = json.dumps([
            {"name": "only", "description": "唯一任务", "duration_minutes": 5, "prompt": "done"},
        ])
        create_result = await plugin.chain_task(mock_event, action="create", tasks_json=tasks_json)
        chain_id = create_result.split("id=")[1].split(")")[0]

        # First advance completes it
        await plugin.chain_task(mock_event, action="advance", chain_id=chain_id)
        # Second advance should say already completed
        result = await plugin.chain_task(mock_event, action="advance", chain_id=chain_id)
        assert "已经完成" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, plugin, mock_event):
        result = await plugin.chain_task(mock_event, action="foobar")
        assert "未知操作" in result

    @pytest.mark.asyncio
    async def test_session_isolation(self, plugin, mock_event):
        # Create chain for session A
        tasks_json = json.dumps([
            {"name": "t1", "description": "任务1", "duration_minutes": 10, "prompt": ""},
        ])
        await plugin.chain_task(mock_event, action="create", tasks_json=tasks_json)

        # Different session should see no chains
        event2 = MagicMock()
        event2.unified_msg_origin = "different_session"
        result = await plugin.chain_task(event2, action="list")
        assert "没有活跃的任务链" in result


# ============================================================
# 3. 调度器 _tick 逻辑 Mock 测试
# ============================================================


class TestSchedulerTick:
    """测试调度器的 _tick 逻辑。"""

    @pytest.fixture
    def plugin(self, tmp_path):
        data_dir = tmp_path / "taskchain_data"
        data_dir.mkdir()
        data_file = data_dir / "task_chains.json"

        ctx = MagicMock()
        plugin = TaskChainPlugin(context=ctx, config={})
        plugin._data_file = str(data_file)
        plugin._chains = {}
        plugin._stop = False
        plugin._proactive_reply = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_tick_no_chains(self, plugin):
        await plugin._tick()
        plugin._proactive_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_not_due_yet(self, plugin):
        now_t = time.time()
        chain = TaskChain(
            id="c1", session_id="s1",
            tasks=[ChainTask(name="t1", description="", duration_minutes=60, prompt="")],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t + 3600,  # 1 hour from now
        )
        plugin._chains["c1"] = chain
        await plugin._tick()
        plugin._proactive_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_due_triggers_wake(self, plugin):
        now_t = time.time()
        chain = TaskChain(
            id="c1", session_id="s1",
            tasks=[
                ChainTask(name="t1", description="任务1", duration_minutes=10, prompt="提示1"),
                ChainTask(name="t2", description="任务2", duration_minutes=20, prompt="提示2"),
            ],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t - 1,  # already due
        )
        plugin._chains["c1"] = chain
        await plugin._tick()

        # Should have triggered proactive reply
        plugin._proactive_reply.assert_called_once()
        call_args = plugin._proactive_reply.call_args
        assert call_args[0][0] == "s1"  # session_id

        # Chain should have advanced to next task
        assert chain.current_index == 1
        assert chain.current_task.name == "t2"
        assert chain.is_active is True

    @pytest.mark.asyncio
    async def test_tick_due_last_task_completes_chain(self, plugin):
        now_t = time.time()
        chain = TaskChain(
            id="c1", session_id="s1",
            tasks=[ChainTask(name="last", description="最后任务", duration_minutes=5, prompt="done")],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t - 1,
        )
        plugin._chains["c1"] = chain
        await plugin._tick()

        plugin._proactive_reply.assert_called_once()
        assert chain.is_active is False
        assert chain.is_completed is True

    @pytest.mark.asyncio
    async def test_tick_skips_inactive(self, plugin):
        now_t = time.time()
        chain = TaskChain(
            id="c1", session_id="s1",
            tasks=[ChainTask(name="t1", description="", duration_minutes=10, prompt="")],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t - 1,
            is_active=False,
        )
        plugin._chains["c1"] = chain
        await plugin._tick()
        plugin._proactive_reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_multiple_due(self, plugin):
        now_t = time.time()
        c1 = TaskChain(
            id="c1", session_id="s1",
            tasks=[ChainTask(name="t1", description="", duration_minutes=10, prompt="")],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t - 1,
        )
        c2 = TaskChain(
            id="c2", session_id="s2",
            tasks=[ChainTask(name="t2", description="", duration_minutes=10, prompt="")],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t - 1,
        )
        plugin._chains["c1"] = c1
        plugin._chains["c2"] = c2
        await plugin._tick()
        assert plugin._proactive_reply.call_count == 2


# ============================================================
# 4. 持久化测试
# ============================================================


class TestPersistence:
    """测试任务链的加载/保存。"""

    @pytest.fixture
    def plugin(self, tmp_path):
        data_dir = tmp_path / "taskchain_data"
        data_dir.mkdir()
        data_file = data_dir / "task_chains.json"

        ctx = MagicMock()
        plugin = TaskChainPlugin(context=ctx, config={})
        plugin._data_file = str(data_file)
        plugin._chains = {}
        plugin._stop = False
        return plugin

    def test_save_and_load(self, plugin):
        now_t = time.time()
        chain = TaskChain(
            id="persist-test", session_id="s1",
            tasks=[
                ChainTask(name="t1", description="desc1", duration_minutes=10, prompt="p1"),
                ChainTask(name="t2", description="desc2", duration_minutes=20, prompt="p2"),
            ],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t + 600,
        )
        plugin._chains["persist-test"] = chain
        plugin._save_chains()

        # Verify file exists and has content
        assert os.path.exists(plugin._data_file)
        with open(plugin._data_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["id"] == "persist-test"
        assert len(data[0]["tasks"]) == 2

    def test_load_empty_file(self, plugin):
        with open(plugin._data_file, "w", encoding="utf-8") as f:
            f.write("[]")
        plugin._load_chains()
        assert len(plugin._chains) == 0

    def test_load_missing_file(self, plugin):
        # File doesn't exist
        plugin._load_chains()
        assert len(plugin._chains) == 0

    def test_load_invalid_json(self, plugin):
        with open(plugin._data_file, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")
        plugin._load_chains()
        assert len(plugin._chains) == 0

    def test_roundtrip_with_advance(self, plugin):
        now_t = time.time()
        chain = TaskChain(
            id="rt1", session_id="s1",
            tasks=[
                ChainTask(name="a", description="", duration_minutes=10, prompt=""),
                ChainTask(name="b", description="", duration_minutes=20, prompt=""),
            ],
            created_at=now_t, current_task_started_at=now_t,
            current_task_wake_at=now_t + 600,
        )
        plugin._chains["rt1"] = chain
        plugin._save_chains()

        # Load into new plugin instance
        ctx2 = MagicMock()
        plugin2 = TaskChainPlugin(context=ctx2, config={})
        plugin2._data_file = plugin._data_file
        plugin2._chains = {}
        plugin2._stop = False
        plugin2._load_chains()

        assert len(plugin2._chains) == 1
        loaded = plugin2._chains["rt1"]
        assert loaded.current_index == 0
        assert len(loaded.tasks) == 2
        assert loaded.tasks[0].name == "a"
