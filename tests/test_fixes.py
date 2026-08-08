"""P0/P1 fixes tests.

Covers:
1. Atomic _save_chains (write tmp then rename)
2. tasks_json size limit (>50 rejected)
3. duration_minutes cap (>1440 clamped)
4. _cleanup_expired_chains removes old completed chains
5. _cleanup_expired_chains keeps active chains
6. tasks_json over 64KB returns error
7. _followup_tasks set exists on plugin
8. Session index is maintained
9. _chain_events is regular dict, not WeakValueDictionary
10. completion_callback_at = wake_at + COMPLETION_LISTEN_SECONDS
11. _followup_check receives interact_text from _wake_and_advance
12. _save_chains cleans up .tmp on error
13. _save_chains_async uses asyncio.to_thread
"""

import asyncio
import json
import os
import sys
import time
import weakref
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
PLUGIN_DIR = Path(__file__).parent.parent
ASTRBOT_PKG = PROJECT_ROOT / "astrbotplugin" / "AstrBot"
for p in [ASTRBOT_PKG, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("ASTRBOT_TEST_MODE", "true")

# Ensure plugin dir is first so `main` resolves to the plugin's main.py
sys.path.insert(0, str(PLUGIN_DIR))
sys.modules.pop("main", None)
from main import (  # noqa: E402, I001
    COMPLETION_LISTEN_SECONDS,
    ChainTask,
    TaskChain,
    TaskChainToolPlugin as TaskChainPlugin,
)


@pytest.fixture
def mock_event():
    event = MagicMock()
    event.unified_msg_origin = "test_umo_fix"
    return event


@pytest.fixture
def plugin(tmp_path):
    data_dir = tmp_path / "taskchain_data"
    data_dir.mkdir()
    data_file = data_dir / "task_chains.json"

    ctx = MagicMock()
    plugin = TaskChainPlugin(context=ctx, config={})
    plugin._data_file = str(data_file)
    plugin._chains = {}
    plugin._stop = False
    conv_mgr = MagicMock()
    conv_mgr.get_curr_conversation_id = AsyncMock(return_value="conv1")
    ctx.conversation_manager = conv_mgr
    return plugin


class TestAtomicSave:
    def test_save_chains_writes_via_tmp_and_renames(self, plugin):
        now_t = time.time()
        chain = TaskChain(
            id="atomic-1",
            session_id="s1",
            tasks=[
                ChainTask(name="t1", description="d", duration_minutes=5, prompt="p"),
            ],
            created_at=now_t,
            current_task_started_at=now_t,
            current_task_wake_at=now_t + 300,
        )
        plugin._chains["atomic-1"] = chain

        plugin._save_chains()

        assert os.path.exists(plugin._data_file)
        assert not os.path.exists(plugin._data_file + ".tmp")

        with open(plugin._data_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["id"] == "atomic-1"

    def test_save_chains_no_tmp_left_behind(self, plugin):
        plugin._save_chains()
        assert not os.path.exists(plugin._data_file + ".tmp")
        assert os.path.exists(plugin._data_file)


class TestTasksJsonSizeLimit:
    @pytest.mark.asyncio
    async def test_over_50_tasks_rejected(self, plugin, mock_event):
        tasks = [{"name": f"t{i}", "duration_minutes": 1} for i in range(51)]
        result = await plugin.chain_task(
            mock_event, action="create", tasks_json=json.dumps(tasks)
        )
        assert "最多 50 个" in result
        assert len(plugin._chains) == 0

    @pytest.mark.asyncio
    async def test_exactly_50_tasks_accepted(self, plugin, mock_event):
        tasks = [{"name": f"t{i}", "duration_minutes": 1} for i in range(50)]
        result = await plugin.chain_task(
            mock_event, action="create", tasks_json=json.dumps(tasks)
        )
        assert "任务已安排" in result
        assert len(plugin._chains) == 1


class TestDurationMinutesCap:
    def test_parse_task_caps_duration_at_1440(self, plugin):
        raw = {"name": "long_task", "duration_minutes": 9999, "prompt": ""}
        task = plugin._parse_task(raw)
        assert task is not None
        assert task.duration_minutes == 1440

    def test_parse_task_keeps_duration_within_cap(self, plugin):
        raw = {"name": "normal", "duration_minutes": 60, "prompt": ""}
        task = plugin._parse_task(raw)
        assert task is not None
        assert task.duration_minutes == 60

    def test_parse_task_caps_1441_to_1440(self, plugin):
        raw = {"name": "edge", "duration_minutes": 1441, "prompt": ""}
        task = plugin._parse_task(raw)
        assert task is not None
        assert task.duration_minutes == 1440


class TestCleanupExpiredChains:
    def test_removes_old_completed_chains(self, plugin):
        now_t = time.time()
        old_chain = TaskChain(
            id="old-completed",
            session_id="s1",
            conversation_id="c1",
            is_active=False,
            current_task_wake_at=now_t - 700,
            tasks=[ChainTask(name="t1", duration_minutes=1)],
            current_index=1,
        )
        plugin._chains["old-completed"] = old_chain
        plugin._session_index.setdefault("s1", []).append("old-completed")

        plugin._cleanup_expired_chains()

        assert "old-completed" not in plugin._chains
        assert (
            "s1" not in plugin._session_index
            or "old-completed" not in plugin._session_index.get("s1", [])
        )

    def test_keeps_active_chains(self, plugin):
        now_t = time.time()
        active_chain = TaskChain(
            id="active-chain",
            session_id="s1",
            conversation_id="c1",
            is_active=True,
            current_task_wake_at=now_t + 300,
            tasks=[ChainTask(name="t1", duration_minutes=10)],
            current_index=0,
        )
        plugin._chains["active-chain"] = active_chain

        plugin._cleanup_expired_chains()

        assert "active-chain" in plugin._chains

    def test_keeps_recent_completed_chains(self, plugin):
        now_t = time.time()
        recent_chain = TaskChain(
            id="recent-completed",
            session_id="s1",
            conversation_id="c1",
            is_active=False,
            current_task_wake_at=now_t - 100,
            tasks=[ChainTask(name="t1", duration_minutes=1)],
            current_index=1,
        )
        plugin._chains["recent-completed"] = recent_chain

        plugin._cleanup_expired_chains()

        assert "recent-completed" in plugin._chains

    def test_cleans_stale_session_system_prompts(self, plugin):
        plugin._session_system_prompts[("s1", "c1")] = "prompt1"
        plugin._session_system_prompts[("s2", "c2")] = "prompt2"

        plugin._cleanup_expired_chains()

        assert len(plugin._session_system_prompts) == 0

    @pytest.mark.asyncio
    async def test_tick_persists_expired_chain_cleanup(self, plugin):
        now_t = time.time()
        plugin._chains["old-completed"] = TaskChain(
            id="old-completed",
            session_id="s1",
            conversation_id="c1",
            is_active=False,
            finished_at=now_t - 700,
            tasks=[ChainTask(name="t1", duration_minutes=1)],
            current_index=1,
        )
        plugin._save_chains_async = AsyncMock()

        await plugin._tick()

        assert "old-completed" not in plugin._chains
        plugin._save_chains_async.assert_awaited_once()


class TestTasksJsonOver64KB:
    @pytest.mark.asyncio
    async def test_over_64kb_rejected(self, plugin, mock_event):
        big_string = "x" * 70000
        result = await plugin.chain_task(
            mock_event, action="create", tasks_json=big_string
        )
        assert "过大" in result
        assert len(plugin._chains) == 0


class TestFollowupTasksTracking:
    def test_followup_tasks_set_exists(self, plugin):
        assert hasattr(plugin, "_followup_tasks")
        assert isinstance(plugin._followup_tasks, set)


class TestSessionIndex:
    def test_session_index_exists(self, plugin):
        assert hasattr(plugin, "_session_index")
        assert isinstance(plugin._session_index, dict)

    @pytest.mark.asyncio
    async def test_session_index_maintained_on_create(self, plugin, mock_event):
        tasks = [{"name": "test_task", "duration_minutes": 5}]
        result = await plugin.chain_task(
            mock_event, action="create", tasks_json=json.dumps(tasks)
        )
        assert "任务已安排" in result
        assert len(plugin._chains) == 1

        cid = list(plugin._chains.keys())[0]
        session_id = mock_event.unified_msg_origin
        assert cid in plugin._session_index.get(session_id, [])

    def test_session_index_cleaned_on_expired(self, plugin):
        now_t = time.time()
        old_chain = TaskChain(
            id="old-idx",
            session_id="s_idx",
            conversation_id="c_idx",
            is_active=False,
            current_task_wake_at=now_t - 700,
            tasks=[ChainTask(name="t1", duration_minutes=1)],
            current_index=1,
        )
        plugin._chains["old-idx"] = old_chain
        plugin._session_index.setdefault("s_idx", []).append("old-idx")

        plugin._cleanup_expired_chains()

        assert "old-idx" not in plugin._chains
        idx_list = plugin._session_index.get("s_idx", [])
        assert "old-idx" not in idx_list

    def test_active_lookup_does_not_restore_inactive_chain(self, plugin):
        chain = TaskChain(
            id="cancelled",
            session_id="s1",
            conversation_id="c1",
            is_active=False,
            current_task_wake_at=time.time() - 60,
            tasks=[ChainTask(name="t1", duration_minutes=1)],
        )
        plugin._chains[chain.id] = chain

        active = plugin._active_chains_for_session("s1")

        assert active == []
        assert "s1" not in plugin._session_index

    def test_active_lookup_repairs_stale_nonempty_index(self, plugin):
        inactive = TaskChain(
            id="inactive",
            session_id="s1",
            is_active=False,
            tasks=[ChainTask(name="old", duration_minutes=1)],
        )
        active = TaskChain(
            id="active",
            session_id="s1",
            tasks=[ChainTask(name="current", duration_minutes=1)],
        )
        plugin._chains = {inactive.id: inactive, active.id: active}
        plugin._session_index["s1"] = [inactive.id]

        result = plugin._active_chains_for_session("s1")

        assert result == [active]
        assert plugin._session_index["s1"] == [active.id]


class TestChainEventsNotWeak:
    """P0: _chain_events must NOT be a WeakValueDictionary, since the
    source event stored there may be the only strong reference, and
    WeakValueDictionary would silently evict it and break callbacks."""

    def test_chain_events_is_regular_dict(self, plugin):
        assert isinstance(plugin._chain_events, dict)
        assert not isinstance(plugin._chain_events, weakref.WeakValueDictionary)

    def test_chain_events_holds_strong_reference(self, plugin):
        # Create an event without keeping another reference; if it were
        # WeakValueDictionary, it would be collected immediately.
        event = MagicMock()
        event.unified_msg_origin = "weak-test-session"
        plugin._chain_events["weak-test"] = event
        del event
        # Force collection
        import gc

        gc.collect()
        assert "weak-test" in plugin._chain_events
        assert plugin._chain_events["weak-test"] is not None


class TestStaleCallbackReply:
    @pytest.mark.asyncio
    async def test_discards_reply_after_callback_token_is_replaced(self, plugin):
        chain = TaskChain(
            id="callback-chain",
            session_id="s1",
            conversation_id="c1",
            pending_callback_kind="completion",
            pending_callback_token="new-token",
            tasks=[ChainTask(name="t1", duration_minutes=1)],
        )
        plugin._chains[chain.id] = chain

        event = MagicMock()
        extras = {
            "taskchain_callback": True,
            "taskchain_chain_id": chain.id,
            "taskchain_callback_token": "old-token",
        }
        event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
        result = MagicMock()
        result.chain = [object()]
        event.get_result.return_value = result

        await plugin._discard_stale_callback_reply(event)

        assert result.chain == []

    @pytest.mark.asyncio
    async def test_keeps_current_callback_reply(self, plugin):
        chain = TaskChain(
            id="callback-chain",
            session_id="s1",
            conversation_id="c1",
            pending_callback_kind="completion",
            pending_callback_token="current-token",
            tasks=[ChainTask(name="t1", duration_minutes=1)],
        )
        plugin._chains[chain.id] = chain

        event = MagicMock()
        extras = {
            "taskchain_callback": True,
            "taskchain_chain_id": chain.id,
            "taskchain_callback_token": "current-token",
        }
        event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
        result = MagicMock()
        result.chain = [object()]
        event.get_result.return_value = result

        await plugin._discard_stale_callback_reply(event)

        assert len(result.chain) == 1


class TestCompletionListenWindow:
    """P0: completion_callback_at must be wake_at + COMPLETION_LISTEN_SECONDS,
    not wake_at itself, so the callback has time to fire after the task
    wake time elapses."""

    @pytest.fixture
    def tick_plugin(self, tmp_path):
        data_dir = tmp_path / "tick_data"
        data_dir.mkdir()
        data_file = data_dir / "task_chains.json"
        ctx = MagicMock()
        plugin = TaskChainPlugin(context=ctx, config={})
        plugin._data_file = str(data_file)
        plugin._chains = {}
        ctx.get_event_queue.return_value = asyncio.Queue()
        return plugin

    @pytest.mark.asyncio
    async def test_completion_callback_at_is_wake_plus_listen_seconds(
        self, tick_plugin
    ):
        now_t = time.time()
        wake_at = now_t + max(1, COMPLETION_LISTEN_SECONDS / 2)
        chain = TaskChain(
            id="listen-window",
            session_id="s-listen",
            tasks=[
                ChainTask(name="t1", description="", duration_minutes=10, prompt="")
            ],
            created_at=now_t,
            current_task_started_at=now_t,
            current_task_wake_at=wake_at,
        )
        tick_plugin._chains["listen-window"] = chain

        await tick_plugin._tick()

        assert chain.is_active is True
        assert chain.current_index == 0
        assert chain.completion_listen_started_at > 0
        assert chain.completion_callback_at == pytest.approx(
            wake_at + COMPLETION_LISTEN_SECONDS
        )
        # And explicitly NOT equal to wake_at
        assert (
            chain.completion_callback_at - wake_at >= COMPLETION_LISTEN_SECONDS - 0.001
        )


class TestFollowupReceivesInteractText:
    """P0: _wake_and_advance must pass the interact text to _followup_check
    so the followup loop finds the correct interact message instead of
    infinitely matching the most recent assistant message."""

    @pytest.fixture
    def wake_plugin(self, tmp_path):
        data_dir = tmp_path / "wake_data"
        data_dir.mkdir()
        data_file = data_dir / "task_chains.json"
        ctx = MagicMock()
        plugin = TaskChainPlugin(context=ctx, config={})
        plugin._data_file = str(data_file)
        plugin._chains = {}
        ctx.get_event_queue.return_value = asyncio.Queue()
        conv = MagicMock()
        conv.cid = "conv1"
        conv.history = "[]"
        conv_mgr = MagicMock()
        conv_mgr.get_conversation = AsyncMock(return_value=conv)
        ctx.conversation_manager = conv_mgr
        return plugin

    def _attach_source_event(self, plugin, chain):
        if not chain.conversation_id:
            chain.conversation_id = "conv1"
        event = MagicMock()
        event.unified_msg_origin = chain.session_id
        event.message_obj = MagicMock()
        event.message_obj.type = "friend"
        event.message_obj.self_id = "bot"
        event.message_obj.session_id = chain.session_id
        event.message_obj.message_id = "123"
        event.message_obj.group = None
        event.message_obj.sender = MagicMock()
        event.get_sender_name.return_value = "tester"
        extras = {}
        event.set_extra.side_effect = extras.__setitem__
        event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
        plugin._chain_events[chain.id] = event
        return event

    @pytest.mark.asyncio
    async def test_wake_and_advance_passes_interact_text(self, wake_plugin):
        """Verify _followup_check is invoked with interact_text=... matching
        the _interact_callback_text that was queued."""
        chain = TaskChain(
            id="wake-1",
            session_id="s1",
            conversation_id="conv1",
            tasks=[
                ChainTask(name="checkin", duration_minutes=0.5, prompt=""),
                ChainTask(name="main", duration_minutes=5, prompt=""),
            ],
            created_at=time.time() - 60,
            current_task_started_at=time.time() - 60,
            current_task_wake_at=time.time() - 1,
        )
        wake_plugin._chains["wake-1"] = chain
        self._attach_source_event(wake_plugin, chain)

        with patch.object(
            wake_plugin, "_followup_check", new=AsyncMock()
        ) as mock_followup:
            await wake_plugin._wake_and_advance(chain)
            event = wake_plugin.context.get_event_queue.return_value.get_nowait()
            event._has_send_oper = True
            await wake_plugin._ack_callback_delivery(event)
            # Let the created task settle
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert mock_followup.await_count >= 1
        call_kwargs = mock_followup.call_args.kwargs
        call_args = mock_followup.call_args.args
        # interact_text must be passed (positionally or by keyword) and match
        # the interact callback text the plugin would build.
        passed_text = call_kwargs.get("interact_text")
        if passed_text is None and len(call_args) >= 3:
            passed_text = call_args[2]
        assert passed_text is not None
        expected_prefix = "[TaskChain callback]"
        assert expected_prefix in passed_text
        # Must reference the new current task, not the old one
        assert "「main」" in passed_text
        assert "「checkin」" in passed_text
        assert "任务还没有完成" in passed_text

    @pytest.mark.asyncio
    async def test_wake_and_advance_queues_interact_callback(self, wake_plugin):
        """The queued callback event's prompt should equal the interact text
        passed to _followup_check, ensuring the two stay in sync."""
        chain = TaskChain(
            id="wake-2",
            session_id="s1",
            conversation_id="conv1",
            tasks=[
                ChainTask(name="checkin", duration_minutes=0.5, prompt=""),
                ChainTask(name="main", duration_minutes=5, prompt=""),
            ],
            created_at=time.time() - 60,
            current_task_started_at=time.time() - 60,
            current_task_wake_at=time.time() - 1,
        )
        wake_plugin._chains["wake-2"] = chain
        self._attach_source_event(wake_plugin, chain)

        followup_kwargs: dict[str, Any] = {}

        async def _capture_followup(*args, **kwargs):
            followup_kwargs.update(kwargs)

        with patch.object(wake_plugin, "_followup_check", new=_capture_followup):
            await wake_plugin._wake_and_advance(chain)
            q = wake_plugin.context.get_event_queue.return_value
            assert q.qsize() == 1
            event = q.get_nowait()
            event._has_send_oper = True
            await wake_plugin._ack_callback_delivery(event)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        interact_text = followup_kwargs.get("interact_text")
        assert interact_text is not None
        # Find the provider prompt that was queued
        provider_prompt = None
        for call in event.set_extra.call_args_list:
            if call.args and call.args[0] == "provider_request":
                provider_prompt = call.args[1].prompt
                break
        assert provider_prompt is not None
        assert provider_prompt == interact_text


class TestSaveChainsTmpCleanup:
    """P0: _save_chains must remove the .tmp file when the write fails."""

    def test_save_chains_removes_tmp_on_write_error(self, plugin, monkeypatch):
        chain = TaskChain(
            id="err-test",
            session_id="s1",
            tasks=[ChainTask(name="t1", duration_minutes=1)],
            created_at=time.time(),
            current_task_started_at=time.time(),
            current_task_wake_at=time.time() + 60,
        )
        plugin._chains["err-test"] = chain

        tmp_path = plugin._data_file + ".tmp"

        def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("json.dump", _boom)

        plugin._save_chains()

        # Tmp file should have been cleaned up
        assert not os.path.exists(tmp_path)
        # Real data file should not exist either since write failed
        assert not os.path.exists(plugin._data_file)

    def test_save_chains_handles_missing_tmp_on_error(self, plugin, monkeypatch):
        chain = TaskChain(
            id="err-test-2",
            session_id="s1",
            tasks=[ChainTask(name="t1", duration_minutes=1)],
            created_at=time.time(),
            current_task_started_at=time.time(),
            current_task_wake_at=time.time() + 60,
        )
        plugin._chains["err-test-2"] = chain

        # Make replace fail AFTER a fake .tmp is left behind, then make
        # os.remove itself fail; _save_chains should still swallow that.
        tmp_path = plugin._data_file + ".tmp"

        def _write_fake(*_args, **_kwargs):
            # Pre-create the .tmp so the cleanup branch hits os.remove
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("{}")

        def _replace_boom(_src, _dst):
            raise OSError("replace failed")

        def _remove_boom(_p):
            raise OSError("remove failed")

        monkeypatch.setattr("json.dump", _write_fake)
        monkeypatch.setattr("os.replace", _replace_boom)
        monkeypatch.setattr("os.remove", _remove_boom)

        # Should not raise
        plugin._save_chains()


class TestSaveChainsAsync:
    """P0: _save_chains_async must run the synchronous save off the
    event loop thread to avoid blocking."""

    @pytest.mark.asyncio
    async def test_save_chains_async_uses_to_thread(self, plugin):
        chain = TaskChain(
            id="async-save",
            session_id="s1",
            tasks=[ChainTask(name="t1", duration_minutes=1)],
            created_at=time.time(),
            current_task_started_at=time.time(),
            current_task_wake_at=time.time() + 60,
        )
        plugin._chains["async-save"] = chain

        called = {"to_thread": False, "write": False}
        real_to_thread = asyncio.to_thread
        real_write = plugin._write_chains

        async def _spy_to_thread(func, *args, **kwargs):
            called["to_thread"] = True
            return await real_to_thread(func, *args, **kwargs)

        def _spy_write(raw):
            called["write"] = True
            return real_write(raw)

        with patch("asyncio.to_thread", side_effect=_spy_to_thread):
            plugin._write_chains = _spy_write
            await plugin._save_chains_async()

        assert called["to_thread"] is True
        assert called["write"] is True
        assert os.path.exists(plugin._data_file)
        assert not os.path.exists(plugin._data_file + ".tmp")

    @pytest.mark.asyncio
    async def test_save_chains_async_is_coroutine(self, plugin):
        assert asyncio.iscoroutinefunction(plugin._save_chains_async)


class TestFollowupCheckNoContentMatching:
    """Fix 1: _followup_check should find the first assistant message after
    start_idx without comparing content, since the prompt text sent to LLM
    will never match the LLM's generated response."""

    @pytest.fixture
    def followup_plugin(self, tmp_path):
        data_dir = tmp_path / "followup_data"
        data_dir.mkdir()
        data_file = data_dir / "task_chains.json"
        ctx = MagicMock()
        plugin = TaskChainPlugin(context=ctx, config={"interact_enabled": True})
        plugin._data_file = str(data_file)
        plugin._chains = {}
        ctx.get_event_queue.return_value = asyncio.Queue()
        conv = MagicMock()
        conv.cid = "conv1"
        conv_mgr = MagicMock()
        conv_mgr.get_conversation = AsyncMock(return_value=conv)
        ctx.conversation_manager = conv_mgr
        return plugin

    @pytest.mark.asyncio
    async def test_finds_assistant_without_content_match(self, followup_plugin):
        """_followup_check should find the first assistant msg after start_idx
        even when its content differs from interact_text."""
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "completely different response"},
        ]
        conv = MagicMock()
        conv.history = json.dumps(history)
        followup_plugin.context.conversation_manager.get_conversation = AsyncMock(
            return_value=conv
        )

        chain = TaskChain(
            id="fu-1",
            session_id="s1",
            conversation_id="conv1",
            tasks=[
                ChainTask(name="checkin", duration_minutes=0.5, prompt=""),
                ChainTask(name="main", duration_minutes=5, prompt=""),
            ],
            created_at=time.time(),
            current_task_started_at=time.time(),
            current_task_wake_at=time.time() + 300,
            followup_after_history_len=1,
            is_active=True,
        )
        followup_plugin._chains["fu-1"] = chain

        interact_text = "[TaskChain callback] some prompt text that won't match"

        with patch.object(
            followup_plugin,
            "_queue_pipeline_callback",
            new=AsyncMock(return_value=True),
        ) as mock_queue:
            with patch("random.random", return_value=0.0):
                with patch("asyncio.sleep", new=AsyncMock()):
                    await followup_plugin._followup_check(
                        chain, "s1", interact_text=interact_text
                    )

        assert mock_queue.await_count == 1
        call_args = mock_queue.call_args
        assert call_args.args[2] == "followup"


class TestInitializeDoubleStartGuard:
    """Fix 2: initialize() must not create a second scheduler task if one is
    already running."""

    @pytest.fixture
    def init_plugin(self, tmp_path):
        data_dir = tmp_path / "init_data"
        data_dir.mkdir()
        data_file = data_dir / "task_chains.json"
        ctx = MagicMock()
        plugin = TaskChainPlugin(context=ctx, config={})
        plugin._data_file = str(data_file)
        plugin._chains = {}
        return plugin

    @pytest.mark.asyncio
    async def test_no_double_start(self, init_plugin):
        async def dummy_loop():
            await asyncio.sleep(999)

        task = asyncio.create_task(dummy_loop())
        init_plugin._scheduler_task = task

        await init_plugin.initialize()

        assert init_plugin._scheduler_task is task

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_starts_when_no_task(self, init_plugin):
        init_plugin._scheduler_task = None

        with patch.object(
            init_plugin,
            "_scheduler_loop",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            await init_plugin.initialize()
            assert init_plugin._scheduler_task is not None
            init_plugin._scheduler_task.cancel()
            try:
                await init_plugin._scheduler_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_restarts_when_done(self, init_plugin):
        async def _noop():
            pass

        done_task = asyncio.create_task(_noop())
        await done_task
        assert done_task.done()
        init_plugin._scheduler_task = done_task

        with patch.object(
            init_plugin,
            "_scheduler_loop",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            await init_plugin.initialize()
            assert init_plugin._scheduler_task is not done_task
            init_plugin._scheduler_task.cancel()
            try:
                await init_plugin._scheduler_task
            except asyncio.CancelledError:
                pass


class TestTickUsesSaveChainsAsync:
    """Fix 4: _tick and _wake_and_advance should use _save_chains_async instead
    of _save_chains in async contexts."""

    def test_tick_source_uses_save_chains_async(self):
        import inspect

        source = inspect.getsource(TaskChainPlugin._tick)
        assert "_save_chains_async" in source
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if "self._save_chains()" in stripped and "async" not in stripped:
                pytest.fail(
                    f"_tick still calls self._save_chains() synchronously: {stripped}"
                )

    def test_wake_and_advance_source_uses_save_chains_async(self):
        import inspect

        source = inspect.getsource(TaskChainPlugin._wake_and_advance)
        assert "_save_chains_async" in source
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if "self._save_chains()" in stripped and "async" not in stripped:
                pytest.fail(
                    f"_wake_and_advance still calls self._save_chains() synchronously: {stripped}"
                )


class TestLoadChainsRestartRecovery:
    """活动任务重启后必须恢复；只有已结束的旧记录才允许过期清理。"""

    def test_load_chains_keeps_chain_with_recent_completion_callback(self, plugin):
        now_t = time.time()
        data = [
            {
                "id": "exp-1",
                "session_id": "s1",
                "conversation_id": "c1",
                "system_prompt": "",
                "current_index": 0,
                "is_active": True,
                "created_at": now_t - 600,
                "current_task_started_at": now_t - 600,
                "current_task_wake_at": now_t - 400,
                "completion_listen_started_at": now_t - 100,
                "completion_callback_at": now_t - 50,
                "callback_retry_count": 0,
                "followup_after_history_len": 0,
                "event_metadata": {},
                "tasks": [
                    {
                        "name": "t1",
                        "description": "",
                        "duration_minutes": 5,
                        "prompt": "",
                    }
                ],
            }
        ]
        with open(plugin._data_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        plugin._chains = {}
        plugin._session_index = {}
        plugin._load_chains()

        assert "exp-1" in plugin._chains

    def test_load_chains_keeps_overdue_active_chain(self, plugin):
        now_t = time.time()
        data = [
            {
                "id": "exp-2",
                "session_id": "s1",
                "conversation_id": "c1",
                "system_prompt": "",
                "current_index": 0,
                "is_active": True,
                "created_at": now_t - 1000,
                "current_task_started_at": now_t - 1000,
                "current_task_wake_at": now_t - 500,
                "completion_listen_started_at": now_t - 400,
                "completion_callback_at": now_t - 400,
                "callback_retry_count": 0,
                "followup_after_history_len": 0,
                "event_metadata": {},
                "tasks": [
                    {
                        "name": "t1",
                        "description": "",
                        "duration_minutes": 5,
                        "prompt": "",
                    }
                ],
            }
        ]
        with open(plugin._data_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        plugin._chains = {}
        plugin._session_index = {}
        plugin._load_chains()

        assert "exp-2" in plugin._chains
        assert plugin._chains["exp-2"].is_active is True
        assert "exp-2" in plugin._session_index["s1"]

    def test_load_chains_expires_old_inactive_chain(self, plugin):
        now_t = time.time()
        data = [
            {
                "id": "exp-3",
                "session_id": "s1",
                "conversation_id": "c1",
                "current_index": 1,
                "is_active": False,
                "created_at": now_t - 2000,
                "current_task_wake_at": now_t - 1500,
                "finished_at": now_t - 1000,
                "tasks": [{"name": "t1", "duration_minutes": 5}],
            }
        ]
        with open(plugin._data_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        plugin._chains = {}
        plugin._session_index = {}
        plugin._load_chains()

        assert "exp-3" not in plugin._chains
