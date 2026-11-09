"""Exercise the real voice hub and scheduler together, without sockets or model calls."""
import asyncio
import importlib.util
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

import pytest

from heartbeat2.model import Session, Wake
from heartbeat2.poller import Poller
from heartbeat2.policy import Policy
from heartbeat2.store import Store
from test_engine import GatewayCtx


@pytest.fixture
def voice_hub():
    root = Path(__file__).resolve().parents[2] / 'hermes-converse'
    if not root.is_dir():
        pytest.skip('optional hermes-converse sibling is not installed')
    name = 'hotpatch_converse'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / '__init__.py',
                                                    submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from hotpatch_converse.conversation import ConversationHub
    return ConversationHub()


def device():
    return SimpleNamespace(stopped=False, _turn_active=threading.Event(), _tail_active=False,
                           capture_pending=lambda: False, transcripts=queue.Queue())


@pytest.mark.asyncio
async def test_voice_wake_identity_verdict_delivery_and_offline_deferral(tmp_path, clock, monkeypatch, voice_hub):
    from heartbeat2 import tool
    ctx = GatewayCtx(tmp_path)
    ctx._manager._heartbeat2_voice_hub = voice_hub
    state = {'HERMES_SESSION_PLATFORM': 'api_server', 'HERMES_SESSION_ID': 'voice-root'}
    monkeypatch.setattr(tool, 'session_env', lambda name: state.get(name, ''))
    store = Store(tmp_path / 'pulse.db', clock)
    e = Poller(ctx, store, Policy(clock=clock, quiet_hours='', timezone='UTC'), clock,
               session_key_fn=lambda: state.get('run', 'run-1'), paused=lambda: False,
               compression_tip=lambda sid: sid, jitter=lambda: 0)
    live = device()
    attachment = voice_hub.attach('voice-root', live)
    replies = iter([False, True])
    turns = []

    async def wake(text):
        notify = next(replies)
        turn = 'wake-' + str(len(turns))
        turns.append(turn)
        e.intake.pre_llm_call(session_id='voice-root', turn_id=turn, user_message=text, platform='api_server')
        assert e.intake.flights['voice-root'].invalid is False
        assert e.tool.handle(dict(action='respond', outcome='done', notify=notify,
                                 summary='The timer is ready.'), session_id='voice-root')['ok']
        e.intake.post_api_request(session_id='voice-root', turn_id=turn,
                                  usage={'prompt_tokens': 10, 'completion_tokens': 5})
        final = e.intake.transform_llm_output(response_text='The timer is ready.',
                                              session_id='voice-root', turn_id=turn)
        if final is None:
            final = 'The timer is ready.'
        voice_hub.on_response(session_id='voice-root', turn_id=turn, assistant_response=final)
        e.intake.on_session_end(session_id='voice-root', turn_id=turn, completed=True)

    voice_hub.bind_wakes(attachment, wake)
    try:
        # Approval run IDs rotate, while the voice destination and schedule identity remain stable.
        first = e.tool.identity()
        state['run'] = 'run-2'
        assert e.tool.identity().session_key == first.session_key == 'heartbeat2:voice:voice-root'
        assert first.platform == 'api_server' and first.chat_type == 'dm'
        with voice_hub.owned_run(attachment, 'Remind me in five minutes.'):
            e.intake.pre_llm_call(session_id='voice-root', turn_id='owner',
                                  user_message='Remind me in five minutes.', platform='api_server')
            result = e.tool.handle(dict(action='create', note='Say the timer is ready.',
                                       trigger={'kind': 'at', 'at': 'in 5m'}, owner_asked=True),
                                   session_id='voice-root')
            assert result['armed'] and result['wake']['owner_requested']
            e.intake.on_session_end(session_id='voice-root', turn_id='owner', completed=True)
        w = store.get_wake(result['wake']['id'])
        clock.advance(301)
        for index in range(2):
            if index:
                w = store.create_wake(Wake(first.session_key, 'Notify the owner.', 'at',
                    platform='api_server', chat_type='dm', chat_id='voice-root',
                    armed_session_id='voice-root', next_due_at=clock.now(), expires_at=clock.now()+3600))
                clock.advance(601)
            key = 't:' + str(clock.now())
            store.add_occurrence(w.id, key)
            e.dispatch(store.get_occurrence(w.id, key), cache_line='')
            for _ in range(8):
                await asyncio.sleep(0)
            occurrence = store.get_occurrence(w.id, key)
            assert occurrence.state == 'done'
            assert occurrence.delivered == bool(index)
            if not index:
                assert live.transcripts.empty()
        output = live.transcripts.get_nowait()
        assert output.text == 'The timer is ready.'
        output.on_dequeue()
        # Nothing went through the messaging adapter or to a second device/profile.
        assert ctx.calls == []
        assert len(turns) == 2
        offline = store.create_wake(Wake(first.session_key, 'Wait for reconnection.', 'at',
            platform='api_server', chat_type='dm', chat_id='voice-root',
            armed_session_id='voice-root', next_due_at=clock.now(), expires_at=clock.now()+3600))
        store.add_occurrence(offline.id, 't:offline')
        voice_hub.detach('voice-root', attachment)
        e.dispatch(store.get_occurrence(offline.id, 't:offline'), cache_line='')
        occurrence = store.get_occurrence(offline.id, 't:offline')
        assert occurrence.state == 'pending' and occurrence.error == 'voice_disconnected'
    finally:
        voice_hub.unbind_wakes(attachment)
        e.close()
        store.close()


@pytest.mark.asyncio
async def test_busy_wrong_profile_and_unattached_api_remain_blocked(tmp_path, clock, monkeypatch, voice_hub):
    from heartbeat2 import tool
    ctx = GatewayCtx(tmp_path)
    ctx._manager._heartbeat2_voice_hub = voice_hub
    state = {'HERMES_SESSION_PLATFORM': 'api_server', 'HERMES_SESSION_ID': 'unattached'}
    monkeypatch.setattr(tool, 'session_env', lambda name: state.get(name, ''))
    store = Store(tmp_path / 'pulse.db', clock)
    e = Poller(ctx, store, Policy(clock=clock, quiet_hours=''), clock,
               session_key_fn=lambda: 'run-ordinary', paused=lambda: False)
    called = []

    async def wake(text):
        called.append(text)

    item = voice_hub.attach('connected', device())
    voice_hub.bind_wakes(item, wake)
    try:
        result = e.tool.handle(dict(action='create', note='Check.', trigger={'kind':'at','at':'in 5m'}))
        assert result['error'] == 'non_gateway'
        assert voice_hub.voice_target('unattached') is None
        item.session._turn_active.set()
        assert not voice_hub.inject_wake('connected', 'wake')
        item.session._turn_active.clear()
        item.session.capture_pending = lambda: True
        assert not voice_hub.inject_wake('connected', 'wake')
        item.session.capture_pending = lambda: False
        # A pending wake cannot race a second scheduler claim.
        assert voice_hub.inject_wake('connected', 'wake')
        assert not voice_hub.inject_wake('connected', 'duplicate')
        # A disconnect before the loop picks up the callback performs no run.
        voice_hub.unbind_wakes(item)
        for _ in range(4):
            await asyncio.sleep(0)
        assert called == []
        other = GatewayCtx(tmp_path / 'other-profile')
        assert not hasattr(other._manager, '_heartbeat2_voice_hub')
    finally:
        voice_hub.unbind_wakes(item)
        e.close()
        store.close()
