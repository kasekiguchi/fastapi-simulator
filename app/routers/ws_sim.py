# app/routers/ws_sim.py
import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..runtime import create_manager
from ..simulators.base import SimState
from ..simulators import SIM_REGISTRY

logger = logging.getLogger("ws_sim")

router = APIRouter(tags=["ws"])


def _simstate_to_dict(simState: SimState) -> dict:
    return simState.__dict__.copy()


def _ensure_time_mode(params: dict, src: dict) -> dict:
    """Inject time mode aliases into params if present in either dict."""
    if not isinstance(params, dict):
        return params
    tm = params.get("timeMode") or params.get("time_mode") or src.get("time_mode") or src.get("timeMode")
    if tm is None:
        return params
    if params.get("timeMode") == tm and params.get("time_mode") == tm:
        return params
    return {**params, "timeMode": tm, "time_mode": tm}


def _extract_duration(msg: dict) -> Optional[float]:
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else None
    val = None
    for key in ("duration", "simTime", "sim_time"):
        if val is not None:
            break
        if payload and key in payload:
            val = payload.get(key)
        elif key in msg:
            val = msg.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


async def _safe_send(ws: WebSocket, text: str):
    try:
        await ws.send_text(text)
    except Exception as e:
        print("[ws_sim] send error:", e, flush=True)


@router.websocket("/ws/{sim_type}")
async def sim_ws(websocket: WebSocket, sim_type: str):
    """
    WebSocket 接続ごとに独立した SimulationManager を生成する。
    複数クライアントが同じ sim_type に接続しても、各自独立したシミュレータを持つ。
    """
    sim_type = sim_type.lower()
    print(f"[ws_sim] connect sim_type={sim_type}")
    try:
        mgr = create_manager(sim_type)
    except ValueError as e:
        print(f"[ws_sim] {e} unknown sim_type={sim_type}, registry={list(SIM_REGISTRY.keys())}")
        await websocket.accept()
        await websocket.close(code=1003)
        return

    await websocket.accept()
    logger.info("connected %s from %s", sim_type, websocket.client)

    # この接続専用のリスナーを登録
    def state_listener(state: SimState):
        msg = json.dumps({"type": "state", "payload": _simstate_to_dict(state)})
        asyncio.create_task(_safe_send(websocket, msg))

    async def stop_listener(auto: bool, state: SimState | None):
        payload = {
            "auto": bool(auto),
            "t": getattr(state, "t", None) if state is not None else None,
            "limit": mgr.current_limit(),
        }
        await _safe_send(websocket, json.dumps({"type": "stopped", "payload": payload}))
        if auto:
            trace = await mgr.get_trace()
            if trace is not None:
                await _safe_send(websocket, json.dumps({"type": "trace", "payload": trace}))

    mgr.add_listener(state_listener)
    mgr.add_stop_listener(stop_listener)

    try:
        while True:
            text = await websocket.receive_text()
            msg = json.loads(text)
            logger.info("recv %s: %s", sim_type, msg)
            print(f"[ws_sim] recv {sim_type}: {msg}")
            msg_type = msg.get("type")

            if msg_type == "click":
                payload = {}
                if isinstance(msg.get("payload"), dict):
                    payload.update(msg["payload"])
                for k in ("force", "torque"):
                    if k in msg:
                        payload[k] = msg[k]
                await mgr.apply_impulse(**payload)
            elif msg_type == "start":
                await mgr.start(duration=_extract_duration(msg))
            elif msg_type == "stop":
                await mgr.stop()
                trace = await mgr.get_trace()
                if trace is not None:
                    await _safe_send(websocket, json.dumps({"type": "trace", "payload": trace}))
            elif msg_type == "reset":
                await mgr.reset()
            elif msg_type == "set_initial":
                initial = msg.get("initial") or msg.get("payload") or msg
                await mgr.set_initial(**(initial if isinstance(initial, dict) else {}))
            elif msg_type == "set_params":
                params = msg.get("params") or msg.get("payload") or {}
                await mgr.set_params(**params)
            elif msg_type == "set_control_params":
                control_params = msg.get("control_params") or msg.get("payload") or msg
                control_params = _ensure_time_mode(control_params, msg)
                await mgr.set_control_params(control_params=control_params)
            elif msg_type == "set_estimator_params":
                estimator_params = msg.get("estimator_params") or msg.get("payload") or msg
                estimator_params = _ensure_time_mode(estimator_params, msg)
                if hasattr(mgr, "set_estimator_params"):
                    await mgr.set_estimator_params(estimator_params=estimator_params)
            elif msg_type == "set_reference":
                reference = msg.get("reference") or msg.get("payload") or msg
                if hasattr(mgr, "set_reference"):
                    await mgr.set_reference(reference=reference)
            elif msg_type == "set_exp_mode":
                await mgr.set_exp_mode(msg.get("expMode"))

    except WebSocketDisconnect:
        print(f"[ws_sim] client disconnected: {sim_type}")
        logger.info("disconnected %s", sim_type)
    finally:
        # 接続が切れたらシミュレーションを必ず停止（タスクリーク防止）
        try:
            await mgr.stop()
        except Exception:
            pass
