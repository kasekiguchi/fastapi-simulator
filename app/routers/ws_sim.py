# app/routers/ws_sim.py
import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from ..runtime import get_manager
from ..simulators.base import SimState
import logging
from ..simulators import SIM_REGISTRY 

logger = logging.getLogger("ws_sim")

router = APIRouter(tags=["ws"])

# sim_type -> set(WebSocket)
_clients: Dict[str, Set[WebSocket]] = {}


def _simstate_to_dict(simState: SimState) -> dict:
    # dataclass をそのまま dict に
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


async def _safe_send(ws: WebSocket, text: str):
    try:
        await ws.send_text(text)
    except Exception as e:
        print("[ws_sim] send error:", e)


def _attach_broadcast_listener(sim_type: str):
    mgr = get_manager(sim_type)

    def listener(state: SimState):
        clients = _clients.get(sim_type)
        if not clients:
            return
        msg = json.dumps({"type": "state", "payload": _simstate_to_dict(state)})
        # print(f"[ws_sim] send {sim_type}: {msg.payload}")
        for ws in list(clients):
            asyncio.create_task(_safe_send(ws, msg))

    mgr.add_listener(listener)


@router.websocket("/ws/{sim_type}")
async def sim_ws(websocket: WebSocket, sim_type: str):
    print(f"[ws_sim] connect sim_type={sim_type}")
    try:
        mgr = get_manager(sim_type)
    except ValueError as e:
        print(f"[ws_sim] {e} unknown sim_type={sim_type}, registry={list(SIM_REGISTRY.keys())}")
        await websocket.accept()
        await websocket.close(code=1003)
        # raise HTTPException(status_code=404, detail="Unknown simulator")
        return 

    await websocket.accept()
    logger.info("connected %s from %s", sim_type, websocket.client)
    sim_type = sim_type.lower()
    if sim_type not in _clients:
        _clients[sim_type] = set()
        _attach_broadcast_listener(sim_type)

    _clients[sim_type].add(websocket)
    print(f"[ws_sim] client connected: {sim_type}, total={len(_clients[sim_type])}")

    try:
        while True:
            text = await websocket.receive_text()
            msg = json.loads(text)
            logger.info("recv %s: %s", sim_type, msg)  # ここで丸ごと出力
            print(f"[ws_sim] recv {sim_type}: {msg}")
            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type == "click":
                # 優先: 直下キー force/torque, 後方互換で payload も見る
                payload = {}
                if isinstance(msg.get("payload"), dict):
                    payload.update(msg["payload"])
                for k in ("force", "torque"):
                    if k in msg:
                        payload[k] = msg[k]
                await mgr.apply_impulse(**payload)
            elif msg_type == "start":
                await mgr.start()
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
            elif msg_type == "set_exp_mode":
                await mgr.set_exp_mode(msg.get("expMode"))

    except WebSocketDisconnect:
        print(f"[ws_sim] client disconnected: {sim_type}")
        logger.info("disconnected %s", sim_type)
    finally:
        _clients[sim_type].discard(websocket)
