# app/routers/ws_sim.py
import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from ..runtime import get_manager
from ..simulators.base import SimState

router = APIRouter(tags=["ws"])

# sim_type -> set(WebSocket)
_clients: Dict[str, Set[WebSocket]] = {}


def _state_to_dict(state: SimState) -> dict:
    # dataclass をそのまま dict に
    return state.__dict__.copy()


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
        msg = json.dumps({"type": "state", "payload": _state_to_dict(state)})
        for ws in list(clients):
            asyncio.create_task(_safe_send(ws, msg))

    mgr.add_listener(listener)


@router.websocket("/ws/{sim_type}")
async def sim_ws(websocket: WebSocket, sim_type: str):
    try:
        mgr = get_manager(sim_type)
    except ValueError:
        await websocket.accept()
        await websocket.close(code=1003)
        raise HTTPException(status_code=404, detail="Unknown simulator")

    await websocket.accept()

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
            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type == "click":
                # SMD: {force}, Furuta: {torque} など
                await mgr.apply_impulse(**payload)
            elif msg_type == "start":
                await mgr.start()
            elif msg_type == "stop":
                await mgr.stop()

    except WebSocketDisconnect:
        print(f"[ws_sim] client disconnected: {sim_type}")
    finally:
        _clients[sim_type].discard(websocket)
