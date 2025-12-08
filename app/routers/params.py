# app/routers/params.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..runtime import get_manager

router = APIRouter(prefix="/sim", tags=["sim"])


class ParamsRequest(BaseModel):
    # SMD 用
    mass: float | None = None
    k: float | None = None
    c: float | None = None

    # Furuta 用（必要に応じて拡張）
    m1: float | None = None
    m2: float | None = None
    l1: float | None = None
    l2: float | None = None


@router.post("/{sim_type}/start")
async def start_sim(sim_type: str):
    try:
        mgr = get_manager(sim_type)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown simulator")
    await mgr.start()
    return {"status": "started", "sim_type": sim_type}


@router.post("/{sim_type}/stop")
async def stop_sim(sim_type: str):
    try:
        mgr = get_manager(sim_type)
    except ValueError:
        raise HTTPException(status_code=405, detail="Unknown simulator")
    await mgr.stop()
    return {"status": "stopped", "sim_type": sim_type}


@router.post("/{sim_type}/params")
async def update_params(sim_type: str, body: ParamsRequest):
    try:
        mgr = get_manager(sim_type)
    except ValueError:
        raise HTTPException(status_code=406, detail="Unknown simulator")

    kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
    await mgr.set_params(**kwargs)
    return {"status": "ok", "sim_type": sim_type}
