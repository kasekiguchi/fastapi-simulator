# Path: app/api/v1/furuta.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import  Literal,List

from .simulatorOrg import simulate_furutaPendulum

router = APIRouter()


# ==== Pydantic モデル ====
TimeMode = Literal["discrete", "continuous"]
EstimatorMode = Literal["EKF", "observer"]

class SimRequest(BaseModel):
    init: List[float]     # [p, th, dp, dth]
    dt: float = 0.01
    duration: float = 10.0
    time_mode: str = "discrete"   # "discrete" | "continuous"
    estimator: str = "observer"   # "observer" | "EKF"
class SimConfig:
    dt: float = 0.01
    duration: float = 10.0
    time_mode: TimeMode = "discrete"
    estimator: EstimatorMode = "observer"

class SimState(BaseModel):
    t: float
    y: List[float]        # [p, th]
    xhat: List[float]     # [4]
    u: float


class SimResponse(BaseModel):
    states: List[SimState]


# ==== ルート定義 ====

@router.post("/simulate", response_model=SimResponse)
async def run_furuta_sim(req: SimRequest):
    """
    Furuta 振子のシミュレーションを実行して結果を返す。
    URL: POST /furutaPendulum/simulate
    """
    result = simulate_furutaPendulum(
        init=req.init,
        dt=req.dt,
        duration=req.duration,
        time_mode=req.time_mode,
        estimator=req.estimator,
    )

    # engine 側から {t: [...], y: [...], xhat: [...], u: [...]} が返る想定
    states = [
        SimState(t=t, y=y, xhat=xhat, u=u)
        for t, y, xhat, u in zip(
            result["t"], result["y"], result["xhat"], result["u"]
        )
    ]
    return SimResponse(states=states)
