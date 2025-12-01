# Path: app/furuta_pendulum/main.py
from __future__ import annotations

from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .simulator import simulate_furuta, SimConfig


app = FastAPI(
    title="Furuta Pendulum API",
    version="0.1.0",
    description="Furuta pendulum simulation backend (for web frontends).",
)

origins = [
    "http://localhost:3000",
    # Next.js をデプロイした URL に合わせて書き換え
    "https://acsl-simulator.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 開発中なら ["*"] でもOK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimRequest(BaseModel):
    init: List[float]  # [p, th, dp, dth]
    dt: float = 0.01
    duration: float = 10.0
    time_mode: str = "discrete"   # "discrete" or "continuous"
    estimator: str = "observer"   # "observer" or "EKF"


class SimState(BaseModel):
    t: float
    y: List[float]
    xhat: List[float]
    u: float


class SimResponse(BaseModel):
    states: List[SimState]


@app.get("/")
async def root():
    return {"message": "Furuta Pendulum API is running"}


@app.post("/simulate", response_model=SimResponse)
async def run_sim(req: SimRequest):
    config = SimConfig(
        dt=req.dt,
        duration=req.duration,
        time_mode=req.time_mode,   # 簡略化：Literalチェックは省略
        estimator=req.estimator,
    )

    result = simulate_furuta(
        init=np.array(req.init, dtype=float),
        config=config,
    )

    states: List[SimState] = []
    for i, t in enumerate(result["t"]):
        states.append(
            SimState(
                t=t,
                y=result["y"][i],
                xhat=result["xhat"][i],
                u=result["u"][i],
            )
        )

    return SimResponse(states=states)
