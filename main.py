from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import numpy as np  # ここは実際のシミュレーションで使うことを想定

app = FastAPI(
    title="Furuta Pendulum API",
    version="0.1.0",
    description="Furuta pendulum simulation backend for Next.js frontend.",
)

# --- CORS 設定（本番の Origin を書く） ---
origins = [
    "http://localhost:3000",
    "https://acsl-simulator.vercel.app",  # 実際のドメインに合わせる
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 必要なら ["*"] でもOK（最初は雑に許可でも）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 型定義 ---

class InitialState(BaseModel):
    theta: float          # アーム角
    alpha: float          # 振り子角
    theta_dot: float
    alpha_dot: float

class SimParams(BaseModel):
    dt: float = 0.01
    duration: float = 5.0

class SimRequest(BaseModel):
    state0: InitialState
    params: SimParams

class SimState(BaseModel):
    t: float
    theta: float
    alpha: float
    theta_dot: float
    alpha_dot: float

class SimResponse(BaseModel):
    states: List[SimState]


# --- ダミーのシミュレーション（あとで本物に差し替え） ---
def simulate_furuta(state0: InitialState, params: SimParams) -> list[SimState]:
    dt = params.dt
    n_steps = int(params.duration / dt)
    states: list[SimState] = []

    theta = state0.theta
    alpha = state0.alpha
    theta_dot = state0.theta_dot
    alpha_dot = state0.alpha_dot

    for i in range(n_steps):
        t = i * dt

        # ここに本物のダイナミクスを書く
        # 今はとりあえずゆっくり減衰するような適当なモデル
        theta += dt * theta_dot
        alpha += dt * alpha_dot
        theta_dot *= 0.999
        alpha_dot *= 0.999

        states.append(
            SimState(
                t=t,
                theta=theta,
                alpha=alpha,
                theta_dot=theta_dot,
                alpha_dot=alpha_dot,
            )
        )
    return states


@app.post("/simulate", response_model=SimResponse)
async def run_sim(req: SimRequest):
    states = simulate_furuta(req.state0, req.params)
    return SimResponse(states=states)


@app.get("/")
async def root():
    return {"message": "Furuta Pendulum API is running"}
