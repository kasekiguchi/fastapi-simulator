# Path: app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import params, ws_sim

app = FastAPI(
    title="Simulator API",
    version="1.0.0",
    description="Multiple simulators (Furuta, others) backend.",
)

origins = [
    "http://localhost:3000",             # Next.js ローカル
    "http://localhost:3001",             # Next.js ローカル
    "https://acsl-simulator.vercel.app",  # デプロイ先
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 開発中は ["*"] でもOK  # 例: ["https://your-next-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルート登録（prefix で URL を分ける）
app.include_router(params.router)
app.include_router(ws_sim.router)
