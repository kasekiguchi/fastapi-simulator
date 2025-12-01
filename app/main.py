# Path: app/main.py
from fastapi import FastAPI
from .api.v1.furutaPendulum import router as furuta_router
# from .api.v1.other_sim import router as other_sim_router

app = FastAPI(
    title="Simulator API",
    version="1.0.0",
    description="Multiple simulators (Furuta, others) backend.",
)

# CORS 設定（必要なら）
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",             # Next.js ローカル
    "https://acsl-simulator.vercel.app",  # デプロイ先
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 開発中は ["*"] でもOK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルート登録（prefix で URL を分ける）
app.include_router(furuta_router, prefix="/furutaPendulum", tags=["furutaPendulum"])
# app.include_router(other_sim_router, prefix="/other-sim", tags=["other-sim"])


@app.get("/")
async def root():
    return {"message": "Simulator API is running"}
