# fastapi-simulator

## 環境構築

```bash
python3 -m venv venv
source venv/bin/activate
pip install uvicorn fastapi numpy scipy
```

```bash
source venv/bin/activate
uvicorn app.main:app --reload  --log-level debug
```