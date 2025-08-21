# Melon Calls Dashboard (Streamlit) — Render Free Deployment

This repo contains a Streamlit app that fetches data from QMP reporting and shows aggregated metrics with two-level segmentation.

## One-click setup (Render)

1. Push these files to a new Git repo (include `CallAnalysisApp.py`, `requirements.txt`, `Procfile`, and `render.yaml`).  
2. In Render, click **New +** → **Web Service**, connect your repo.
3. Render will auto-detect the Python environment.
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run CallAnalysisApp.py --server.headless true --server.address 0.0.0.0 --server.port $PORT`
4. Choose the **Free** plan.
5. In the service **Environment** tab, add these environment variables:
   - `PYTHON_VERSION=3.11.9` (optional but recommended)
   - `MELON_APP_PASSWORD` (required — password gate for the app)
   - `MELON_CLIENT_ID` (required — QMP API client_id)
   - `MELON_CLIENT_SECRET` (required — QMP API client_secret)
6. Deploy. Once healthy, open the app URL and enter your password.

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

export MELON_APP_PASSWORD="your_password"
export MELON_CLIENT_ID="your_client_id"
export MELON_CLIENT_SECRET="your_client_secret"

streamlit run CallAnalysisApp.py
```

## Notes
- The app binds to `0.0.0.0` and uses the `$PORT` injected by Render.
- The Free plan sleeps after inactivity; first request may take longer.
- No Google OAuth — access is gated by `MELON_APP_PASSWORD` only.
