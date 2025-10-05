from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path

# Import sub-apps (optional, skip if deps missing)
mounted = {}
try:
    import backend.earth_main as earth_main
    mounted["earth"] = earth_main.app
except Exception:
    mounted["earth"] = None
try:
    import backend.mars_main as mars_main
    mounted["mars"] = mars_main.app
except Exception:
    mounted["mars"] = None
try:
    import backend.moon_main as moon_main
    mounted["moon"] = moon_main.app
except Exception:
    mounted["moon"] = None
try:
    import backend.space_main as space_main
    mounted["space"] = space_main.app
except Exception:
    mounted["space"] = None

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="NASA Project API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend as static site under /, handled by index redirect below
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Mount backend sub-apps under /api if available
if mounted.get("earth"):
    app.mount("/api/earth", mounted["earth"])
if mounted.get("mars"):
    app.mount("/api/mars", mounted["mars"])
if mounted.get("moon"):
    app.mount("/api/moon", mounted["moon"])
if mounted.get("space"):
    app.mount("/api/space", mounted["space"])

@app.get("/api")
def api_index():
    return {k: (f"/api/{k}" if v else None) for k, v in mounted.items()}

@app.get("/")
def root_redirect():
    # Always serve index from frontend directory
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "mounted": [k for k, v in mounted.items() if v]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


