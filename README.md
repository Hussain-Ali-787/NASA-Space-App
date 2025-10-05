# NASA_PROJECT - Run

This project runs with a single Conda environment that includes all required scientific dependencies.

## Prerequisites
- Conda (Miniconda/Anaconda). Verify with: `conda --version`.
- Datasets present in the repo (required):
  - `MODIS_DATA/` with MODIS HDF/TIF/PNG exports (Earth)
  - `Mars_datasets/` with CTX/HiRISE/MOLA/THEMIS datasets
  - `Moon_datasets/` with NAC/WAC/LOLA datasets
  - `Deepspace_datasets/` for TESS/Hubble/JWST/OSIRIS/ACE (optional)

## Setup and Run (Windows/macOS/Linux)
1) Create/update the environment from `backend/environment.yml`:
   - `conda env create -f backend/environment.yml -n nasa` (first time)
   - or `conda env update -f backend/environment.yml -n nasa` (subsequent runs)
2) Start the server (no manual activation needed):
   - `conda run -n nasa python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend`

Shortcuts:
- Windows PowerShell: `./deploy.ps1 -Port 8000`
- Bash: `bash start.sh`

## Open in browser
- Frontend: `http://127.0.0.1:8000/`
- API index: `http://127.0.0.1:8000/api`
- Health: `http://127.0.0.1:8000/health`

## Configuration
- `EARTH_DATA_PATH` (optional): set to a custom path for MODIS HDF/TIF files.

## Structure
- `frontend/` static HTML/CSS
- `backend/` FastAPI gateway (`backend/main.py`) and feature modules
- `backend/environment.yml` Conda environment with all dependencies
