

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from pathlib import Path

app = FastAPI(title="Earth Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data path within the repository (adjustable via env var EARTH_DATA_PATH)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.environ.get("EARTH_DATA_PATH", str(REPO_ROOT / "MODIS_DATA"))

@app.get("/earth/files")
def list_files():
    """List all HDF/TIF files available in MODIS_data folder."""
    try:
        files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith((".hdf", ".tif", ".tiff", ".png"))]
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}

@app.get("/earth/image/{filename}")
def get_image(filename: str):
    """Serve a PNG/TIF image file directly from the MODIS data directory."""
    filepath = os.path.join(DATA_PATH, filename)
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    mt = "image/png" if filename.lower().endswith(".png") else "image/tiff"
    return FileResponse(filepath, media_type=mt)

@app.get("/earth/datasets/{filename}")
def list_datasets(filename: str):
    """List subdatasets (variables) inside an HDF file."""
    filepath = os.path.join(DATA_PATH, filename)
    try:
        from osgeo import gdal  # lazy import to avoid hard dependency for file listing
    except Exception:
        return {"error": "GDAL not installed"}
    ds = gdal.Open(filepath)
    if not ds:
        return {"error": f"Could not open {filename}"}
    return {"subdatasets": [s[0] for s in ds.GetSubDatasets()]}

@app.get("/earth/data/{filename}")
def get_dataset(filename: str, variable: str):
    """Extract a dataset/variable from HDF into array form."""
    filepath = os.path.join(DATA_PATH, filename)
    try:
        from osgeo import gdal  # lazy import
        sub = gdal.Open(f'HDF4_EOS:EOS_GRID:"{filepath}":mod08:{variable}')
        if not sub:
            return {"error": f"Variable {variable} not found in {filename}"}
        arr = sub.ReadAsArray()
        return {
            "variable": variable,
            "shape": arr.shape,
            "sample": arr.tolist()[:5]  # return only first 5 rows as preview
        }
    except Exception as e:
        return {"error": str(e)}
@app.get("/earth/export/{filename}")
def export_dataset(filename: str, variable: str):
    """Export HDF variable as global PNG with georeferencing."""
    filepath = os.path.join(DATA_PATH, filename)
    output_png = os.path.join(DATA_PATH, f"export_{variable}.png")

    try:
        from osgeo import gdal  # lazy import
        subdataset = f'HDF4_EOS:EOS_GRID:"{filepath}":mod08:{variable}'
        # Translate with georeferencing (global extent)
        gdal.Translate(
            output_png,
            subdataset,
            format="PNG",
            outputBounds=[-180, -90, 180, 90],  # lon/lat bounds
            outputSRS="EPSG:4326"
        )
        return FileResponse(output_png, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}
