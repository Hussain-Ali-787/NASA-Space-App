from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

app = FastAPI(title="Deep Space API")

# Directories
TESS_DIR = Path("/mnt/c/NASA_Project/Deepspace_datasets/TESS/s0027/cam1-ccd1")
PREVIEW_DIR = Path("previews/tess")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# 🔹 Helper: safely load first 2D image HDU from FITS
def load_image_hdu(filepath):
    with fits.open(filepath) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "data") and hdu.data is not None and hdu.data.ndim == 2:
                return np.nan_to_num(hdu.data.astype(np.float32))
    raise ValueError("No 2D image HDU found in FITS file")


# 1️⃣ Endpoint: List available TESS FITS files
@app.get("/deep-space/tess/files")
def tess_files():
    files = [f.name for f in TESS_DIR.glob("*.fits")]
    return {"files": files}


# 2️⃣ Endpoint: Preview of a TESS FITS star field
@app.get("/deep-space/tess/preview/{filename}")
def tess_preview(filename: str):
    filepath = TESS_DIR / filename
    try:
        arr = load_image_hdu(filepath)

        # Normalize and clip for visualization
        low, high = np.percentile(arr, (2, 98))
        arr = np.clip(arr, low, high)
        arr = (arr - arr.min()) / (arr.max() - arr.min())

        preview_path = PREVIEW_DIR / f"{filename}_preview.png"
        plt.imsave(preview_path, arr, cmap="gray")
        return FileResponse(preview_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TESS preview error: {str(e)}")


# 3️⃣ Endpoint: Crop zoom into a star cluster
@app.get("/deep-space/tess/crop/{filename}")
def tess_crop(filename: str, x: int, y: int, size: int = 200):
    filepath = TESS_DIR / filename
    try:
        arr = load_image_hdu(filepath)

        cropped = arr[y:y+size, x:x+size]
        if cropped.size == 0:
            raise ValueError("Requested crop region is outside image bounds")

        cropped = (cropped - cropped.min()) / (cropped.max() - cropped.min())

        crop_path = PREVIEW_DIR / f"{filename}crop{x}_{y}.png"
        plt.imsave(crop_path, cropped, cmap="gray")
        return FileResponse(crop_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TESS crop error: {str(e)}")


# 4️⃣ Endpoint: Extract light curve for a given star (pixel coords)
@app.get("/deep-space/tess/lightcurve/{filename}")
def tess_lightcurve(filename: str, x: int, y: int, aperture: int = 3):
    filepath = TESS_DIR / filename
    try:
        arr = load_image_hdu(filepath)

        star_flux = arr[y-aperture:y+aperture, x-aperture:x+aperture]
        if star_flux.size == 0:
            raise ValueError("Requested aperture is outside image bounds")

        flux = np.sum(star_flux)

        # Simulated time-series (placeholder)
        time = np.arange(0, 100, 1)
        brightness = flux * (1 + 0.01*np.sin(time/10))

        return JSONResponse({
            "time": time.tolist(),
            "brightness": brightness.tolist(),
            "note": "Simulated light curve – replace with real TESS time-series"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TESS light curve error: {str(e)}")


# 5️⃣ Endpoint: Compare TESS vs Hubble (placeholder overlay)
@app.get("/deep-space/tess/compare/{filename}")
def tess_compare(filename: str):
    try:
        return JSONResponse({
            "tess": f"/deep-space/tess/preview/{filename}",
            "hubble": "/deep-space/hubble/sample.png",
            "note": "Overlay star cluster brightness between TESS & Hubble"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TESS compare error: {str(e)}")


# 6️⃣ Endpoint: Detect stars in a TESS image (basic centroid detection)
@app.get("/deep-space/tess/stars/{filename}")
def tess_stars(filename: str, threshold: float = 5.0):
    filepath = TESS_DIR / filename
    try:
        arr = load_image_hdu(filepath)

        mean, std = np.mean(arr), np.std(arr)
        stars = np.argwhere(arr > mean + threshold * std)

        star_list = [{"x": int(x), "y": int(y)} for y, x in stars[:200]]  # limit 200
        return {"filename": filename, "stars_detected": len(star_list), "positions": star_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TESS stars error: {str(e)}")


# 7️⃣ Endpoint: Lookup TIC catalog info for a star (placeholder demo)
@app.get("/deep-space/tess/tic/{star_id}")
def tess_tic_lookup(star_id: str):
    try:
        return {
            "TIC_ID": star_id,
            "RA": "19:23:34.56",
            "Dec": "-45:12:34.1",
            "Magnitude": 11.4,
            "Exoplanet_candidate": True,
            "Note": "Replace with real MAST/TIC query for live star data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TESS TIC lookup error: {str(e)}")

# ===============================
# 🌌 Hubble Endpoints
# ===============================
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse

HUBBLE_DIR = Path("/mnt/c/NASA_Project/Deepspace_datasets/Hubble/MAST_2025/HLSP")
HUBBLE_PREVIEW_DIR = Path("previews/hubble")
HUBBLE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# 1️⃣ List available Hubble FITS files
@app.get("/deep-space/hubble/files")
def hubble_files():
    try:
        files = [f.name for f in HUBBLE_DIR.glob("*.fits")]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubble files error: {str(e)}")


# 2️⃣ Preview of a Hubble FITS image
@app.get("/deep-space/hubble/preview/{filename}")
def hubble_preview(filename: str):
    filepath = HUBBLE_DIR / filename
    try:
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")

        with fits.open(filepath) as hdul:
            data = hdul[1].data

        arr = np.nan_to_num(data)
        low, high = np.percentile(arr, (2, 98))
        arr = np.clip(arr, low, high)
        arr = (arr - arr.min()) / (arr.max() - arr.min())

        preview_path = HUBBLE_PREVIEW_DIR / f"{filename}_preview.png"
        plt.imsave(preview_path, arr, cmap="inferno")

        return FileResponse(preview_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubble preview error: {str(e)}")


# 3️⃣ Zoom / Crop into a galaxy or nebula
@app.get("/deep-space/hubble/crop/{filename}")
def hubble_crop(filename: str, x: int, y: int, size: int = 300):
    filepath = HUBBLE_DIR / filename
    try:
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")

        with fits.open(filepath) as hdul:
            data = hdul[1].data

        arr = np.nan_to_num(data)
        cropped = arr[y:y+size, x:x+size]
        cropped = (cropped - cropped.min()) / (cropped.max() - cropped.min())

        crop_path = HUBBLE_PREVIEW_DIR / f"{filename}crop{x}_{y}.png"
        plt.imsave(crop_path, cropped, cmap="inferno")

        return FileResponse(crop_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubble crop error: {str(e)}")


# 4️⃣ Compare Hubble vs JWST (placeholder demo)
@app.get("/deep-space/hubble/compare/{filename}")
def hubble_compare(filename: str):
    try:
        return JSONResponse({
            "hubble": f"/deep-space/hubble/preview/{filename}",
            "jwst": "/deep-space/jwst/sample.png",
            "note": "Overlay comparison between Hubble (visible) and JWST (infrared)"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubble compare error: {str(e)}")

# 5️⃣ Endpoint: Detect stars in Hubble image
@app.get("/deep-space/hubble/stars/{filename}")
def hubble_stars(filename: str, threshold: float = 5.0):
    filepath = HUBBLE_DIR / filename
    try:
        with fits.open(filepath) as hdul:
            data = np.nan_to_num(hdul[1].data)

        mean, std = np.mean(data), np.std(data)
        stars = np.argwhere(data > mean + threshold * std)

        star_list = [{"x": int(x), "y": int(y)} for y, x in stars[:200]]
        return {"filename": filename, "stars_detected": len(star_list), "positions": star_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubble stars error: {str(e)}")


# 6️⃣ Endpoint: JWST comparison (direct reference placeholder)
@app.get("/deep-space/hubble/jwst/{filename}")
def hubble_jwst(filename: str):
    try:
        return {
            "filename": filename,
            "hubble_preview": f"/deep-space/hubble/preview/{filename}",
            "jwst_preview": f"/deep-space/jwst/preview/{filename}",
            "note": "Compare JWST and Hubble observations of same object"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubble JWST error: {str(e)}")

# 7️⃣ Endpoint: Basic metadata/info about file
@app.get("/deep-space/hubble/info/{filename}")
def hubble_info(filename: str):
    filepath = HUBBLE_DIR / filename
    try:
        with fits.open(filepath) as hdul:
            header = hdul[0].header
        return {
            "filename": filename,
            "telescope": "Hubble Space Telescope",
            "object": header.get("OBJECT", "Unknown"),
            "date_obs": header.get("DATE-OBS", "Unknown"),
            "exposure": header.get("EXPTIME", "Unknown"),
            "instrument": header.get("INSTRUME", "Unknown"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubble info error: {str(e)}")

# =========================
# JWST CONFIG
# =========================
JWST_DIR = Path("/mnt/c/NASA_Project/DeepSpace_datasets/JWST/MAST_2025-0/HLSP/")
JWST_PREVIEW_DIR = Path("previews/jwst")
JWST_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# 1️⃣ List JWST FITS files
@app.get("/deep-space/jwst/files")
def jwst_files():
    files = [f.name for f in JWST_DIR.glob("*.fits")]
    return {"files": files}


# 2️⃣ Preview JWST FITS file
@app.get("/deep-space/jwst/preview/{filename}")
def jwst_preview(filename: str):
    filepath = JWST_DIR / filename
    try:
        with fits.open(filepath) as hdul:
            data = np.nan_to_num(hdul[1].data.astype(float))  # use extension 1 for images

        # Normalize and scale
        low, high = np.percentile(data, (2, 98))
        norm = np.clip(data, low, high)
        norm = (norm - norm.min()) / (norm.max() - norm.min())

        preview_path = JWST_PREVIEW_DIR / f"{filename}_preview.png"
        plt.imsave(preview_path, norm, cmap="inferno")
        return FileResponse(preview_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST preview error: {str(e)}")


# 3️⃣ Crop region from JWST image
@app.get("/deep-space/jwst/crop/{filename}")
def jwst_crop(filename: str, x: int, y: int, size: int = 200):
    filepath = JWST_DIR / filename
    try:
        with fits.open(filepath) as hdul:
            data = np.nan_to_num(hdul[1].data.astype(float))

        cropped = data[y:y+size, x:x+size]
        cropped = (cropped - cropped.min()) / (cropped.max() - cropped.min())

        crop_path = JWST_PREVIEW_DIR / f"{filename}crop{x}_{y}.png"
        plt.imsave(crop_path, cropped, cmap="inferno")
        return FileResponse(crop_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST crop error: {str(e)}")


# 4️⃣ Compare JWST vs Hubble (placeholder)
@app.get("/deep-space/jwst/compare/{filename}")
def jwst_compare(filename: str):
    try:
        return JSONResponse({
            "jwst": f"/deep-space/jwst/preview/{filename}",
            "hubble": f"/deep-space/hubble/preview/{filename}",
            "note": "Overlay comparison JWST (infrared) vs Hubble (visible light)"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST compare error: {str(e)}")


# 5️⃣ Detect bright objects (stars/galaxies)
@app.get("/deep-space/jwst/stars/{filename}")
def jwst_stars(filename: str, threshold: float = 5.0):
    filepath = JWST_DIR / filename
    try:
        with fits.open(filepath) as hdul:
            data = np.nan_to_num(hdul[1].data.astype(float))

        mean, std = np.mean(data), np.std(data)
        stars = np.argwhere(data > mean + threshold * std)

        star_list = [{"x": int(x), "y": int(y)} for y, x in stars[:200]]
        return {"filename": filename, "stars_detected": len(star_list), "positions": star_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST stars error: {str(e)}")


# 6️⃣ Gallery Mode (list curated images)
@app.get("/deep-space/jwst/gallery")
def jwst_gallery():
    files = [f.name for f in JWST_DIR.glob("*.fits")][:20]
    return {"gallery": files, "note": "Swipe between nebulae, galaxies, exoplanets"}


# 7️⃣ Metadata info for JWST FITS file
@app.get("/deep-space/jwst/info/{filename}")
def jwst_info(filename: str):
    filepath = JWST_DIR / filename
    try:
        with fits.open(filepath) as hdul:
            header = hdul[0].header

        return {
            "filename": filename,
            "object": header.get("OBJECT", "Unknown"),
            "date_obs": header.get("DATE-OBS", "Unknown"),
            "instrument": header.get("INSTRUME", "JWST"),
            "filter": header.get("FILTER", "Unknown"),
            "exptime": header.get("EXPTIME", "Unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST info error: {str(e)}")


# 8️⃣ Fun Fact / Did You Know
@app.get("/deep-space/jwst/facts/{filename}")
def jwst_facts(filename: str):
    try:
        return {
            "filename": filename,
            "fact": "JWST infrared reveals stars hidden behind cosmic dust. Example: Pillars of Creation looks totally different compared to Hubble."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST facts error: {str(e)}")


# 9️⃣ Extract objects from JWST image (bright spot detection)
@app.get("/deep-space/jwst/objects/{filename}")
def jwst_objects(filename: str):
    filepath = JWST_DIR / filename
    try:
        with fits.open(filepath) as hdul:
            data = np.nan_to_num(hdul[1].data.astype(float))

        bright_spots = np.argwhere(data > np.percentile(data, 99.5))
        objects = [{"x": int(x), "y": int(y)} for y, x in bright_spots[:100]]
        return {"filename": filename, "objects_detected": len(objects), "positions": objects}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST objects error: {str(e)}")


# 🔟 Lookup by object ID (placeholder for real catalog query)
@app.get("/deep-space/jwst/lookup/{object_id}")
def jwst_lookup(object_id: str):
    try:
        return {
            "object_id": object_id,
            "type": "Galaxy",
            "RA": "12:34:56.78",
            "Dec": "-12:34:56.7",
            "infrared_features": "Dust lanes, hidden star clusters",
            "note": "Replace with real MAST JWST object lookup"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JWST lookup error: {str(e)}")

# osiris_main.py  — Deep Space / OSIRIS-REx endpoints
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
from typing import Optional, List
import io
import os
from PIL import Image

# ---- IMPORTANT for huge TIFFs ----
Image.MAX_IMAGE_PIXELS = None  # allow very large TIFFs (avoid PIL DecompressionBombError)

# --------- Paths ---------
OSIRIS_DIR = Path("/mnt/c/NASA_Project/Deepspace_datasets/OSIRIS")
PREVIEW_DIR = OSIRIS_DIR / "_previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = OSIRIS_DIR / "models"  # optional: put .glb/.obj here if you add any
FEATURES_CATALOG = None            # optional: path to CSV/JSON catalog in future


# --------- Helpers ---------
def _sanitize_name(name: str) -> str:
    """Remove any path components to avoid traversal."""
    return Path(name).name

def _find_image_path(filename: str) -> Path:
    """Resolve a requested filename to an actual file inside OSIRIS_DIR (recursive).
       Accepts names with or without extension; tries .tif / .tiff if needed."""
    wanted = _sanitize_name(filename)
    # If a direct file exists (with whatever extension), prefer it.
    direct = OSIRIS_DIR / wanted
    if direct.exists() and direct.is_file():
        return direct

    stem = Path(wanted).stem  # strip any provided extension
    # Search recursively for matching stems with tif/tiff
    candidates: List[Path] = []
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidates += list(OSIRIS_DIR.rglob(stem + ext))

    if not candidates:
        # As a fallback, try to find any file that startswith this stem (handles long/suffixed names)
        candidates = [p for p in OSIRIS_DIR.rglob(".tif") if _sanitize_name(p.name).startswith(stem)]

    if not candidates:
        raise HTTPException(status_code=404, detail=f"OSIRIS file not found: {filename}")

    # Pick the shortest path / first match (usually the intended file)
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]

def _safe_open_image(path: Path) -> Image.Image:
    """Open a TIFF safely with PIL."""
    try:
        img = Image.open(path)
        # Defer actual decoding until needed; PIL does lazy load.
        img.load()  # ensure it's readable; raises if corrupt
        return img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open image: {e}")

def _save_png_bytes(img: Image.Image) -> bytes:
    """Return PNG bytes from a PIL image."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# --------- 1) List files ---------
@app.get("/deep-space/osiris-rex/files", summary="List Osiris files")
def list_osiris_files():
    if not OSIRIS_DIR.exists():
        raise HTTPException(status_code=404, detail=f"OSIRIS directory not found: {OSIRIS_DIR}")

    files = [
        str(p.relative_to(OSIRIS_DIR)).replace("\\", "/")
        for p in OSIRIS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    ]
    files.sort()
    return {"files": files}


# --------- 2) Preview (downscaled PNG) ---------
@app.get("/deep-space/osiris-rex/preview/{filename}", response_class=StreamingResponse, summary="Preview downscaled PNG")
def osiris_preview(
    filename: str,
    max_size: int = Query(1024, ge=256, le=4096, description="Max preview dimension (px)")
):
    path = _find_image_path(filename)
    img = _safe_open_image(path)

    # Downscale while preserving aspect ratio
    img = img.convert("L") if img.mode not in ("L", "RGB", "RGBA") else img
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    png = _save_png_bytes(img)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


# --------- 3) Crop (x,y,width,height) -> PNG ---------
@app.get("/deep-space/osiris-rex/crop/{filename}", response_class=StreamingResponse, summary="Crop region as PNG")
def osiris_crop(
    filename: str,
    x: int = Query(..., ge=0),
    y: int = Query(..., ge=0),
    width: int = Query(512, gt=0),
    height: int = Query(512, gt=0),
    preview_max: int = Query(1024, ge=128, le=4096, description="Optional downscale of cropped area")
):
    path = _find_image_path(filename)
    img = _safe_open_image(path)

    W, H = img.size
    x2, y2 = min(W, x + width), min(H, y + height)
    if x >= W or y >= H or x >= x2 or y >= y2:
        raise HTTPException(status_code=400, detail="Crop box outside image bounds")

    # Crop region
    region = img.crop((x, y, x2, y2))

    # Convert multi-band → grayscale (scientific data usually floats)
    region = region.convert("L")

    # Stretch contrast for visibility
    from PIL import ImageOps
    region = ImageOps.autocontrast(region)

    # Downscale for safety
    region.thumbnail((preview_max, preview_max), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    region.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")



# --------- 4) Basic info (safe metadata) ---------
@app.get("/deep-space/osiris-rex/info/{filename}", summary="Basic image info (no heavy metadata)")
def osiris_info(filename: str):
    path = _find_image_path(filename)
    img = _safe_open_image(path)

    try:
        stat = path.stat()
    except Exception:
        stat = None

    info = {
        "filename": str(path.name),
        "relative_path": str(path.relative_to(OSIRIS_DIR)).replace("\\", "/"),
        "format": img.format,
        "mode": img.mode,
        "size_px": {"width": img.size[0], "height": img.size[1]},
        "filesize_bytes": stat.st_size if stat else None,
    }
    return info



# --------- 5) Stream original object (TIFF) ---------
@app.get("/deep-space/osiris-rex/objects/{filename}", summary="Download/stream original TIFF")
def osiris_object(filename: str):
    path = _find_image_path(filename)
    # Stream raw file back (let frontend decide how to handle)
    return FileResponse(path, media_type="image/tiff", filename=path.name)


# --------- 6) 3D model fetch (placeholder) ---------
@app.get("/deep-space/osiris-rex/model/{model_id}", summary="Fetch 3D model (OBJ/GLB) if available")
def osiris_model(model_id: str):
    """
    Looks for OBJ/GLB in OSIRIS/models. You currently have TIFF rasters, not mesh models.
    This endpoint will return 404 unless you add a real model file.
    """
    sanitized = _sanitize_name(model_id)
    # Try exact, then try by stem for .obj/.glb
    direct = MODELS_DIR / sanitized
    if direct.exists():
        mt = "model/gltf-binary" if direct.suffix.lower() == ".glb" else "text/plain"
        return FileResponse(direct, media_type=mt, filename=direct.name)

    stem = Path(sanitized).stem
    for ext in (".glb", ".obj"):
        candidate = MODELS_DIR / f"{stem}{ext}"
        if candidate.exists():
            mt = "model/gltf-binary" if ext == ".glb" else "text/plain"
            return FileResponse(candidate, media_type=mt, filename=candidate.name)

    # Graceful message instead of 500
    raise HTTPException(status_code=404, detail="3D model not found (add .obj/.glb to OSIRIS/models)")


# --------- 7) Lookup object (placeholder unless catalog added) ---------
@app.get("/deep-space/osiris-rex/lookup/{object_id}", summary="Lookup Bennu surface feature (placeholder)")
def osiris_lookup(object_id: str):
    """
    If you add a CSV/JSON catalog of Bennu features, load it and return real metadata here.
    For now, we return a friendly placeholder.
    """
    data = {
        "object_id": object_id,
        "status": "ok",
        "notes": "No feature catalog loaded. Add a CSV/JSON and wire it here for real lookups."
    }
    return JSONResponse(data)

from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse, FileResponse
from fastapi.responses import StreamingResponse
import numpy as np
from io import BytesIO
import seaborn as sns   # add this at the top with other imports

# Path for ACE datasets
ACE_DIR = Path("/mnt/c/NASA_Project/Deepspace_datasets/ACE")

# Utility: load dataset
import numpy as np

def load_dataset(filename: str) -> pd.DataFrame:
    file_path = ACE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found in ACE datasets")

    if filename.endswith(".csv"):
        # Force no header, treat -999 as missing
        df = pd.read_csv(file_path, header=None, na_values=[-999])
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file_path, na_values=[-999])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format (only CSV/XLSX allowed)")

    return df

# 1. List all files in ACE directory
@app.get("/deep-space/ace/files")
async def list_files():
    files = [f.name for f in ACE_DIR.glob("*") if f.is_file()]
    return {"available_files": files}

# 2. Histogram of a dataset
@app.get("/deep-space/ace/histogram/{filename}")
def histogram_file(filename: str):
    filepath = ACE_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Load CSV (no header)
        df = pd.read_csv(filepath, header=None)
        df = df.replace(-999, np.nan)

        # Pick first numeric column
        col = df.columns[0]

        # Plot histogram
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=30)
        plt.title(f"Histogram of {filename}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # Stream as PNG
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Histogram failed: {str(e)}")

# 3. Summary statistics
@app.get("/deep-space/ace/stats/{filename}")
async def stats_file(filename: str):
    df = load_dataset(filename)
    stats = df.describe(include="all").to_dict()
    return {"file": filename, "stats": stats}

# 4. Filter data by time range
@app.get("/deep-space/ace/filter/{filename}")
def filter_file(filename: str, start: int = 0, end: int = 100):
    filepath = ACE_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(filepath, header=None)

        # if no header, name the column after filename
        if df.shape[1] == 1:
            colname = filename.replace(".csv", "")
            df.columns = [colname]
            df.index.name = "index"  # pseudo timestamp

        # filter by index range (not timestamp anymore)
        filtered = df.iloc[start:end]
        return filtered.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Plot variable over time
@app.get("/deep-space/ace/plot/{filename}/{variable}")
def plot_variable(filename: str, variable: str):
    filepath = ACE_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(filepath, header=None)

        if df.shape[1] == 1:
            df.columns = [filename.replace(".csv", "")]
            df.index.name = "index"

        if variable not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {variable} not found in file")

        plt.figure()
        df[variable].plot()
        plt.title(f"{variable} over index")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Correlation matrix
@app.get("/deep-space/ace/correlation/{file1}/{var1}/{file2}/{var2}")
def correlation_plot(file1: str, var1: str, file2: str, var2: str):
    path1 = ACE_DIR / file1
    path2 = ACE_DIR / file2
    if not path1.exists() or not path2.exists():
        raise HTTPException(status_code=404, detail="One or both files not found")

    try:
        df1 = pd.read_csv(path1, header=None)
        df2 = pd.read_csv(path2, header=None)

        # Auto assign headers if missing
        if df1.shape[1] == 1:
            df1.columns = [file1.replace(".csv", "")]
        if df2.shape[1] == 1:
            df2.columns = [file2.replace(".csv", "")]

        # Replace -999 with NaN
        df1.replace(-999, pd.NA, inplace=True)
        df2.replace(-999, pd.NA, inplace=True)

        if var1 not in df1.columns or var2 not in df2.columns:
            raise HTTPException(status_code=400, detail=f"Column {var1} or {var2} not found")

        series1 = pd.to_numeric(df1[var1], errors="coerce")
        series2 = pd.to_numeric(df2[var2], errors="coerce")

        merged = pd.DataFrame({var1: series1, var2: series2}).dropna()

        if merged.shape[0] < 2:
            raise HTTPException(status_code=400, detail="Not enough data for correlation")

        # Plot correlation
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=var1, y=var2, data=merged)
        plt.title(f"Correlation: {var1} vs {var2}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. Anomaly detection (z-score method)
@app.get("/deep-space/ace/anomalies/{filename}/{variable}")
def detect_anomalies(filename: str, variable: str):
    path = ACE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(path, header=None)

        if df.shape[1] == 1:
            df.columns = [filename.replace(".csv", "")]

        df.replace(-999, pd.NA, inplace=True)

        if variable not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {variable} not found")

        series = pd.to_numeric(df[variable], errors="coerce").dropna()

        if series.empty:
            raise HTTPException(status_code=400, detail=f"No valid data for {variable}")

        # Anomaly detection with z-score
        mean = series.mean()
        std = series.std()
        anomaly_mask = ((series - mean).abs() > 3 * std)

        # Plot anomalies
        plt.figure(figsize=(10, 5))
        plt.plot(series.index, series.values, label="Data")
        plt.scatter(series.index[anomaly_mask], series[anomaly_mask], color="red", label="Anomalies")
        plt.legend()
        plt.title(f"Anomalies in {variable} ({filename})")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 8. Download filtered dataset
@app.get("/deep-space/ace/download/{filename}")
async def download_file(filename: str, start: str = None, end: str = None):
    df = load_dataset(filename)
    if "timestamp" in df.columns and start and end:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    out_path = ACE_DIR / f"filtered_{filename}"
    df.to_csv(out_path, index=False)
    return FileResponse(out_path, filename=f"filtered_{filename}")
