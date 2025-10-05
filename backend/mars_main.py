# mars_main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from pvl import load as pvl_load
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep a single app instance
app.title = "Mars Backend API"
app.version = "1.0.0"

# Enable CORS so Swagger UI and browser requests work properly
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (you can restrict later if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Shared paths (repo-relative; no hardcoded OS paths)
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = str(REPO_ROOT / "Mars_datasets")
CTX_DIR  = os.path.join(BASE_DIR, "CTX")
PREVIEW_DIR = os.path.join(CTX_DIR, "_previews")
THUMB_DIR   = os.path.join(CTX_DIR, "_thumbnails")
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def deep_get(pvl_obj, key):
    """Recursively search for a key in a PVL structure (dict/list)."""
    if isinstance(pvl_obj, dict):
        if key in pvl_obj:
            return pvl_obj[key]
        for v in pvl_obj.values():
            out = deep_get(v, key)
            if out is not None:
                return out
    elif isinstance(pvl_obj, list):
        for v in pvl_obj:
            out = deep_get(v, key)
            if out is not None:
                return out
    return None

def safe_get(lbl, key):
    """Check OBJECT=IMAGE first, else fallback to deep_get."""
    try:
        if "IMAGE" in lbl and isinstance(lbl["IMAGE"], dict) and key in lbl["IMAGE"]:
            return lbl["IMAGE"][key]
    except Exception:
        pass
    return deep_get(lbl, key)

def find_ctx_label_for(img_path: str) -> str | None:  
    """Find the matching *.txt label for a CTX *.IMG file."""  
    base = os.path.basename(img_path)  
    dirname = os.path.dirname(img_path)  
    stem = os.path.splitext(base)[0]  
    candidates = [  
        base + ".txt",  
        base.replace(".IMG", ".IMG.txt"),  
        base.replace(".IMG", ".txt"),  
        stem + ".txt",  
        stem + ".TXT",  
    ]  
    for c in candidates:  
        p = os.path.join(dirname, c)  
        if os.path.exists(p):  
            return p  
    return None  
  
def np_dtype_from_label(sample_bits: int, sample_type: str) -> np.dtype:  
    st = (sample_type or "").upper()  
    if sample_bits == 8:  
        return np.uint8 if "UNSIGNED" in st else np.int8  
    if sample_bits == 16:  
        big = ("MSB" in st) or ("BIG" in st)  
        is_unsigned = "UNSIGNED" in st  
        if is_unsigned:  
            return np.dtype(">u2") if big else np.dtype("<u2")  
        else:  
            return np.dtype(">i2") if big else np.dtype("<i2")  
    # Fallback  
    return np.uint8  
  
def read_ctx_image(img_path: str) -> np.ndarray:  
    """Decode a CTX IMG using its PVL label (handles prefix/suffix bytes)."""  
    lbl_path = find_ctx_label_for(img_path)  
    if not lbl_path:  
        raise RuntimeError("Matching .txt label not found for IMG.")  
  
    lbl = pvl_load(open(lbl_path))  
  
    lines            = int(safe_get(lbl, "LINES"))  
    line_samples     = int(safe_get(lbl, "LINE_SAMPLES"))  
    sample_bits      = int(safe_get(lbl, "SAMPLE_BITS"))  
    sample_type      = str(safe_get(lbl, "SAMPLE_TYPE") or "UNSIGNED_INTEGER")  
    line_prefix_bytes = int(safe_get(lbl, "LINE_PREFIX_BYTES") or 0)  
    line_suffix_bytes = int(safe_get(lbl, "LINE_SUFFIX_BYTES") or 0)  
  
    sample_bytes = sample_bits // 8  
    row_bytes = line_prefix_bytes + line_samples * sample_bytes + line_suffix_bytes  
  
    with open(img_path, "rb") as f:  
        raw = np.frombuffer(f.read(), dtype=np.uint8)  
  
    expected = lines * row_bytes  
    if raw.size < expected:  
        raise RuntimeError("File smaller than expected, check label values.")  
  
    rows_u8 = raw[:expected].reshape((lines, row_bytes))  
    pix_u8 = rows_u8[:, line_prefix_bytes : line_prefix_bytes + line_samples * sample_bytes]  
  
    if sample_bytes == 1:  
        img = pix_u8  
    else:  
        dt = np_dtype_from_label(sample_bits, sample_type)  
        img = pix_u8.reshape(lines, line_samples, sample_bytes).view(dt).reshape(lines, line_samples)  
    return img  
  
def to_8bit_preview(img: np.ndarray) -> np.ndarray:  
    """Stretch contrast to 8-bit (2–98 percentile)."""  
    arr = img.astype(np.float32)  
    finite = arr[np.isfinite(arr)]  
    if finite.size == 0:  
        return np.zeros_like(arr, dtype=np.uint8)  
    p2, p98 = np.percentile(finite, [2, 98])  
    if p98 <= p2:  
        p2, p98 = float(finite.min()), float(finite.max())  
    arr = np.clip((arr - p2) / max(p98 - p2, 1e-6), 0, 1)  
    return (arr * 255).astype(np.uint8)  
  
def ensure_ctx_file(filename: str) -> str:  
    """Return absolute path inside CTX_DIR or 404."""  
    path = os.path.join(CTX_DIR, filename)  
    if not os.path.exists(path):  
        raise HTTPException(status_code=404, detail="File not found")  
    return path  
  
  
# ====================================================================  
# ORDERED ENDPOINTS (1 → 8) — EXACTLY AS YOU REQUESTED  
# ====================================================================  
  
# 1) Read Mars root  
@app.get("/mars")  
def read_mars_root():  
    return {  
        "datasets": [  
            {"id": "CTX", "desc": "Context Camera (6 m/px)", "preview": True},  
            {"id": "HiRISE", "desc": "High-resolution (0.25 m/px)", "preview": True},  
            {"id": "MOLA", "desc": "Elevation / 3D terrain", "preview": False},  
            {"id": "THEMIS", "desc": "Infrared + Visible", "preview": True},  
        ]  
    }  
  
# 2) List Ctx Files  
@app.get("/mars/ctx/files")  
def list_ctx_files():  
    if not os.path.exists(CTX_DIR):  
        return {"count": 0, "files": [], "message": "CTX directory not found"}  
    files = sorted(  
        [f for f in os.listdir(CTX_DIR) if f.lower().endswith((".img", ".txt"))]  
    )  
    return {"count": len(files), "files": files}  
  
# 3) Ctx File Stats  
@app.get("/mars/ctx/stats/{filename}")  
def ctx_file_stats(filename: str):  
    path = ensure_ctx_file(filename)  
  
    if filename.lower().endswith(".txt"):  
        try:  
            meta = pvl_load(open(path))  
            return {"filename": filename, "metadata": dict(meta)}  
        except Exception as e:  
            return {"filename": filename, "error": str(e)}  
  
    if filename.lower().endswith(".img"):  
        lbl = find_ctx_label_for(path)  
        if not lbl:  
            return {"filename": filename, "message": "No metadata file found"}  
        try:  
            meta = pvl_load(open(lbl))  
            return {"filename": filename, "label": os.path.basename(lbl), "metadata": dict(meta)}  
        except Exception as e:  
            return {"filename": filename, "label": os.path.basename(lbl), "error": str(e)}  
  
    return {"error": "Unsupported file type"}  
  
# 4) Ctx Preview  (returns a PNG image)  
@app.get("/mars/ctx/preview/{filename}")  
def ctx_preview(filename: str, width: int = Query(1024, ge=64, le=4096)):  
    img_path = ensure_ctx_file(filename)  
    if filename.lower().endswith(".txt"):  
        try:  
            meta = pvl_load(open(img_path))  
            return {"filename": filename, "metadata": dict(meta)}  
        except Exception as e:  
            return {"error": str(e)}  
  
    try:  
        raw_img = read_ctx_image(img_path)  
        preview = to_8bit_preview(raw_img)  
  
        # Resize to requested width, keep aspect  
        h, w = preview.shape  
        new_h = int(round(h * (width / w)))  
        im = Image.fromarray(preview, mode="L").resize((width, new_h), Image.LANCZOS)  
  
        out_path = os.path.join(PREVIEW_DIR, f"{filename}.png")  
        im.save(out_path)  
        return FileResponse(out_path, media_type="image/png")  
    except Exception as e:  
        return JSONResponse({"error": f"Could not generate preview: {e}"}, status_code=500)  
  
# 5) Ctx Full image (download the IMG as-is)  
@app.get("/mars/ctx/image/{filename}")  
def ctx_full_image(filename: str):  
    file_path = ensure_ctx_file(filename)  
    return FileResponse(file_path, media_type="application/octet-stream")  
  
# 6) Ctx Thumbnail (returns a small JPEG)  
@app.get("/mars/ctx/thumb/{filename}")
def ctx_thumbnail(
    filename: str,
    size: int = Query(256, ge=64, le=2048),
):
    img_path = ensure_ctx_file(filename)

    thumb_path = os.path.join(THUMB_DIR, f"{filename}_{size}.jpg")
    if not os.path.exists(thumb_path):
        # Build from decoded preview (correct geometry)
        raw_img = read_ctx_image(img_path)
        small = to_8bit_preview(raw_img)
        im = Image.fromarray(small, mode="L")
        im = im.resize((size, size))  # square thumbnail for UI grids
        im.save(thumb_path, quality=92)

    return FileResponse(thumb_path, media_type="image/jpeg")

# 7) Ctx Search (by year range, using the label’s START_TIME)  
@app.get("/mars/ctx/search")  
def ctx_search(
    min_year: int = Query(None),  
    max_year: int = Query(None),  
    instrument: str = Query("CTX")  
):
    results = []  
    for name in sorted(os.listdir(CTX_DIR)):  
        if not name.lower().endswith(".img"):  
            continue  
        fpath = os.path.join(CTX_DIR, name)  
        try:  
            lbl_path = find_ctx_label_for(fpath)  
            if not lbl_path:  
                continue  
            raw = pvl_load(open(lbl_path))  
            start = str(safe_get(raw, "START_TIME") or "")  
            year = int(start[:4]) if len(start) >= 4 and start[:4].isdigit() else None  
  
            if min_year is not None and (year is None or year < min_year):  
                continue  
            if max_year is not None and (year is None or year > max_year):  
                continue  
  
            results.append(name)  
        except Exception:  
            continue  
  
    return {"results": results, "count": len(results), "instrument": instrument}  
  
# 8) Ctx index (small summary for CTX only)  
@app.get("/mars/ctx")  
def ctx_index():  
    if not os.path.exists(CTX_DIR):  
        raise HTTPException(status_code=404, detail="CTX directory not found")  
    imgs = [f for f in os.listdir(CTX_DIR) if f.lower().endswith(".img")]  
    txts = [f for f in os.listdir(CTX_DIR) if f.lower().endswith(".txt")]  
    return {  
        "dataset": "CTX",  
        "path": CTX_DIR,  
        "files": {"img": len(imgs), "txt": len(txts)},  
        "endpoints": [
            "/mars",
            "/mars/ctx/files",
            "/mars/ctx/stats/{filename}",
            "/mars/ctx/preview/{filename}",
            "/mars/ctx/image/{filename}",
            "/mars/ctx/thumb/{filename}",
            "/mars/ctx/search",
            "/mars/ctx",
        ],
    }

# ========== HiRISE endpoints (paste below your CTX endpoints; do not re-create app) ==========
import os, io, re
from urllib.parse import unquote
import numpy as np
from PIL import Image
import glymur
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse, JSONResponse

# ----- CONFIG -----
HIRISE_DIR      = str(REPO_ROOT / "Mars_datasets" / "HiRISE")
HIRISE_PREV_DIR = os.path.join(HIRISE_DIR, "previews")
HIRISE_THUMB_DIR= os.path.join(HIRISE_DIR, "thumbnails")

# Optional hosted fallback (your OneDrive preview)
HIRISE_FALLBACK_PREVIEW = (
    "https://1drv.ms/i/c/7ec4b887578df863/Ec2BPEhA2LZFk8gbEFDsxC0BIQo38qQi2JOvnrC1ItmNuQ?e=T6NWiE"
)

os.makedirs(HIRISE_DIR, exist_ok=True)
os.makedirs(HIRISE_PREV_DIR, exist_ok=True)
os.makedirs(HIRISE_THUMB_DIR, exist_ok=True)


# ----- HELPERS -----
def _ensure_hirise_file(name: str) -> str:
    """Return absolute path for a file inside HIRISE_DIR; 404 if missing."""
    path = os.path.join(HIRISE_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return path

def _safe_name(name: str) -> str:
    """Filesystem-safe cache name (keeps extension)."""
    base, ext = os.path.splitext(name)
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base + ext

def _stretch_8bit(arr: np.ndarray) -> np.ndarray:
    """
    Percentile stretch to 8-bit. Handles 2D (grayscale) or 3-band arrays.
    Uses 2–98 percentile for contrast; robust to NaNs/Infs.
    """
    if arr.ndim == 2:
        a = arr.astype(np.float32)
        finite = np.isfinite(a)
        if not np.any(finite):
            return np.zeros_like(a, dtype=np.uint8)
        p2, p98 = np.percentile(a[finite], (2, 98))
        if p98 <= p2:
            p2, p98 = float(a.min()), float(a.max())
        a = np.clip((a - p2) / max(p98 - p2, 1e-6), 0, 1)
        return (a * 255).astype(np.uint8)

    # 3-band
    out = []
    for i in range(min(3, arr.shape[2])):
        ch = arr[..., i].astype(np.float32)
        finite = np.isfinite(ch)
        if not np.any(finite):
            out.append(np.zeros_like(ch, dtype=np.uint8))
            continue
        p2, p98 = np.percentile(ch[finite], (2, 98))
        if p98 <= p2:
            p2, p98 = float(ch.min()), float(ch.max())
        ch = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)
        out.append((ch * 255).astype(np.uint8))
    return np.dstack(out)


def _read_jp2_reduced(jp2: glymur.Jp2k, target: int) -> np.ndarray:
    """
    Read a reduced resolution (or center window) to roughly match target pixels on the long side.
    This avoids loading full ~GB images.
    """
    # Image shape: (rows, cols) or (rows, cols, bands)
    shape = jp2.shape
    h, w = (shape[0], shape[1])
    # Compute reduction level ~ log2(scale)
    long_side = max(h, w)
    scale = max(long_side / max(target, 1), 1.0)
    rlevel = int(np.ceil(np.log2(scale)))
    rlevel = max(0, min(rlevel, 6))  # clamp to a reasonable range

    try:
        data = jp2.read(rlevel=rlevel)
        return data
    except Exception:
        # Fallback: center crop window
        half = target // 2
        cy, cx = h // 2, w // 2
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        return jp2.read(window=((y0, y1), (x0, x1)))


# =================================
# 1) List HiRISE files
# =================================
@app.get("/mars/hirise/list")
def hirise_list():
    if not os.path.exists(HIRISE_DIR):
        raise HTTPException(status_code=404, detail="HiRISE directory not found")
    files = [
        f for f in os.listdir(HIRISE_DIR)
        if f.lower().endswith((".jp2", ".lbl", ".txt"))
    ]
    files.sort()
    return {"files": files}


# =================================
# 2) HiRISE Preview  (image/*)
# =================================
@app.get("/preview/{filename}")
def hirise_preview(filename: str, size: int = 2000):
    link = HIRISE_FALLBACK_PREVIEW
    return {
        "filename": filename,
        "size": size,
        "preview_link": link
    }

# =================================
# 4) HiRISE Stats  (JP2 and LBL/TXT)
# =================================
@app.get("/mars/hirise/stats/{filename}")
def hirise_stats(filename: str):
    """
    For JP2: shape/dtype/filesize.
    For LBL/TXT: parsed key/value pairs (tiny subset).
    """
    filename = unquote(filename)
    path = _ensure_hirise_file(filename)

    low = filename.lower()
    if low.endswith(".jp2"):
        try:
            jp2 = glymur.Jp2k(path)
            shape = tuple(int(x) for x in jp2.shape)
            # read a 1x1 sample to get dtype cheaply
            sample = jp2[0:1, 0:1]
            dtype = str(sample.dtype)
            size_bytes = os.path.getsize(path)
            return {"type": "jp2", "shape": shape, "dtype": dtype, "size_bytes": size_bytes}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read JP2 stats: {e}")

    elif low.endswith((".lbl", ".txt")):
        # very light parser → k = v
        meta = {}
        try:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        meta[k.strip()] = v.strip().strip('"')
            return {"type": "label", "keys": list(meta.keys())[:50], "meta_preview": {k: meta[k] for k in list(meta)[:20]}}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read label: {e}")

    else:
        raise HTTPException(status_code=400, detail="unsupported file type")


# =================================
# 5) HiRISE Metadata (LBL/TXT full text)
# =================================
@app.get("/mars/hirise/metadata/{filename}")
def hirise_metadata(filename: str):
    filename = unquote(filename)
    path = _ensure_hirise_file(filename)
    if not filename.lower().endswith((".lbl", ".txt")):
        raise HTTPException(status_code=400, detail="metadata endpoint only supports .LBL or .TXT files")
    try:
        with open(path, "r", errors="ignore") as f:
            return JSONResponse({"filename": filename, "content": f.read()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read metadata: {e}")
# ========== END HiRISE block ==========

# =====================  MOLA (Mars Orbiter Laser Altimeter)  =====================
# Put this block in mars_main.py after your CTX/HiRISE endpoints.

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
import os

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELS
# ============================================

class Point(BaseModel):
    latitude: float
    longitude: float

class ElevationResponse(BaseModel):
    latitude: float
    longitude: float
    elevation_m: float
    data_quality: str

class ProfilePoint(BaseModel):
    latitude: float
    longitude: float
    elevation_m: float
    distance_km: float

class ProfileResponse(BaseModel):
    start: Point
    end: Point
    total_distance_km: float
    profile: List[ProfilePoint]
    elevation_gain_m: float
    elevation_loss_m: float

class MOLAFileInfo(BaseModel):
    filename: str
    resolution_ppd: int  # pixels per degree
    coverage: str
    dimensions: dict
    elevation_range: dict

class StatsResponse(BaseModel):
    region: str
    min_elevation_m: float
    max_elevation_m: float
    mean_elevation_m: float
    std_elevation_m: float
    area_sq_km: Optional[float] = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def read_mola_img(file_path: str):
    """Read MOLA .img file as raw binary"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    total_pixels = len(raw_data) // 2
    
    # MOLA standard dimensions
    possible_dims = [
        (1440, 720),    # 4 ppd
        (2880, 1440),   # 8 ppd
        (5760, 2880),   # 16 ppd
        (11520, 5760),  # 32 ppd
    ]
    
    width, height = None, None
    for w, h in possible_dims:
        if w * h == total_pixels:
            width, height = w, h
            break
    
    if width is None:
        raise ValueError(f"Unknown MOLA file dimensions: {total_pixels} pixels")
    
    # Read as 16-bit signed int, big-endian
    data = np.frombuffer(raw_data, dtype='>i2')
    data = data.reshape((height, width))
    
    # Mark invalid data
    data = data.astype(float)
    data[data < -30000] = np.nan
    
    return data, width, height

def latlon_to_pixel(lat: float, lon: float, width: int, height: int):
    """Convert lat/lon to pixel coordinates"""
    # MOLA global grids: -90 to 90 lat, 0 to 360 lon
    lon = lon % 360  # Normalize to 0-360
    
    # Convert to pixel indices
    x = int((lon / 360.0) * width)
    y = int(((90 - lat) / 180.0) * height)
    
    # Clamp to valid range
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    
    return x, y

def get_elevation_at_point(data, lat: float, lon: float, width: int, height: int):
    """Get elevation at a specific lat/lon"""
    x, y = latlon_to_pixel(lat, lon, width, height)
    elevation = float(data[y, x])
    
    return elevation if not np.isnan(elevation) else None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Mars (km)"""
    # Mars radius in km
    R = 3389.5
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "service": "MOLA Elevation API",
        "description": "Mars topography data from MOLA",
        "endpoints": {
            "elevation": "/mars/mola/elevation",
            "profile": "/mars/mola/profile",
            "stats": "/mars/mola/stats",
            "files": "/mars/mola/files"
        }
    }

@app.get("/mars/mola/files")
def list_mola_files(
    include_metadata: bool = Query(True, description="Include .lbl metadata files")
):
    """List all available MOLA datasets and metadata files"""
    mola_dir = str(REPO_ROOT / "Mars_datasets" / "Mola") + "/"
    
    try:
        all_files = os.listdir(mola_dir)
        
        # Separate .img and .lbl files
        img_files = [f for f in all_files if f.endswith('.img')]
        lbl_files = [f for f in all_files if f.endswith('.lbl')]
        
        file_info = []
        
        # Process .img files
        for filename in sorted(img_files):
            # Parse filename to get info
            ppd = 4  # Default resolution
            if 'megr' in filename.lower():
                ppd = 128
            elif 'megt' in filename.lower():
                ppd = 4
            
            try:
                file_path = os.path.join(mola_dir, filename)
                data, w, h = read_mola_img(file_path)
                
                # Check if there's a matching .lbl file
                lbl_filename = filename.replace('.img', '.lbl')
                has_metadata = lbl_filename in lbl_files
                
                file_info.append({
                    "type": "data",
                    "filename": filename,
                    "resolution_ppd": ppd,
                    "coverage": "Global" if 'gb' in filename else "Regional",
                    "dimensions": {"width": w, "height": h},
                    "elevation_range": {
                        "min": float(np.nanmin(data)),
                        "max": float(np.nanmax(data))
                    },
                    "has_metadata": has_metadata,
                    "metadata_file": lbl_filename if has_metadata else None
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        
        # Add .lbl files if requested
        if include_metadata:
            for lbl_file in sorted(lbl_files):
                # Only include .lbl files that don't have a matching .img
                img_match = lbl_file.replace('.lbl', '.img')
                if img_match not in img_files:
                    file_info.append({
                        "type": "metadata",
                        "filename": lbl_file,
                        "description": "PDS label file (metadata only)"
                    })
        
        return {
            "directory": mola_dir,
            "total_files": len(file_info),
            "data_files": len(img_files),
            "metadata_files": len(lbl_files),
            "files": file_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mars/mola/elevation", response_model=ElevationResponse)
def get_elevation(
    lat: float = Query(..., ge=-90, le=90, description="Latitude (-90 to 90)"),
    lon: float = Query(..., ge=-180, le=360, description="Longitude (-180 to 360)"),
    filename: str = Query("megt00n000gb.img", description="MOLA file to query")
):
    """
    Get elevation at a specific latitude/longitude point
    
    Example: /mars/mola/elevation?lat=-4.5&lon=137.4 (Gale Crater)
    """
    file_path = str(REPO_ROOT / "Mars_datasets" / "Mola" / filename)
    
    try:
        data, width, height = read_mola_img(file_path)
        elevation = get_elevation_at_point(data, lat, lon, width, height)
        
        if elevation is None:
            raise HTTPException(status_code=404, detail="No elevation data at this location")
        
        return ElevationResponse(
            latitude=lat,
            longitude=lon,
            elevation_m=elevation,
            data_quality="valid" if abs(elevation) < 20000 else "questionable"
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mars/mola/profile", response_model=ProfileResponse)
def get_elevation_profile(
    start_lat: float = Query(..., ge=-90, le=90),
    start_lon: float = Query(..., ge=-180, le=360),
    end_lat: float = Query(..., ge=-90, le=90),
    end_lon: float = Query(..., ge=-180, le=360),
    num_points: int = Query(100, ge=10, le=1000, description="Number of sample points"),
    filename: str = Query("megt00n000gb.img")
):
    """
    Get elevation profile between two points
    
    Example: Cross-section through Valles Marineris
    /mars/mola/profile?start_lat=-14&start_lon=300&end_lat=-8&end_lon=310
    """
    file_path = str(REPO_ROOT / "Mars_datasets" / "Mola" / filename)
    
    try:
        data, width, height = read_mola_img(file_path)
        
        # Generate points along the line
        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)
        
        profile = []
        cumulative_distance = 0
        prev_lat, prev_lon = start_lat, start_lon
        
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            elevation = get_elevation_at_point(data, lat, lon, width, height)
            
            if i > 0:
                cumulative_distance += haversine_distance(prev_lat, prev_lon, lat, lon)
            
            if elevation is not None:
                profile.append(ProfilePoint(
                    latitude=lat,
                    longitude=lon,
                    elevation_m=elevation,
                    distance_km=cumulative_distance
                ))
            
            prev_lat, prev_lon = lat, lon
        
        # Calculate elevation changes
        elevations = [p.elevation_m for p in profile]
        elevation_diffs = np.diff(elevations)
        elevation_gain = float(np.sum(elevation_diffs[elevation_diffs > 0]))
        elevation_loss = float(abs(np.sum(elevation_diffs[elevation_diffs < 0])))
        
        return ProfileResponse(
            start=Point(latitude=start_lat, longitude=start_lon),
            end=Point(latitude=end_lat, longitude=end_lon),
            total_distance_km=cumulative_distance,
            profile=profile,
            elevation_gain_m=elevation_gain,
            elevation_loss_m=elevation_loss
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mars/mola/stats", response_model=StatsResponse)
def get_region_stats(
    min_lat: float = Query(..., ge=-90, le=90),
    max_lat: float = Query(..., ge=-90, le=90),
    min_lon: float = Query(..., ge=-180, le=360),
    max_lon: float = Query(..., ge=-180, le=360),
    filename: str = Query("megt00n000gb.img")
):
    """
    Get elevation statistics for a rectangular region
    
    Example: Olympus Mons area
    /mars/mola/stats?min_lat=15&max_lat=25&min_lon=220&max_lon=230
    """
    file_path = f"/mnt/c/NASA_PROJECT/Mars_datasets/Mola/{filename}"
    
    try:
        data, width, height = read_mola_img(file_path)
        
        # Ensure min < max
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon
        
        # Convert bounds to pixel coordinates
        x1, y1 = latlon_to_pixel(max_lat, min_lon, width, height)  # Top-left
        x2, y2 = latlon_to_pixel(min_lat, max_lon, width, height)  # Bottom-right
        
        # Ensure proper ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Extract region
        region = data[y1:y2, x1:x2]
        
        # Check if region has valid data
        valid_data = region[~np.isnan(region)]
        
        if len(valid_data) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No valid elevation data in region [{min_lat},{min_lon}] to [{max_lat},{max_lon}]"
            )
        
        # Calculate stats
        return StatsResponse(
            region=f"[{min_lat},{min_lon}] to [{max_lat},{max_lon}]",
            min_elevation_m=float(np.min(valid_data)),
            max_elevation_m=float(np.max(valid_data)),
            mean_elevation_m=float(np.mean(valid_data)),
            std_elevation_m=float(np.std(valid_data)),
            area_sq_km=None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# FAMOUS LOCATIONS HELPER
# ============================================

@app.get("/mars/mola/locations")
def famous_locations():
    """Get elevations at famous Mars locations"""
    locations = {
        "Olympus Mons (Summit)": {"lat": 18.65, "lon": 226.2, "description": "Tallest mountain in solar system"},
        "Gale Crater (Curiosity)": {"lat": -4.5, "lon": 137.4, "description": "Curiosity rover landing site"},
        "Valles Marineris (Deepest)": {"lat": -13.9, "lon": 301.6, "description": "Deepest point in canyon"},
        "Hellas Basin (Floor)": {"lat": -42.4, "lon": 70.5, "description": "Lowest point on Mars"},
        "North Pole": {"lat": 90, "lon": 0, "description": "Mars north polar ice cap"},
    }
    
    return locations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, Literal
from pydantic import BaseModel
import os
from PIL import Image
import numpy as np
import io

app = FastAPI(title="THEMIS API", description="Mars Thermal Emission Imaging System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

THEMIS_DIR = "/mnt/c/NASA_PROJECT/Mars_datasets/THEMIS/"

# ============================================
# MODELS
# ============================================

class BandInfo(BaseModel):
    band_number: int
    wavelength_um: float
    description: str
    typical_use: str

# ============================================
# BAND DEFINITIONS
# ============================================

THEMIS_IR_BANDS = [
    BandInfo(band_number=1, wavelength_um=6.78, description="Band 1", typical_use="Surface temperature"),
    BandInfo(band_number=2, wavelength_um=6.78, description="Band 2", typical_use="Surface temperature"),
    BandInfo(band_number=3, wavelength_um=7.93, description="Band 3", typical_use="Atmospheric correction"),
    BandInfo(band_number=4, wavelength_um=8.56, description="Band 4", typical_use="Surface properties"),
    BandInfo(band_number=5, wavelength_um=9.35, description="Band 5", typical_use="Surface properties"),
    BandInfo(band_number=6, wavelength_um=10.21, description="Band 6", typical_use="Silicate features"),
    BandInfo(band_number=7, wavelength_um=11.04, description="Band 7", typical_use="Silicate features"),
    BandInfo(band_number=8, wavelength_um=11.79, description="Band 8", typical_use="Silicate features"),
    BandInfo(band_number=9, wavelength_um=12.57, description="Band 9", typical_use="Carbonate features"),
    BandInfo(band_number=10, wavelength_um=14.88, description="Band 10", typical_use="Atmospheric dust")
]

THEMIS_VIS_BANDS = [
    BandInfo(band_number=1, wavelength_um=0.425, description="Blue", typical_use="Surface color"),
    BandInfo(band_number=2, wavelength_um=0.540, description="Green", typical_use="Surface color"),
    BandInfo(band_number=3, wavelength_um=0.654, description="Red", typical_use="Surface color"),
    BandInfo(band_number=4, wavelength_um=0.749, description="NIR", typical_use="Mineralogy"),
    BandInfo(band_number=5, wavelength_um=0.860, description="NIR", typical_use="Mineralogy")
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def parse_lbl_file(lbl_path: str) -> dict:
    """Parse PDS .LBL file"""
    metadata = {}
    try:
        with open(lbl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('/*') and not line.startswith('END'):
                    try:
                        key, value = line.split('=', 1)
                        metadata[key.strip()] = value.strip().strip('"').strip()
                    except:
                        continue
    except Exception as e:
        print(f"Error parsing LBL: {e}")
    return metadata

def read_isis_cube_gdal(cub_path: str, band: int = 1):
    """Read ISIS cube using GDAL"""
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        
        dataset = gdal.Open(cub_path)
        if dataset is None:
            raise Exception("Could not open cube file")
        
        band_obj = dataset.GetRasterBand(band)
        data = band_obj.ReadAsArray()
        
        return data, dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount
    
    except ImportError:
        raise HTTPException(status_code=501, detail="GDAL not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading cube: {str(e)}")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "service": "THEMIS API",
        "description": "Mars Thermal Emission Imaging System",
        "version": "1.0.0",
        "note": "Using pre-extracted .CUB files for fast access",
        "endpoints": {
            "files": "/mars/themis/files",
            "metadata": "/mars/themis/metadata/{filename}",
            "bands": "/mars/themis/bands?image_type=IR",
            "image": "/mars/themis/image/{filename}?band=1",
            "stats": "/mars/themis/stats/{filename}?band=1"
        }
    }

@app.get("/mars/themis/files")
def list_themis_files(image_type: Optional[Literal["VIS", "IR", "ALL"]] = Query("ALL")):
    """List all THEMIS files"""
    
    try:
        if not os.path.exists(THEMIS_DIR):
            raise HTTPException(status_code=404, detail=f"Directory not found: {THEMIS_DIR}")
        
        all_files = os.listdir(THEMIS_DIR)
        
        # Look for .CUB files (uncompressed)
        cub_files = [f for f in all_files if f.endswith('.CUB') and not f.endswith('.gz')]
        lbl_files = [f for f in all_files if f.endswith('.LBL')]
        
        file_list = []
        
        for filename in sorted(cub_files):
            base_name = filename[:-4]  # Remove .CUB
            
            # Determine type
            if base_name.upper().startswith('I'):
                img_type = "IR"
                bands = 10
            elif base_name.upper().startswith('V'):
                img_type = "VIS"
                bands = 5
            else:
                continue
            
            if image_type != "ALL" and img_type != image_type:
                continue
            
            # Find matching .LBL
            lbl_match = None
            for lbl in lbl_files:
                if lbl[:-4].upper() == base_name.upper():
                    lbl_match = lbl
                    break
            
            # Get file size
            file_path = os.path.join(THEMIS_DIR, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            
            file_list.append({
                "filename": filename,
                "product_id": base_name,
                "image_type": img_type,
                "bands": bands,
                "file_size_mb": round(file_size, 2),
                "has_metadata": lbl_match is not None,
                "metadata_file": lbl_match
            })
        
        return file_list
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mars/themis/metadata/{filename}")
def get_themis_metadata(filename: str):
    """Get THEMIS metadata from .LBL file"""
    
    # Handle both .CUB and .LBL extensions
    if filename.endswith('.CUB'):
        base_name = filename[:-4]
    else:
        base_name = filename.replace('.LBL', '')
    
    lbl_filename = base_name + '.LBL'
    lbl_path = os.path.join(THEMIS_DIR, lbl_filename)
    
    if not os.path.exists(lbl_path):
        raise HTTPException(status_code=404, detail=f"Metadata file not found: {lbl_filename}")
    
    try:
        metadata = parse_lbl_file(lbl_path)
        product_id = metadata.get('PRODUCT_ID', base_name)
        image_type = "IR" if product_id.upper().startswith('I') else "VIS"
        
        return {
            "product_id": product_id,
            "instrument": metadata.get('INSTRUMENT_NAME', 'THEMIS'),
            "image_type": image_type,
            "target": metadata.get('TARGET_NAME', 'MARS'),
            "center_latitude": float(metadata.get('CENTER_LATITUDE', 0)) if 'CENTER_LATITUDE' in metadata else None,
            "center_longitude": float(metadata.get('CENTER_LONGITUDE', 0)) if 'CENTER_LONGITUDE' in metadata else None,
            "raw_metadata": metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mars/themis/bands")
def get_band_info(image_type: Literal["VIS", "IR"] = Query(...)):
    """Get THEMIS band information"""
    
    if image_type == "IR":
        return {
            "image_type": "Infrared",
            "total_bands": 10,
            "wavelength_range": "6.78 - 14.88 μm",
            "bands": [band.dict() for band in THEMIS_IR_BANDS]
        }
    else:
        return {
            "image_type": "Visible",
            "total_bands": 5,
            "wavelength_range": "0.425 - 0.860 μm",
            "bands": [band.dict() for band in THEMIS_VIS_BANDS]
        }

@app.get("/mars/themis/image/{filename}")
def get_themis_image(
    filename: str,
    band: int = Query(1, ge=1, le=10),
    stretch: Literal["linear", "histogram"] = Query("linear"),
    format: Literal["png", "jpeg"] = Query("png")
):
    """Get THEMIS image as rendered PNG/JPEG"""
    
    file_path = os.path.join(THEMIS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    try:
        # Read cube directly (no decompression needed)
        data, width, height, num_bands = read_isis_cube_gdal(file_path, band)
        
        # Remove invalid values
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            raise HTTPException(status_code=404, detail="No valid data in this band")
        
        # Normalize
        if stretch == "histogram":
            vmin, vmax = np.percentile(valid_data, [2, 98])
        else:
            vmin, vmax = np.min(valid_data), np.max(valid_data)
        
        img_array = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        
        # Create image
        img = Image.fromarray(img_array, mode='L')
        
        # Save to buffer
        buf = io.BytesIO()
        img.save(buf, format=format.upper())
        buf.seek(0)
        
        return StreamingResponse(buf, media_type=f"image/{format}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/mars/themis/stats/{filename}")
def get_themis_stats(filename: str, band: int = Query(1, ge=1, le=10)):
    """Get statistics for a THEMIS band"""
    
    file_path = os.path.join(THEMIS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    try:
        # Read the cube
        data, width, height, num_bands = read_isis_cube_gdal(file_path, band)
        
        # Handle different data types and invalid values
        # Convert to float to handle NaN properly
        if data.dtype != np.float64 and data.dtype != np.float32:
            data = data.astype(float)
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(data)
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            raise HTTPException(status_code=404, detail="No valid data in this band")
        
        return {
            "filename": filename,
            "band": band,
            "dimensions": {"width": width, "height": height},
            "total_bands": num_bands,
            "min_value": float(np.min(valid_data)),
            "max_value": float(np.max(valid_data)),
            "mean_value": float(np.mean(valid_data)),
            "median_value": float(np.median(valid_data)),
            "std_value": float(np.std(valid_data)),
            "valid_pixels": int(len(valid_data)),
            "total_pixels": int(data.size)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)
