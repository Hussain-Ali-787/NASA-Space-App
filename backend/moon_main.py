# =====================  MOON NAC (LROC Narrow Angle Camera)  =====================
import os
import numpy as np
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI

app = FastAPI(title="Moon API", description="Endpoints for LRO NAC, LOLA, and WAC datasets", version="1.0")

# ---- Paths ----
NAC_DIR = "/mnt/c/NASA_Project/Moon_datasets/NAC"
NAC_PREV_DIR = os.path.join(NAC_DIR, "previews")
os.makedirs(NAC_PREV_DIR, exist_ok=True)

# ---- Helper: open NAC dataset ----
def _open_nac_dataset(filename: str) -> gdal.Dataset:
    fp = os.path.join(NAC_DIR, filename)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    ds = gdal.Open(fp, gdal.GA_ReadOnly)
    if ds is None:
        raise HTTPException(status_code=500, detail=f"GDAL could not open {filename}")
    return ds

# ---- 1) List NAC files ----
@app.get("/moon/nac/files", summary="List NAC files")
def list_nac_files():
    if not os.path.isdir(NAC_DIR):
        raise HTTPException(status_code=404, detail="NAC directory not found")
    files = sorted([f for f in os.listdir(NAC_DIR) if f.lower().endswith((".img", ".cub", ".tif", ".tiff"))])
    return {"nac_files": files}

# ---- 2) Preview NAC image ----
@app.get("/moon/nac/preview/{filename}", summary="Preview NAC image (PNG)")
def nac_preview(
    filename: str,
    size: int = Query(800, ge=256, le=4096, description="Preview width in px"),
):
    ds = _open_nac_dataset(filename)
    band = ds.GetRasterBand(1)
    if band is None:
        raise HTTPException(status_code=500, detail="No raster band found")

    src_w, src_h = band.XSize, band.YSize
    out_h = max(1, int(round(size * (src_h / src_w))))
    arr = band.ReadAsArray(0, 0, src_w, src_h, size, out_h).astype(float)

    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan

    # Contrast stretch
    p2, p98 = np.nanpercentile(arr, [2, 98])
    arr = np.clip(arr, p2, p98)

    # Save preview
    stem, _ = os.path.splitext(filename)
    out_path = os.path.join(NAC_PREV_DIR, f"{stem}_preview.png")

    plt.figure(figsize=(size / 100, out_h / 100), dpi=100)
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return FileResponse(out_path, media_type="image/png")

# ---- 3) Full NAC image ----
@app.get("/moon/nac/image/{filename}", summary="Get full NAC image")
def nac_image(filename: str):
    fp = os.path.join(NAC_DIR, filename)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fp, media_type="application/octet-stream")

# ---- 4) NAC stats ----
@app.get("/moon/nac/stats/{filename}", summary="NAC image stats")
def nac_stats(filename: str):
    ds = _open_nac_dataset(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)

    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan

    return {
        "filename": filename,
        "width": band.XSize,
        "height": band.YSize,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }

# ---- 5) NAC Feature Detection ----
@app.get("/moon/nac/features/detect", summary="Detect features in NAC imagery")
def detect_nac_features(
    lat: float = Query(..., description="Latitude in degrees"),
    lon: float = Query(..., description="Longitude in degrees"),
    radius_km: float = Query(5, description="Search radius in kilometers"),
    feature_types: str = Query("crater,rille,boulder", description="Comma-separated feature types"),
):
    """
    Placeholder: detects features within given radius around (lat, lon).
    In real pipeline, you'd call a ML model or feature catalog.
    """
    features = feature_types.split(",")

    # Dummy response
    detected = [
        {
            "type": "crater",
            "lat": lat + 0.01,
            "lon": lon + 0.01,
            "diameter_m": 500,
        },
        {
            "type": "boulder",
            "lat": lat - 0.005,
            "lon": lon - 0.004,
            "size_m": 15,
        },
    ]

    return JSONResponse(
        {
            "lat": lat,
            "lon": lon,
            "radius_km": radius_km,
            "requested_features": features,
            "detected_features": detected,
        }
    )
# =====================  END NAC block  =====================
# =====================  WAC (Wide Angle Camera Mosaic) =====================
import os
import numpy as np
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths ----
WAC_DIR = "/mnt/c/NASA_Project/moon_datasets/WAC"
WAC_PREV_DIR = os.path.join(WAC_DIR, "previews")
os.makedirs(WAC_PREV_DIR, exist_ok=True)

# ---- Helper: open WAC dataset ----
def _open_wac_dataset(filename: str) -> gdal.Dataset:
    fp = os.path.join(WAC_DIR, filename)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    ds = gdal.Open(fp, gdal.GA_ReadOnly)
    if ds is None:
        raise HTTPException(status_code=500, detail=f"GDAL could not open {filename}")
    return ds

# ---- 1) List WAC files ----
@app.get("/moon/wac/files", summary="List WAC mosaics")
def list_wac_files():
    if not os.path.isdir(WAC_DIR):
        raise HTTPException(status_code=404, detail="WAC directory not found")
    files = sorted([f for f in os.listdir(WAC_DIR) if f.lower().endswith(".tif")])
    return {"wac_files": files}

# ---- 2) Preview WAC mosaic ----
@app.get("/moon/wac/preview/{filename}", summary="WAC preview (PNG)", response_class=FileResponse)
def wac_preview(filename: str, size: int = Query(800, ge=256, le=4096)):
    ds = _open_wac_dataset(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray(0, 0, band.XSize, band.YSize, size, int(size * band.YSize / band.XSize)).astype(float)

    # Normalize for display
    p2, p98 = np.percentile(arr, [2, 98])
    arr = np.clip(arr, p2, p98)

    out_path = os.path.join(WAC_PREV_DIR, f"{filename}_preview.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return FileResponse(out_path, media_type="image/png")

# ---- 3) Stats ----
@app.get("/moon/wac/stats/{filename}", summary="WAC statistics")
def wac_stats(filename: str):
    ds = _open_wac_dataset(filename)
    band = ds.GetRasterBand(1)

    # Let GDAL compute statistics without loading full array
    stats = band.GetStatistics(True, True)
    if stats is None:
        raise HTTPException(status_code=500, detail="Could not compute statistics")

    return {
        "min": float(stats[0]),
        "max": float(stats[1]),
        "mean": float(stats[2]),
        "std": float(stats[3]),
        "width": band.XSize,
        "height": band.YSize
    }

# ---- 4) Crop by lat/lon ----
@app.get("/moon/wac/crop", summary="Crop WAC mosaic around lat/lon")
def wac_crop(filename: str, lat: float, lon: float, radius_km: float = 50):
    ds = _open_wac_dataset(filename)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)

    # Convert lat/lon → pixel coordinates
    x = int((lon - gt[0]) / gt[1])
    y = int((lat - gt[3]) / gt[5])

    # Radius in pixels (approx, assuming 100m resolution)
    radius_px = int(radius_km * 1000 / abs(gt[1]))

    arr = band.ReadAsArray(
        max(0, x - radius_px),
        max(0, y - radius_px),
        min(radius_px * 2, band.XSize),
        min(radius_px * 2, band.YSize)
    ).astype(float)

    out_path = os.path.join(WAC_PREV_DIR, f"{filename}_crop.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return FileResponse(out_path, media_type="image/png")

# =====================  END WAC block =====================
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import os
import rasterio
import numpy as np
from PIL import Image

# Path to LOLA dataset folder
LOLA_PATH = "/mnt/c/NASA_Project/moon_datasets/LOLA"


# 1. List LOLA files
@app.get("/moon/lola/files", summary="List LOLA files")
def list_lola_files():
    try:
        files = [f for f in os.listdir(LOLA_PATH) if f.endswith((".tif", ".img"))]
        return {"lola_files": files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 2. Get metadata / stats
@app.get("/moon/lola/stats/{filename}", summary="LOLA file statistics")
def lola_stats(filename: str):
    file_path = os.path.join(LOLA_PATH, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    try:
        with rasterio.open(file_path) as src:
            stats = {
                "filename": filename,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "bounds": src.bounds,
                "driver": src.driver,
                "count": src.count
            }
        return stats
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 3. Get elevation at lat/lon
@app.get("/moon/lola/elevation", summary="Elevation at given coordinates")
def get_lola_elevation(
    filename: str = Query(...),
    lat: float = Query(...),
    lon: float = Query(...)
):
    file_path = os.path.join(LOLA_PATH, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    try:
        with rasterio.open(file_path) as src:
            # Sample only one pixel instead of reading whole raster
            for val in src.sample([(lon, lat)]):
                elevation = float(val[0])
        return {"lat": lat, "lon": lon, "elevation": elevation}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 4. Preview
@app.get("/moon/lola/preview/{filename}", summary="Preview LOLA dataset")
def lola_preview(filename: str):
    file_path = os.path.join(LOLA_PATH, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    try:
        with rasterio.open(file_path) as src:
            # Read a small thumbnail instead of full dataset
            scale = 8  # Increase this for smaller preview (e.g., 16, 32)
            data = src.read(
                1,
                out_shape=(
                    src.height // scale,
                    src.width // scale
                )
            )

            arr = np.nan_to_num(data)
            arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)

            preview_path = os.path.join(LOLA_PATH, f"{filename}_preview.png")
            Image.fromarray(arr).save(preview_path)

        return FileResponse(preview_path, media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/moon/lola/crop/{filename}")
async def lola_crop(
    filename: str,
    lat: float = Query(..., description="Latitude in degrees"),
    lon: float = Query(..., description="Longitude in degrees"),
    radius_km: float = Query(50, description="Crop radius in kilometers")
):
    filepath = f"moon_datasets/LOLA/{filename}"

    with rasterio.open(filepath) as src:
        # Transform input coordinates to dataset CRS
        lon_t, lat_t = transform('EPSG:4326', src.crs, [lon], [lat])
        lon_t, lat_t = lon_t[0], lat_t[0]

        # Convert km radius → degrees (Moon radius ~1737 km)
        deg_per_km = 1 / (2 * np.pi * 1737) * 360
        lat_buffer = radius_km * deg_per_km
        lon_buffer = radius_km * deg_per_km

        # Define bounding box
        minx, maxx = lon_t - lon_buffer, lon_t + lon_buffer
        miny, maxy = lat_t - lat_buffer, lat_t + lat_buffer

        # Crop window
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(1, window=window)

    # Replace invalid values
    data = np.where((data == src.nodata) | np.isnan(data), 0, data)

    # Save cropped raster as temporary GeoTIFF
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=src.crs,
        transform=src.window_transform(window),
    ) as dst:
        dst.write(data, 1)

    buf.seek(0)
    return StreamingResponse(buf, media_type="image/tiff")
