#!/usr/bin/env python3
import os, glob, sys, zipfile, argparse
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import img2pdf
    HAS_IMG2PDF = True
except Exception:
    HAS_IMG2PDF = False

def natural_sort_key(path: str) -> str:
    return os.path.basename(path).lower()

def resize_max_side(im: Image.Image, max_side=1200) -> Tuple[Image.Image, float]:
    w, h = im.size
    scale = max(w, h) / float(max_side)
    if scale <= 1:
        return im.copy(), 1.0
    new_w, new_h = int(round(w/scale)), int(round(h/scale))
    return im.resize((new_w, new_h), Image.Resampling.LANCZOS), scale

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def central_crop_to_aspect(im: Image.Image, aspect=16/9, margin=0.06) -> Image.Image:
    w, h = im.size
    cur_aspect = w / h
    if cur_aspect > aspect:
        new_w = int(h * aspect)
        x0 = (w - new_w)//2
        x1 = x0 + new_w
        y0, y1 = 0, h
    else:
        new_h = int(w / aspect)
        y0 = (h - new_h)//2
        y1 = y0 + new_h
        x0, x1 = 0, w
    dx = int((x1-x0)*margin)
    dy = int((y1-y0)*margin)
    x0 += dx; x1 -= dx; y0 += dy; y1 -= dy
    return im.crop((x0, y0, x1, y1))

def detect_and_warp_screen(im_small: Image.Image, target_w=1600, target_h=900) -> Image.Image | None:
    if not HAS_CV2:
        return None
    img_cv = np.array(im_small)[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    screen_rect = None
    H, W = gray.shape[:2]
    for cnt in contours[:30]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area < (W*H*0.12):
                continue
            pts = approx.reshape(4,2).astype(np.float32)
            rect = order_points(pts)
            widthA = np.linalg.norm(rect[2]-rect[3])
            widthB = np.linalg.norm(rect[1]-rect[0])
            heightA = np.linalg.norm(rect[1]-rect[2])
            heightB = np.linalg.norm(rect[0]-rect[3])
            width = max(int(widthA), int(widthB))
            height = max(int(heightA), int(heightB))
            if width <= 0 or height <= 0:
                continue
            aspect = width/height
            if 1.25 <= aspect <= 2.4:
                screen_rect = rect
                break
    if screen_rect is None:
        return None
    dst = np.float32([[0,0],[target_w,0],[target_w,target_h],[0,target_h]])
    M = cv2.getPerspectiveTransform(screen_rect, dst)
    warp = cv2.warpPerspective(img_cv, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    return Image.fromarray(warp[:, :, ::-1], mode="RGB")

def safe_open_rgb(path: str) -> Image.Image | None:
    try:
        im = Image.open(path)
        try:
            exif = im.getexif()
            if 274 in exif:
                im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        return im.convert("RGB")
    except Exception:
        return None

from dataclasses import dataclass
@dataclass
class Config:
    input_dir: str
    out_dir: str
    out_pdf: str
    manifest_path: str
    batch_size: int = 500
    target_w: int = 1600
    target_h: int = 900
    note_height: int = 80
    max_side_detect: int = 1200

def build_manifest(input_dir: str, manifest_path: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic")
    files = []
    for root, _, fs in os.walk(input_dir):
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    files = sorted(files, key=lambda p: os.path.basename(p).lower())
    with open(manifest_path, "w", encoding="utf-8") as f:
        for p in files:
            f.write(p + "\n")
    return files

def read_manifest(manifest_path: str) -> list[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def process_batch(cfg: Config) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    if not os.path.exists(cfg.manifest_path):
        files = build_manifest(cfg.input_dir, cfg.manifest_path)
    else:
        files = read_manifest(cfg.manifest_path)

    existing_pages = sorted(glob.glob(os.path.join(cfg.out_dir, "page_*.jpg")))
    next_idx = len(existing_pages) + 1
    end_idx = min(next_idx + cfg.batch_size - 1, len(files))

    processed_now = 0
    for i in tqdm(range(next_idx, end_idx+1), desc="Processing", unit="img"):
        path = files[i-1]
        im = safe_open_rgb(path)
        if im is None:
            continue
        im_small, _ = resize_max_side(im, max_side=cfg.max_side_detect)
        ppt = detect_and_warp_screen(im_small, cfg.target_w, cfg.target_h)
        if ppt is None:
            cropped = central_crop_to_aspect(im_small, aspect=16/9, margin=0.06)
            ppt = cropped.resize((cfg.target_w, cfg.target_h), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (cfg.target_w, cfg.target_h + cfg.note_height), "white")
        canvas.paste(ppt, (0,0))
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, cfg.target_h), (cfg.target_w, cfg.target_h)], fill=(200,200,200), width=2)
        out_path = os.path.join(cfg.out_dir, f"page_{i:04d}.jpg")
        canvas.save(out_path, "JPEG", quality=88, subsampling=1)
        processed_now += 1

    pages = sorted(glob.glob(os.path.join(cfg.out_dir, "page_*.jpg")))
    if pages and HAS_IMG2PDF:
        with open(cfg.out_pdf, "wb") as f:
            f.write(img2pdf.convert(pages))

    return {
        "total_images_in_manifest": len(files),
        "already_had_pages": len(existing_pages),
        "processed_this_run": processed_now,
        "total_pages_now": len(pages),
        "pdf_path": os.path.abspath(cfg.out_pdf),
        "pages_dir": os.path.abspath(cfg.out_dir),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing photos (recursively).")
    ap.add_argument("--out-dir", default="ppt_pages", help="Output pages directory.")
    ap.add_argument("--out-pdf", default="PPT_with_notes.pdf", help="Output merged PDF path.")
    ap.add_argument("--manifest", default="manifest.txt", help="Manifest file to support resume.")
    ap.add_argument("--batch-size", type=int, default=500, help="Images per run (resume supported).")
    ap.add_argument("--note-height", type=int, default=80, help="Note area height in pixels.")
    ap.add_argument("--target-w", type=int, default=1600, help="Warped PPT width.")
    ap.add_argument("--target-h", type=int, default=900, help="Warped PPT height.")
    ap.add_argument("--max-side-detect", type=int, default=1200, help="Downscale for detection speed.")
    args = ap.parse_args()

    cfg = Config(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        out_pdf=args.out_pdf,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        target_w=args.target_w, target_h=args.target_h,
        note_height=args.note_height,
        max_side_detect=args.max_side_detect
    )
    stats = process_batch(cfg)
    print(stats)

if __name__ == "__main__":
    main()


# =============================================================================
# ppt_extractor_batch.py
#
# Example usage:
'''
  # 1️⃣ 基础用法：处理桌面上一个文件夹（递归处理所有照片）
  python ppt_extractor_batch.py \
      --input-dir "/mnt/c/Users/YourName/Desktop/BEIHAI_SUMMIT" \
      --out-dir "./BEIHAI_pages" \
      --out-pdf "./BEIHAI_SUMMIT_PPT_with_notes.pdf"

  # 2️⃣ 可调参数示例：
  python ppt_extractor_batch.py \
      --input-dir "/home/zitiantang/DesktopWinShared/wsl_shared/combine/BEIHAI_SUMMIT/converted" \
      --out-dir "./Section_1+2_pages" \
      --out-pdf "./Section_1+2_slides.pdf" \
      --batch-size 500 \
      --note-height 80 \
      --target-w 1920 --target-h 1080

  # 3️⃣ 高速模式（检测更快但略降精度）
  python ppt_extractor_batch.py \
      --input-dir "./photos" \
      --max-side-detect 800

  # 4️⃣ 断点续跑：
  #    重新执行同一命令时，会自动从 manifest.txt 中记录的上次进度继续。

  # 5️⃣ 如果你用的是WSL-Ubuntu且文件在桌面：
  #    Windows 路径 /Users/YourName/Desktop 对应 WSL 下的 /mnt/c/Users/YourName/Desktop
'''
#
# =============================================================================

