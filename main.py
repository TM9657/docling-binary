import os
from pathlib import Path
from typing import List, Optional, Type
import tempfile
import shutil
import zipfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from docling_core.types.doc import ImageRefMode
from docling.datamodel.document import ConversionResult
from docling.utils.model_downloader import download_models
from convert import convert_files
from docling.datamodel.pipeline_options import (
    PdfBackend,
)

WEIGHTS_PATH = os.environ.get("HF_HOME", None)
IMAGE_RESOLUTION_SCALE = 2.0
LOCK_FILE = ".models_downloaded.lock"
THREADS = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model download logic (runs once at startup)
    if WEIGHTS_PATH:
        artifacts_path = Path(WEIGHTS_PATH)
        os.environ["HF_HOME"] = str(artifacts_path)
    else:
        artifacts_path = Path(tempfile.gettempdir()) / "docling_artifacts"
        os.environ["HF_HOME"] = str(artifacts_path / "huggingface")
        artifacts_path.mkdir(parents=True, exist_ok=True)

    lock_path = artifacts_path / LOCK_FILE
    if not lock_path.exists():
        download_models(output_dir=artifacts_path, progress=True, force=False)
        lock_path.touch()
    yield

app = FastAPI(lifespan=lifespan)

def package_output(result: ConversionResult, output_dir: Path, base_name: str, img_mode: ImageRefMode = ImageRefMode.EMBEDDED) -> Path:
    print(f"Packaging output to {output_dir}")
    print(f"Image mode: {img_mode}")
    md_path = output_dir / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown(image_mode=img_mode, page_break_placeholder="<--- PAGE BREAK --->"))

    images_dir = output_dir / "images"
    if images_dir.exists():
        image_files = list(images_dir.iterdir())
    else:
        image_files = []

    zip_path = output_dir / f"{base_name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(md_path, arcname=f"{base_name}.md")
        for img_file in image_files:
            zipf.write(img_file, arcname=f"images/{img_file.name}")
    return zip_path

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Service is running"}

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    image_export_mode: Optional[ImageRefMode] = Form(ImageRefMode.EMBEDDED),
    force_ocr: Optional[bool] = Form(False),
    ocr_lang: Optional[str] = Form(None),
    pdf_backend: Optional[PdfBackend] = Form(PdfBackend.DLPARSE_V4),
    table_mode: Optional[bool] = Form(False),
    enrich_code: Optional[bool] = Form(True),
    enrich_formula: Optional[bool] = Form(False),
    enrich_picture_classes: Optional[bool] = Form(False),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        if WEIGHTS_PATH:
            artifacts_path = Path(WEIGHTS_PATH)
            os.environ["HF_HOME"] = str(artifacts_path)
            tmpdir_path = Path(tmpdir)
        else:
            tmpdir_path = Path(tmpdir)
            artifacts_path = tmpdir_path / "artifacts"
            os.environ["HF_HOME"] = str(artifacts_path / "huggingface")

        file_path = tmpdir_path / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        results = list(convert_files(
            artifacts_path=str(artifacts_path),
            input_sources=[str(file_path)],
            image_export_mode=image_export_mode,
            force_ocr=force_ocr,
            ocr_lang=ocr_lang,
            pdf_backend=pdf_backend,
            table_mode=table_mode,
            enrich_code=enrich_code,
            enrich_formula=enrich_formula,
            enrich_picture_classes=enrich_picture_classes,
            num_threads=THREADS if THREADS else 4,
            tempdir=str(tmpdir_path)
        ))
    
        zip_path = package_output(results[0], tmpdir_path, base_name=file.filename.rsplit(".", 1)[0], img_mode=image_export_mode)
        return StreamingResponse(
            open(zip_path, "rb"),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_path.name}"}
        )

@app.post("/batch-convert")
async def batch_convert(
    files: List[UploadFile] = File(...),
    image_export_mode: Optional[ImageRefMode] = Form(ImageRefMode.EMBEDDED),
    force_ocr: Optional[bool] = Form(False),
    ocr_lang: Optional[str] = Form(None),
    pdf_backend: Optional[PdfBackend] = Form(PdfBackend.DLPARSE_V4),
    table_mode: Optional[bool] = Form(False),
    enrich_code: Optional[bool] = Form(True),
    enrich_formula: Optional[bool] = Form(False),
    enrich_picture_classes: Optional[bool] = Form(False),
):
    with tempfile.TemporaryDirectory() as tmpdir:
        if WEIGHTS_PATH:
            artifacts_path = Path(WEIGHTS_PATH)
            os.environ["HF_HOME"] = str(artifacts_path)
            tmpdir_path = Path(tmpdir)
        else:
            tmpdir_path = Path(tmpdir)
            artifacts_path = tmpdir_path / "artifacts"
            os.environ["HF_HOME"] = str(artifacts_path / "huggingface")

        file_paths = []
        for file in files:
            file_path = tmpdir_path / file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(str(file_path))

        # Batch process all files at once
        results = list(convert_files(
            artifacts_path=str(artifacts_path),
            input_sources=file_paths,
            image_export_mode=image_export_mode,
            force_ocr=force_ocr,
            ocr_lang=ocr_lang,
            pdf_backend=pdf_backend,
            table_mode=table_mode,
            enrich_code=enrich_code,
            enrich_formula=enrich_formula,
            enrich_picture_classes=enrich_picture_classes,
            num_threads=THREADS if THREADS else 4,
            tempdir=str(tmpdir_path)
        ))

        zip_path = tmpdir_path / "batch_output.zip"
        with zipfile.ZipFile(zip_path, "w") as batch_zip:
            for file, result in zip(files, results):
                base_name = file.filename.rsplit(".", 1)[0]
                subdir = tmpdir_path / base_name
                subdir.mkdir(exist_ok=True)
                sub_zip = package_output(result, subdir, base_name=base_name, img_mode=img_ref_mode)
                batch_zip.write(sub_zip, arcname=f"{base_name}.zip")

        return StreamingResponse(
            open(zip_path, "rb"),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=batch_output.zip"}
        )

if __name__ == "__main__":
    import multiprocessing
    import sys
    import uvicorn
    multiprocessing.freeze_support()
    host = "0.0.0.0"
    port = 8000
    weights_path = None 
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Invalid port '{sys.argv[2]}', using default port 8000.")
    if len(sys.argv) > 3:
        weights_path = sys.argv[3]
        os.environ["HF_HOME"] = weights_path 
    if len(sys.argv) > 4:
        try:
            threads = int(sys.argv[4])
            if threads > 0:
                THREADS = threads
                print(f"Using {THREADS} threads for processing.")
            else:
                print("Invalid number of threads, using default (None).")
        except ValueError:
            print(f"Invalid thread count '{sys.argv[4]}', using default (None).")
    if weights_path:
        WEIGHTS_PATH = weights_path
        print(f"Using custom HF_HOME: {WEIGHTS_PATH}")

    uvicorn.run(app, host=host, port=port, timeout_keep_alive=1000, timeout_graceful_shutdown=1000)