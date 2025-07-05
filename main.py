import os
from pathlib import Path
from typing import List, Optional, Type
import tempfile
import shutil
import zipfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, ConversionResult
from docling.utils.model_downloader import download_models
from docling_core.types.doc import ImageRefMode
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

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

def convert(
    file_path: Path,
    artifacts_path: str,
    force_ocr: Optional[bool] = False,
    do_formula_enrichment: Optional[bool] = True,
    do_code_enrichment: Optional[bool] = True,
    do_picture_classification: Optional[bool] = True,
    backend:  Type[PdfDocumentBackend] = PdfBackend.DLPARSE_V4,
    images_scale: float = IMAGE_RESOLUTION_SCALE,
    generate_picture_images: bool = True
) -> ConversionResult:
    print(file_path)

    backend: Type[PdfDocumentBackend]
    if pdf_backend == PdfBackend.DLPARSE_V1:
        backend = DoclingParseDocumentBackend
    elif pdf_backend == PdfBackend.DLPARSE_V2:
        backend = DoclingParseV2DocumentBackend
    elif pdf_backend == PdfBackend.DLPARSE_V4:
        backend = DoclingParseV4DocumentBackend  # type: ignore
    elif pdf_backend == PdfBackend.PYPDFIUM2:
        backend = PyPdfiumDocumentBackend  # type: ignore
    else:
        raise RuntimeError(f"Unexpected PDF backend type {pdf_backend}")

    pdf_pipeline = PdfPipelineOptions(artifacts_path=artifacts_path)
    pdf_pipeline.images_scale = images_scale
    pdf_pipeline.do_code_enrichment = True
    pdf_pipeline.do_formula_enrichment = True
    pdf_pipeline.do_picture_classification = True
    pdf_pipeline.generate_picture_images = generate_picture_images
    pdf_pipeline.generate_table_images = True
    pdf_pipeline.generate_page_images = True

    if THREADS is not None and THREADS > 0:
        accelerator_options = AcceleratorOptions(
            num_threads=THREADS, device=AcceleratorDevice.CPU
        )
        pdf_pipeline.accelerator_options = accelerator_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline, backend=backend)
        }
    )
    result = converter.convert(str(file_path))
    return result

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

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    img_ref_mode: Optional[ImageRefMode] = Form(ImageRefMode.EMBEDDED),
    images_scale: Optional[float] = Form(IMAGE_RESOLUTION_SCALE),
    generate_picture_images: Optional[bool] = Form(True),
    force_ocr: Optional[bool] = Form(False),
    do_formula_enrichment: Optional[bool] = Form(True),
    do_code_enrichment: Optional[bool] = Form(True),
    do_picture_classification: Optional[bool] = Form(True),
):
    if WEIGHTS_PATH:
        artifacts_path = Path(WEIGHTS_PATH)
        os.environ["HF_HOME"] = str(artifacts_path)
        tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = Path(tmpdir.name)
    else:
        tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = Path(tmpdir.name)
        artifacts_path = tmpdir_path / "artifacts"
        os.environ["HF_HOME"] = str(artifacts_path / "huggingface")

    file_path = tmpdir_path / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = convert(
        file_path=file_path,
        artifacts_path=str(artifacts_path),
        force_ocr=force_ocr,
        do_formula_enrichment=do_formula_enrichment,
        do_code_enrichment=do_code_enrichment,
        do_picture_classification=do_picture_classification,
        images_scale=images_scale,
        generate_picture_images=generate_picture_images,
    )

    zip_path = package_output(result, tmpdir_path, base_name=file.filename.rsplit(".", 1)[0], img_mode=img_ref_mode)
    return StreamingResponse(
        open(zip_path, "rb"),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_path.name}"}
    )

@app.post("/batch-convert")
async def batch_convert(
    files: List[UploadFile] = File(...),
    img_ref_mode: Optional[ImageRefMode] = Form(ImageRefMode.EMBEDDED),
    force_ocr: Optional[bool] = Form(False),
    do_formula_enrichment: Optional[bool] = Form(True),
    do_code_enrichment: Optional[bool] = Form(True),
    do_picture_classification: Optional[bool] = Form(True),
    images_scale: Optional[float] = Form(IMAGE_RESOLUTION_SCALE),
    generate_picture_images: Optional[bool] = Form(True),
):
    if WEIGHTS_PATH:
        artifacts_path = Path(WEIGHTS_PATH)
        os.environ["HF_HOME"] = str(artifacts_path)
        tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = Path(tmpdir.name)
    else:
        tmpdir = tempfile.TemporaryDirectory()
        tmpdir_path = Path(tmpdir.name)
        artifacts_path = tmpdir_path / "artifacts"
        os.environ["HF_HOME"] = str(artifacts_path / "huggingface")

    zip_path = tmpdir_path / "batch_output.zip"
    with zipfile.ZipFile(zip_path, "w") as batch_zip:
        for file in files:
            file_path = tmpdir_path / file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            result = convert(
                file_path=file_path,
                artifacts_path=str(artifacts_path),
                force_ocr=force_ocr,
                do_formula_enrichment=do_formula_enrichment,
                do_code_enrichment=do_code_enrichment,
                do_picture_classification=do_picture_classification,
                images_scale=images_scale,
                generate_picture_images=generate_picture_images,
            )
            subdir = tmpdir_path / file.filename.rsplit(".", 1)[0]
            subdir.mkdir(exist_ok=True)
            sub_zip = package_output(result, subdir, base_name=file.filename.rsplit(".", 1)[0], img_mode=img_ref_mode)
            batch_zip.write(sub_zip, arcname=f"{file.filename.rsplit('.', 1)[0]}.zip")

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

    uvicorn.run(app, host=host, port=port)