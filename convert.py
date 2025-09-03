import importlib
import logging
import platform
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Type

import rich.table
import torch
import typer
from docling_core.transforms.serializer.html import (
    HTMLDocSerializer,
    HTMLOutputStyle,
    HTMLParams,
)
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.types.doc import ImageRefMode
from docling_core.utils.file import resolve_source_to_path
from pydantic import TypeAdapter
from rich.console import Console

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.asr_model_specs import (
    WHISPER_BASE,
    WHISPER_LARGE,
    WHISPER_MEDIUM,
    WHISPER_SMALL,
    WHISPER_TINY,
    WHISPER_TURBO,
    AsrModelType,
)
from docling.datamodel.base_models import (
    ConversionStatus,
    FormatToExtensions,
    InputFormat,
    OutputFormat,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
    EasyOcrOptions,
    OcrOptions,
    PaginatedPipelineOptions,
    PdfBackend,
    PdfPipelineOptions,
    PipelineOptions,
    ProcessingPipeline,
    TableFormerMode,
    VlmPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_OLLAMA,
    GRANITE_VISION_TRANSFORMERS,
    SMOLDOCLING_MLX,
    SMOLDOCLING_TRANSFORMERS,
    VlmModelType,
)
from docling.document_converter import (
    AudioFormatOption,
    DocumentConverter,
    FormatOption,
    PdfFormatOption,
)
from docling.models.factories import get_ocr_factory
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

_log = logging.getLogger(__name__)

console = Console()
err_console = Console(stderr=True)

IS_MAC_ARM = sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}
DEFAULT_ACCELERATOR_DEVICE = AcceleratorDevice.MPS if IS_MAC_ARM else AcceleratorDevice.AUTO

print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

ocr_factory_internal = get_ocr_factory(allow_external_plugins=False)
ocr_engines_enum_internal = ocr_factory_internal.get_enum()

def _split_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    return re.split(r"[;,]", raw)

def convert_files(  # noqa: C901
    input_sources: Annotated[
        List[str],
        typer.Argument(
            ...,
            metavar="source",
            help="PDF files to convert. Can be local file / directory paths or URL.",
        ),
    ],
    from_formats: List[InputFormat] = typer.Option(
        None,
        "--from",
        help="Specify input formats to convert from. Defaults to all formats.",
    ),
    to_formats: List[OutputFormat] = typer.Option(
        None, "--to", help="Specify output formats. Defaults to Markdown."
    ),
    show_layout: Annotated[
        bool,
        typer.Option(
            ...,
            help="If enabled, the page images will show the bounding-boxes of the items.",
        ),
    ] = False,
    headers: str = typer.Option(
        None,
        "--headers",
        help="Specify http request headers used when fetching url input sources in the form of a JSON string",
    ),
    image_export_mode: Annotated[
        ImageRefMode,
        typer.Option(
            ...,
            help="Image export mode for the document (only in case of JSON, Markdown or HTML). With `placeholder`, only the position of the image is marked in the output. In `embedded` mode, the image is embedded as base64 encoded string. In `referenced` mode, the image is exported in PNG format and referenced from the main exported document.",
        ),
    ] = ImageRefMode.EMBEDDED,
    pipeline: Annotated[
        ProcessingPipeline,
        typer.Option(..., help="Choose the pipeline to process PDF or image files."),
    ] = ProcessingPipeline.STANDARD,
    vlm_model: Annotated[
        VlmModelType,
        typer.Option(..., help="Choose the VLM model to use with PDF or image files."),
    ] = VlmModelType.SMOLDOCLING,
    asr_model: Annotated[
        AsrModelType,
        typer.Option(..., help="Choose the ASR model to use with audio/video files."),
    ] = AsrModelType.WHISPER_TINY,
    ocr: Annotated[
        bool,
        typer.Option(
            ..., help="If enabled, the bitmap content will be processed using OCR."
        ),
    ] = True,
    force_ocr: Annotated[
        bool,
        typer.Option(
            ...,
            help="Replace any existing text with OCR generated text over the full content.",
        ),
    ] = False,
    ocr_engine: Annotated[
        str,
        typer.Option(
            ...,
            help=(
                f"The OCR engine to use. When --allow-external-plugins is *not* set, the available values are: "
                f"{', '.join(o.value for o in ocr_engines_enum_internal)}. "
                f"Use the option --show-external-plugins to see the options allowed with external plugins."
            ),
        ),
    ] = EasyOcrOptions.kind,
    ocr_lang: Annotated[
        Optional[str],
        typer.Option(
            ...,
            help="Provide a comma-separated list of languages used by the OCR engine. Note that each OCR engine has different values for the language names.",
        ),
    ] = None,
    pdf_backend: Annotated[
        PdfBackend, typer.Option(..., help="The PDF backend to use.")
    ] = PdfBackend.DLPARSE_V2,
    table_mode: Annotated[
        TableFormerMode,
        typer.Option(..., help="The mode to use in the table structure model."),
    ] = TableFormerMode.ACCURATE,
    enrich_code: Annotated[
        bool,
        typer.Option(..., help="Enable the code enrichment model in the pipeline."),
    ] = False,
    enrich_formula: Annotated[
        bool,
        typer.Option(..., help="Enable the formula enrichment model in the pipeline."),
    ] = False,
    enrich_picture_classes: Annotated[
        bool,
        typer.Option(
            ...,
            help="Enable the picture classification enrichment model in the pipeline.",
        ),
    ] = False,
    enrich_picture_description: Annotated[
        bool,
        typer.Option(..., help="Enable the picture description model in the pipeline."),
    ] = False,
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(..., help="If provided, the location of the model artifacts."),
    ] = None,
    enable_remote_services: Annotated[
        bool,
        typer.Option(
            ..., help="Must be enabled when using models connecting to remote services."
        ),
    ] = False,
    allow_external_plugins: Annotated[
        bool,
        typer.Option(
            ..., help="Must be enabled for loading modules from third-party plugins."
        ),
    ] = False,
    abort_on_error: Annotated[
        bool,
        typer.Option(
            ...,
            "--abort-on-error/--no-abort-on-error",
            help="If enabled, the processing will be aborted when the first error is encountered.",
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
    debug_visualize_cells: Annotated[
        bool,
        typer.Option(..., help="Enable debug output which visualizes the PDF cells"),
    ] = False,
    debug_visualize_ocr: Annotated[
        bool,
        typer.Option(..., help="Enable debug output which visualizes the OCR cells"),
    ] = False,
    debug_visualize_layout: Annotated[
        bool,
        typer.Option(
            ..., help="Enable debug output which visualizes the layour clusters"
        ),
    ] = False,
    debug_visualize_tables: Annotated[
        bool,
        typer.Option(..., help="Enable debug output which visualizes the table cells"),
    ] = False,
    document_timeout: Annotated[
        Optional[float],
        typer.Option(
            ...,
            help="The timeout for processing each document, in seconds.",
        ),
    ] = 500,
    num_threads: Annotated[int, typer.Option(..., help="Number of threads")] = 4,
    device: Annotated[
        AcceleratorDevice, typer.Option(..., help="Accelerator device")
    ] = DEFAULT_ACCELERATOR_DEVICE,
    tempdir: Optional[str] = None
) -> List[ConversionResult]:
    log_format = "%(asctime)s\t%(levelname)s\t%(name)s: %(message)s"

    if verbose == 0:
        logging.basicConfig(level=logging.WARNING, format=log_format)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    settings.debug.visualize_cells = debug_visualize_cells
    settings.debug.visualize_layout = debug_visualize_layout
    settings.debug.visualize_tables = debug_visualize_tables
    settings.debug.visualize_ocr = debug_visualize_ocr

    if from_formats is None:
        from_formats = list(InputFormat)

    parsed_headers: Optional[Dict[str, str]] = None

    input_doc_paths: List[Path] = []
    for src in input_sources:
        try:
            # check if we can fetch some remote url
            source = resolve_source_to_path(
                source=src, headers=parsed_headers, workdir=Path(tempdir)
            )
            input_doc_paths.append(source)
        except FileNotFoundError:
            err_console.print(
                f"[red]Error: The input file {src} does not exist.[/red]"
            )
            raise typer.Abort()
        except IsADirectoryError:
            # if the input matches to a file or a folder
            try:
                local_path = TypeAdapter(Path).validate_python(src)
                if local_path.exists() and local_path.is_dir():
                    for fmt in from_formats:
                        for ext in FormatToExtensions[fmt]:
                            input_doc_paths.extend(
                                list(local_path.glob(f"**/*.{ext}"))
                            )
                            input_doc_paths.extend(
                                list(local_path.glob(f"**/*.{ext.upper()}"))
                            )
                elif local_path.exists():
                    input_doc_paths.append(local_path)
                else:
                    err_console.print(
                        f"[red]Error: The input file {src} does not exist.[/red]"
                    )
                    raise typer.Abort()
            except Exception as err:
                err_console.print(f"[red]Error: Cannot read the input {src}.[/red]")
                _log.info(err)  # will print more details if verbose is activated
                raise typer.Abort()
        
    ocr_factory = get_ocr_factory(allow_external_plugins=allow_external_plugins)
    ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
        kind=ocr_engine,
        force_full_page_ocr=force_ocr,
    )

    ocr_lang_list = _split_list(ocr_lang)
    if ocr_lang_list is not None:
        ocr_options.lang = ocr_lang_list

    accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
    # pipeline_options: PaginatedPipelineOptions
    pipeline_options: PipelineOptions

    format_options: Dict[InputFormat, FormatOption] = {}

    if pipeline == ProcessingPipeline.STANDARD:
        pipeline_options = PdfPipelineOptions(
            allow_external_plugins=allow_external_plugins,
            enable_remote_services=enable_remote_services,
            accelerator_options=accelerator_options,
            do_ocr=ocr,
            ocr_options=ocr_options,
            do_table_structure=True,
            do_code_enrichment=enrich_code,
            do_formula_enrichment=enrich_formula,
            do_picture_description=enrich_picture_description,
            do_picture_classification=enrich_picture_classes,
            document_timeout=document_timeout,
        )
        pipeline_options.table_structure_options.do_cell_matching = (
            True  # do_cell_matching
        )
        pipeline_options.table_structure_options.mode = table_mode

        if image_export_mode != ImageRefMode.PLACEHOLDER:
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = (
                True  # FIXME: to be deprecated in version 3
            )
            pipeline_options.images_scale = 2

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

        pdf_format_option = PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=backend,  # pdf_backend
        )

        format_options = {
            InputFormat.PDF: pdf_format_option,
            InputFormat.IMAGE: pdf_format_option,
        }

    elif pipeline == ProcessingPipeline.VLM:
        pipeline_options = VlmPipelineOptions(
            enable_remote_services=enable_remote_services,
        )

        if vlm_model == VlmModelType.GRANITE_VISION:
            pipeline_options.vlm_options = GRANITE_VISION_TRANSFORMERS
        elif vlm_model == VlmModelType.GRANITE_VISION_OLLAMA:
            pipeline_options.vlm_options = GRANITE_VISION_OLLAMA
        elif vlm_model == VlmModelType.SMOLDOCLING:
            pipeline_options.vlm_options = SMOLDOCLING_TRANSFORMERS
            if sys.platform == "darwin":
                try:
                    import mlx_vlm

                    pipeline_options.vlm_options = SMOLDOCLING_MLX
                except ImportError:
                    _log.warning(
                        "To run SmolDocling faster, please install mlx-vlm:\n"
                        "pip install mlx-vlm"
                    )

        pdf_format_option = PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        )

        format_options = {
            InputFormat.PDF: pdf_format_option,
            InputFormat.IMAGE: pdf_format_option,
        }

    elif pipeline == ProcessingPipeline.ASR:
        pipeline_options = AsrPipelineOptions(
            # enable_remote_services=enable_remote_services,
            # artifacts_path = artifacts_path
        )

        if asr_model == AsrModelType.WHISPER_TINY:
            pipeline_options.asr_options = WHISPER_TINY
        elif asr_model == AsrModelType.WHISPER_SMALL:
            pipeline_options.asr_options = WHISPER_SMALL
        elif asr_model == AsrModelType.WHISPER_MEDIUM:
            pipeline_options.asr_options = WHISPER_MEDIUM
        elif asr_model == AsrModelType.WHISPER_BASE:
            pipeline_options.asr_options = WHISPER_BASE
        elif asr_model == AsrModelType.WHISPER_LARGE:
            pipeline_options.asr_options = WHISPER_LARGE
        elif asr_model == AsrModelType.WHISPER_TURBO:
            pipeline_options.asr_options = WHISPER_TURBO
        else:
            _log.error(f"{asr_model} is not known")
            raise ValueError(f"{asr_model} is not known")

        _log.info(f"pipeline_options: {pipeline_options}")

        audio_format_option = AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=pipeline_options,
        )

        format_options = {
            InputFormat.AUDIO: audio_format_option,
        }

    if artifacts_path is not None:
        pipeline_options.artifacts_path = artifacts_path
        # audio_pipeline_options.artifacts_path = artifacts_path

    doc_converter = DocumentConverter(
        allowed_formats=None,
        format_options=format_options,
    )

    start_time = time.time()

    _log.info(f"paths: {input_doc_paths}")
    conv_results = doc_converter.convert_all(
        input_doc_paths, raises_on_error=abort_on_error
    )

    end_time = time.time() - start_time

    _log.info(f"All documents were converted in {end_time:.2f} seconds.")
    return conv_results
