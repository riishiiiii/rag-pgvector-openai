from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, MarkdownFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from typing import Iterator

class DoclingFileLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend),
                InputFormat.MD: MarkdownFormatOption(pipeline_cls=SimplePipeline),
            },
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

