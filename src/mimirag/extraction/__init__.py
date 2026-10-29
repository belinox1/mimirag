"""Text extraction: plain text, HTML and PDF (with optional OCR and banner removal)."""

from .document import extract_document, extract_page
from .html import html_text
from .pdf import banner_lines, blank_banners, ocr_pages, pdf_pages

__all__ = ['banner_lines', 'blank_banners', 'extract_document', 'extract_page', 'html_text', 'ocr_pages', 'pdf_pages']
