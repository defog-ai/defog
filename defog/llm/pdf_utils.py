"""
PDF utilities for processing PDFs with page-based splitting and metadata extraction.
"""

import base64
import httpx
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.debug("PyMuPDF not installed. PDF parsing will not be available.")
    logger.debug("Install with: pip install pymupdf")
    pass


class PDFProcessor:
    """Handles PDF downloading, processing, and splitting for Claude API."""

    # Claude API limits
    MAX_PAGES_PER_CHUNK = 80
    MAX_SIZE_BYTES = 24 * 1024 * 1024  # 24MB

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"User-Agent": "Claude-PDF-Processor/1.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def download_pdf(self, url: str) -> bytes:
        """
        Download PDF from URL.

        Args:
            url: PDF URL to download

        Returns:
            PDF content as bytes

        Raises:
            ValueError: If URL is invalid or download fails
            httpx.HTTPError: For network-related errors
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        if not self.client:
            raise RuntimeError("PDFProcessor must be used as async context manager")

        try:
            response = await self.client.get(url)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type and not url.lower().endswith(
                ".pdf"
            ):
                logger.warning(f"Content-Type is {content_type}, proceeding anyway")

            # Read content
            content = response.content

            if len(content) == 0:
                raise ValueError("Downloaded PDF is empty")

            logger.info(f"Downloaded PDF: {len(content)} bytes from {url}")
            return content

        except httpx.HTTPError as e:
            raise httpx.HTTPError(f"Failed to download PDF from {url}: {e}")

    def get_pdf_metadata(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Extract metadata from PDF content.

        Args:
            pdf_content: PDF content as bytes

        Returns:
            Dictionary with page_count, size_bytes, and other metadata

        Raises:
            ValueError: If PDF is corrupted or unreadable
        """
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            metadata = {
                "page_count": len(doc),
                "size_bytes": len(pdf_content),
                "size_mb": len(pdf_content) / (1024 * 1024),
                "is_encrypted": doc.needs_pass,
                "metadata": doc.metadata,
            }

            doc.close()
            logger.info(
                f"PDF metadata: {metadata['page_count']} pages, {metadata['size_mb']:.2f} MB"
            )
            return metadata

        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")

    def should_split_pdf(self, metadata: Dict[str, Any]) -> bool:
        """
        Determine if PDF should be split based on Claude's limits.

        Args:
            metadata: PDF metadata from get_pdf_metadata

        Returns:
            True if PDF should be split
        """
        return (
            metadata["page_count"] > self.MAX_PAGES_PER_CHUNK
            or metadata["size_bytes"] > self.MAX_SIZE_BYTES
        )

    def split_pdf_by_pages(
        self, pdf_content: bytes, metadata: Dict[str, Any]
    ) -> List[bytes]:
        """
        Split PDF into chunks based on page limits and size constraints.

        Args:
            pdf_content: Original PDF content
            metadata: PDF metadata

        Returns:
            List of PDF chunks as bytes
        """
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(doc)

            # Calculate optimal chunk size
            if metadata["size_bytes"] > self.MAX_SIZE_BYTES:
                # Size-based splitting
                size_ratio = metadata["size_bytes"] / self.MAX_SIZE_BYTES
                pages_per_chunk = max(1, int(self.MAX_PAGES_PER_CHUNK / size_ratio))
            else:
                # Page-based splitting
                pages_per_chunk = self.MAX_PAGES_PER_CHUNK

            chunks = []

            for start_page in range(0, total_pages, pages_per_chunk):
                end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)

                # Create new document with page range
                chunk_doc = fitz.open()
                chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)

                # Convert to bytes
                chunk_bytes = chunk_doc.tobytes()
                chunks.append(chunk_bytes)

                chunk_doc.close()

                logger.info(
                    f"Created chunk {len(chunks)}: pages {start_page + 1}-{end_page + 1}, {len(chunk_bytes)} bytes"
                )

            doc.close()

            logger.info(f"Split PDF into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            raise ValueError(f"Failed to split PDF: {e}")

    def encode_pdf_for_claude(self, pdf_content: bytes) -> str:
        """
        Encode PDF content for Claude API.

        Args:
            pdf_content: PDF content as bytes

        Returns:
            Base64 encoded PDF content
        """
        return base64.b64encode(pdf_content).decode("utf-8")

    async def process_pdf_from_url(self, url: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Complete PDF processing pipeline from URL.

        Args:
            url: PDF URL to process

        Returns:
            Tuple of (list of base64 encoded PDF chunks, metadata)
        """
        # Download PDF
        pdf_content = await self.download_pdf(url)

        # Get metadata
        metadata = self.get_pdf_metadata(pdf_content)

        # Check if splitting is needed
        if self.should_split_pdf(metadata):
            # Split PDF
            chunks = self.split_pdf_by_pages(pdf_content, metadata)
            # Encode each chunk
            encoded_chunks = [self.encode_pdf_for_claude(chunk) for chunk in chunks]
            metadata["split"] = True
            metadata["chunk_count"] = len(chunks)
        else:
            # Single chunk
            encoded_chunks = [self.encode_pdf_for_claude(pdf_content)]
            metadata["split"] = False
            metadata["chunk_count"] = 1

        return encoded_chunks, metadata


async def download_and_process_pdf(url: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convenience function to download and process a PDF from URL.

    Args:
        url: PDF URL to process

    Returns:
        Tuple of (list of base64 encoded PDF chunks, metadata)
    """
    async with PDFProcessor() as processor:
        return await processor.process_pdf_from_url(url)
