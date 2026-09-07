"""
Analysis Service Module for ODIN V1 Phase 2 & 3
Orchestrates the document analysis process: retrieve → analyze → classify → process → persist.
"""

from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import logging
import numpy as np
from sqlalchemy import select, delete

from app.services.analysis.pdf_analyzer import PDFAnalyzer, PageAnalysis
from app.services.analysis.page_classifier import PageClassifier, ClassificationResult
from app.core.database import get_async_session_local
from app.services.document_repository import DocumentRepository
from app.storage.factory import get_storage_backend
from app.core.config import get_settings
from app.models.document import Document

# Import processing services
from app.services.processing.content_router import ContentRouter, ProcessingProfile
from app.services.processing.preprocessing_service import PreprocessingService
from app.services.processing.visual_region_service import VisualRegionService
from app.services.processing.visual_embedding_service import VisualEmbeddingService
import cv2
import numpy as np
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service responsible for orchestrating document analysis workflow.

    Workflow:
    1. Retrieve document from storage
    2. Analyze each page for structural signals
    3. Classify each page based on signals
    4. Route to appropriate processing pipeline
    5. Apply preprocessing and extract features
    6. Persist analysis results
    7. Update document status
    """

    def __init__(self):
        """Initialize the analysis service."""
        self.settings = get_settings()
        self.storage_backend = get_storage_backend()
        self.pdf_analyzer = PDFAnalyzer()
        self.page_classifier = PageClassifier()
        
        # Initialize processing services
        self.content_router = ContentRouter()
        self.preprocessing_service = PreprocessingService()
        self.visual_region_service = VisualRegionService()
        self.visual_embedding_service = VisualEmbeddingService()

    async def analyze_document(
        self,
        document_id: str,
        force_reanalysis: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a document and persist page-level classifications.

        Args:
            document_id: UUID of the document to analyze
            force_reanalysis: If True, re-analyze even if already analyzed

        Returns:
            Dictionary containing analysis summary

        Raises:
            ValueError: If document not found
            Exception: For analysis or persistence failures
        """
        settings = get_settings()

        # Create database session
        async with get_async_session_local()() as session:
            # Get document repository
            doc_repo = DocumentRepository(session)

            # Retrieve document
            document = await doc_repo.get_by_id(document_id)
            if not document:
                raise ValueError(f"Document with ID {document_id} not found")

            # Check if already analyzed (unless force_reanalysis)
            if not force_reanalysis and await self._is_analyzed(session, document_id):
                logger.info(f"Document {document_id} already analyzed. Use force_reanalysis=True to re-analyze.")
                return await self.get_analysis_results(document_id)

            # Update status to analyzing
            await doc_repo.update_status(document_id, "analyzing")
            await session.commit()

            try:
                # Retrieve document from storage
                logger.info(f"Retrieving document {document_id} from storage")
                pdf_bytes = await self._retrieve_document_bytes(document.storage_reference)

                # Analyze PDF
                logger.info(f"Analyzing PDF for document {document_id}")
                page_analyses = self.pdf_analyzer.analyze_pdf(pdf_bytes)

                # Classify each page and prepare page records
                page_results = []
                document_pages = []  # For bulk creation

                for analysis in page_analyses:
                    classification_result = self.page_classifier.classify_page(analysis)

                    # Get processing profile based on classification
                    processing_profile = self.content_router.get_processing_profile(classification_result.classification)
                    
                    # Convert analysis to image for preprocessing (simplified - in practice would render PDF page)
                    # For now, we'll create a placeholder image based on analysis signals
                    processing_image = self._create_placeholder_image(analysis)
                    
                    # Apply preprocessing
                    preprocessed_image = self.preprocessing_service.preprocess_page(
                        processing_image, 
                        processing_profile, 
                        classification_result
                    )
                    
                    # Extract visual regions for visual content
                    visual_regions = []
                    if processing_profile in [ProcessingProfile.VISUAL, ProcessingProfile.TABLE_PRESERVING]:
                        visual_regions = self.visual_region_service.extract_visual_regions(
                            preprocessed_image, 
                            classification_result
                        )
                    
                    # Generate visual embedding for multimodal indexing
                    visual_embedding = None
                    if visual_regions or processing_profile == ProcessingProfile.VISUAL:
                        embedding_result = self.visual_embedding_service.create_embedding_from_analysis(
                            preprocessed_image,
                            classification_result,
                            visual_regions
                        )
                        visual_embedding = embedding_result["embedding"]

                    # Prepare analysis metadata with processing information
                    analysis_metadata = analysis.to_dict()
                    analysis_metadata.update({
                        "processing_profile": processing_profile.value,
                        "visual_regions_count": len(visual_regions),
                        "has_visual_embedding": visual_embedding is not None,
                        "visual_embedding_dim": len(visual_embedding) if visual_embedding else 0
                    })

                    # Convert to dictionary for storage
                    #
                    # NOTE: DocumentPage.classification_confidence is a
                    # String(20) column (see app/models/document.py), but
                    # PageClassificationResult.classification_confidence is
                    # a float. asyncpg does not silently coerce a Python
                    # float into a VARCHAR bind parameter, so this must be
                    # stringified here or every page-analysis insert fails
                    # with "invalid input for query argument ... (expected
                    # str, got float)".
                    page_data = {
                        "document_id": document_id,
                        "page_number": analysis.page_number,
                        "classification": classification_result.classification,
                        "classification_confidence": str(classification_result.classification_confidence),
                        "classification_reason": classification_result.classification_reason,
                        "text_length": analysis.text_length,
                        "image_count": analysis.image_count,
                        "drawing_count": analysis.drawing_count,
                        "analysis_metadata": analysis_metadata
                    }

                    # Store visual embedding in metadata if available
                    if visual_embedding:
                        page_data["analysis_metadata"]["visual_embedding"] = visual_embedding

                    page_results.append(page_data)
                    document_pages.append(page_data)

                # Persist page analysis results
                await self._persist_page_analyses(session, document_pages)

                # Update document status to analyzed
                await doc_repo.update_status(document_id, "analyzed")
                await session.commit()

                logger.info(f"Successfully analyzed and classified document {document_id}")

                # Return analysis summary
                return {
                    "document_id": document_id,
                    "status": "analyzed",
                    "page_count": len(page_analyses),
                    "pages": page_results
                }

            except Exception as e:
                # Update status to analysis_failed on error
                await doc_repo.update_status(document_id, "analysis_failed")
                await session.commit()
                logger.error(f"Analysis failed for document {document_id}: {e}")
                raise

    async def _retrieve_document_bytes(self, storage_reference: str) -> bytes:
        """
        Retrieve document bytes from storage.

        Args:
            storage_reference: Storage reference (path or key)

        Returns:
            Document bytes
        """
        file_obj = await self.storage_backend.retrieve_file(storage_reference)

        # Read all bytes from the file object
        if hasattr(file_obj, 'read'):
            bytes_data = await file_obj.read()
        else:
            bytes_data = file_obj  # Already bytes

        # Close the file object if it has a close method
        if hasattr(file_obj, 'close'):
            await file_obj.close()

        return bytes_data

    async def _is_analyzed(self, session, document_id: str) -> bool:
        """
        Check if a document has already been analyzed.

        Args:
            session: Database session
            document_id: Document ID to check

        Returns:
            True if document has been analyzed, False otherwise
        """
        # Import here to avoid circular imports
        from app.models.document import DocumentPage

        stmt = select(DocumentPage).where(DocumentPage.document_id == document_id).limit(1)
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def _persist_page_analyses(
        self,
        session,
        page_data_list: List[Dict[str, Any]]
    ) -> None:
        """
        Persist page analysis results to database.

        Args:
            session: Database session
            page_data_list: List of page data dictionaries
        """
        # Import here to avoid circular imports
        from app.models.document import DocumentPage

        # Delete existing analyses for this document (if not doing incremental updates)
        # For now, we'll replace all analyses when re-analyzing
        delete_stmt = delete(DocumentPage).where(
            DocumentPage.document_id == page_data_list[0]["document_id"]
        )
        await session.execute(delete_stmt)

        # Create new page analysis records
        page_objects = [
            DocumentPage(**page_data) for page_data in page_data_list
        ]
        session.add_all(page_objects)
        await session.commit()

    def _create_placeholder_image(self, analysis: PageAnalysis) -> np.ndarray:
        """
        Create a placeholder image based on analysis signals for processing.
        In a full implementation, this would be the actual rendered PDF page.
        
        Args:
            analysis: PageAnalysis object
            
        Returns:
            Grayscale numpy array representing the page image
        """
        # Create an image based on analysis signals
        # This is a simplified version - in reality we'd render the PDF page
        height, width = int(analysis.height), int(analysis.width)
        
        # Create base image
        image = np.ones((height, width), dtype=np.uint8) * 255  # White background
        
        # Add visual elements based on analysis signals
        # Text areas (darker regions)
        if analysis.text_length > 0:
            # Add some text-like texture
            text_area_height = min(height // 4, 100)
            text_area = image[:text_area_height, :]
            # Add noise to simulate text
            noise = np.random.randint(0, 50, text_area.shape, dtype=np.uint8)
            image[:text_area_height, :] = cv2.subtract(text_area, noise)
        
        # Image areas (rectangular regions)
        if analysis.image_count > 0:
            for i in range(min(analysis.image_count, 3)):  # Max 3 image placeholders
                x = (width // 4) + (i * width // 4)
                y = height // 3
                w, h = width // 6, height // 3
                if x + w < width and y + h < height:
                    # Add image placeholder (slightly darker rectangle)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), -1)
                    # Add some texture inside
                    roi = image[y:y+h, x:x+w]
                    noise = np.random.randint(180, 220, roi.shape, dtype=np.uint8)
                    image[y:y+h, x:x+w] = noise
        
        # Drawing areas (lines and shapes)
        if analysis.drawing_count > 0:
            # Add some line-like structures
            for i in range(min(analysis.drawing_count, 10)):  # Max 10 lines
                x1 = np.random.randint(0, width//2)
                y1 = np.random.randint(0, height//2)
                x2 = np.random.randint(width//2, width)
                y2 = np.random.randint(height//2, height)
                thickness = np.random.randint(1, 3)
                cv2.line(image, (x1, y1), (x2, y2), (50, 50, 50), thickness)
        
        # Table-like structures (grid)
        if analysis.table_candidate_score > 0.3:
            # Add grid lines
            rows, cols = 4, 4
            row_height = height // rows
            col_width = width // cols
            
            # Horizontal lines
            for r in range(1, rows):
                y = r * row_height
                cv2.line(image, (0, y), (width, y), (100, 100, 100), 1)
            
            # Vertical lines
            for c in range(1, cols):
                x = c * col_width
                cv2.line(image, (x, 0), (x, height), (100, 100, 100), 1)
        
        return image

    async def get_analysis_results(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve analysis results for a document.

        Args:
            document_id: Document ID

        Returns:
            Dictionary containing analysis results

        Raises:
            ValueError: If document not found
        """
        async with get_async_session_local()() as session:
            # Get document
            doc_repo = DocumentRepository(session)
            document = await doc_repo.get_by_id(document_id)
            if not document:
                raise ValueError(f"Document with ID {document_id} not found")

            # Get page analyses
            from app.models.document import DocumentPage
            stmt = select(DocumentPage).where(
                DocumentPage.document_id == document_id
            ).order_by(DocumentPage.page_number)
            result = await session.execute(stmt)
            page_records = result.scalars().all()

            # Convert to response format
            pages = []
            for page in page_records:
                pages.append({
                    "page_number": page.page_number,
                    "classification": page.classification,
                    # Stored as a string (see _persist_page_analyses /
                    # the analyze() insert above) because the DB column
                    # is VARCHAR(20); cast back to float so API
                    # consumers get a number, not a string.
                    "classification_confidence": float(page.classification_confidence) if page.classification_confidence is not None else None,
                    "classification_reason": page.classification_reason,
                    "text_length": page.text_length,
                    "image_count": page.image_count,
                    "drawing_count": page.drawing_count,
                    "analysis_metadata": page.analysis_metadata
                })

            return {
                "document_id": document_id,
                "status": document.status,
                "page_count": len(pages),
                "pages": pages
            }


# Convenience functions for external use
async def analyze_document(document_id: str, force_reanalysis: bool = False) -> Dict[str, Any]:
    """
    Convenience function to analyze a document.

    Args:
        document_id: UUID of the document to analyze
        force_reanalysis: If True, re-analyze even if already analyzed

    Returns:
        Dictionary containing analysis summary
    """
    service = AnalysisService()
    return await service.analyze_document(document_id, force_reanalysis)


async def get_analysis_results(document_id: str) -> Dict[str, Any]:
    """
    Convenience function to get analysis results for a document.

    Args:
        document_id: Document ID

    Returns:
        Dictionary containing analysis results
    """
    service = AnalysisService()
    return await service.get_analysis_results(document_id)
