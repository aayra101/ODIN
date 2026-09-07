from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional
from datetime import datetime
import re


class DocumentBase(BaseModel):
    original_filename: str = Field(..., max_length=255)
    content_type: str = Field(..., max_length=100)
    file_size: int = Field(..., gt=0)
    checksum_sha256: str = Field(..., min_length=64, max_length=64)
    source_type: str = Field(..., max_length=50)
    source_identifier: Optional[str] = Field(None, max_length=255)
    subsidiary: Optional[str] = Field(None, max_length=100)
    document_timestamp: Optional[datetime] = None
    storage_reference: str
    status: str = Field(default="ingested", max_length=50)

    @field_validator("checksum_sha256")
    @classmethod
    def validate_checksum(cls, v):
        if not re.match(r"^[a-f0-9]{64}$", v.lower()):
            raise ValueError("Invalid SHA-256 checksum format")
        return v.lower()


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    ingested_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_str(cls, v):
        # `Document.id` is a native PostgreSQL UUID column, so SQLAlchemy
        # hands back a uuid.UUID object here, not a string. Pydantic v2
        # doesn't auto-coerce that for a plain `str` field, so without this
        # `DocumentResponse.from_orm(document)` raises a validation error
        # ("Input should be a valid string ... input_type=UUID").
        return str(v)


class IngestionResult(BaseModel):
    document_id: str
    original_filename: str
    content_type: str
    file_size: int
    checksum_sha256: str
    source_type: str
    source_identifier: Optional[str] = None
    subsidiary: Optional[str] = None
    ingested_at: datetime
    document_timestamp: Optional[datetime] = None
    storage_reference: str
    status: str
    is_duplicate: bool
    message: str
