from pydantic import BaseModel, HttpUrl
from typing import Optional, Union
from pathlib import Path


class VideoSource(BaseModel):
    uri: Union[HttpUrl, Path, str]
    prompt: Optional[str] = None
    job_id: Optional[str] = None
    source_type: str  # "local" or "remote"
