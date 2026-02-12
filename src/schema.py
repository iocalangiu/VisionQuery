from pydantic import BaseModel, HttpUrl
from typing import Union, Literal, Optional
from pathlib import Path


class MediaSource(BaseModel):
    uri: Union[HttpUrl, Path, str]
    media_type: Literal["video", "image"]
    source_type: Literal["local", "cifar"]
    label: Optional[int] = None
