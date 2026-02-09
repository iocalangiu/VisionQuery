import lancedb
import pyarrow as pa
from datetime import datetime

class VisionStorage:
    def __init__(self, uri: str = "data/vision_db"):
        self.uri = uri
        self.db = lancedb.connect(self.uri)
        self.table_name = "video_metadata"

    def _get_schema(self):
        """Defines the structure of our data (the 'Contract')"""
        return pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 768)), # For future CLIP embeddings
            pa.field("uri", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("timestamp", pa.string()),
        ])

    def save_result(self, video_uri: str, caption: str):
        """Saves a single entry to the database."""
        data = [{
            "vector": [0.0] * 768, 
            "uri": video_uri,
            "caption": caption,
            "timestamp": datetime.now().isoformat()
        }]
        
        if self.table_name in self.db.table_names():
            tbl = self.db.open_table(self.table_name)
            tbl.add(data)
        else:
            self.db.create_table(self.table_name, data=data, schema=self._get_schema())
        
        print(f"📦 Stored metadata for: {video_uri}")