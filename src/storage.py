import lancedb
import pyarrow as pa
from datetime import datetime
from sentence_transformers import SentenceTransformer

class VisionStorage:
    def __init__(self, uri: str = "data/vision_db"):
        self.uri = uri
        self.db = lancedb.connect(self.uri)
        self.table_name = "video_metadata"
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def _get_schema(self):
        """Defines the structure of our data (the 'Contract')"""
        return pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 384)), # For future CLIP embeddings
            pa.field("uri", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("timestamp", pa.string()),
        ])

    def save_result(self, video_uri: str, caption: str):
        embedding = self.encoder.encode(caption).tolist()
        
        """Saves a single entry to the database."""
        data = [{
            "vector": embedding, 
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