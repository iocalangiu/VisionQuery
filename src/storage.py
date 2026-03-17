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
        return pa.schema(
            [
                pa.field(
                    "vector", pa.list_(pa.float32(), 384)
                ),  # For future CLIP embeddings
                pa.field("uri", pa.string()),
                pa.field("caption", pa.string()),
                pa.field("timestamp", pa.string()),
            ]
        )

    def build_index(self, num_partitions: int = 8, num_sub_vectors: int = 16):
        """Builds an IVF-PQ ANN index on the vector column for fast semantic search.

        num_partitions: number of clusters (higher = faster search, needs more data)
        num_sub_vectors: PQ compression chunks (must divide embedding dim 384 evenly)
        Requires at least 256 * num_partitions rows.
        """
        if self.table_name not in self.db.table_names():
            print("❌ No table found. Run the pipeline first.")
            return
        tbl = self.db.open_table(self.table_name)
        row_count = len(tbl.to_pandas())
        min_rows = 256 * num_partitions
        if row_count < min_rows:
            print(
                f"❌ Need at least {min_rows} rows for num_partitions={num_partitions}, have {row_count}."
            )
            return
        print(f"Building IVF-PQ index on {row_count} vectors...")
        tbl.create_index(num_partitions=num_partitions, num_sub_vectors=num_sub_vectors)
        print("✅ Index built.")

    def save_result(self, video_uri: str, caption: str, embedding: list):
        """Saves a single entry to the database."""
        data = [
            {
                "vector": embedding,
                "uri": video_uri,
                "caption": caption,
                "timestamp": datetime.now().isoformat(),
            }
        ]

        if self.table_name in self.db.table_names():
            tbl = self.db.open_table(self.table_name)
            tbl.add(data)
        else:
            self.db.create_table(self.table_name, data=data, schema=self._get_schema())

        print(f"📦 Stored metadata for: {video_uri}")
