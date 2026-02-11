# VisionQuery

VisionQuery is a high-performance visual discovery tool designed to query massive image datasets using natural language. By leveraging Modal for serverless GPU orchestration and LanceDB for multi-modal vector indexing, the goal is to enable sub-second retrieval across millions of assets.

**Status: Work in Progress** Current build supports high-performance indexing for **locally saved videos** and **CIFAR-10 images**. Support for direct **S3 video streaming** is under development.

### 1. Setup Environment
Ensure you have Python 3.10+ and the Modal CLI installed.
```bash
pip install -r requirements.txt
modal auth
```

### 2. Deploy the VLM Worker
Deploy the Moondream2 model to Modal's cloud GPU infrastructure:
```bash
modal deploy src/vlm_worker.py
```

### 3. Run the Pipeline
```bash
python main.py
```

### 4. Search the Data
Run the search script to query your indexed frames via natural language:
```bash
python search.py --query "a blue truck"
```

### Next Steps
[ ] S3 Integration: Direct ingestion from AWS S3 buckets.

[ ] Visual Interface: A web-based UI to browse and filter indexed frames visually.

[ ] LoRA Fine-tuning: Implementing Low-Rank Adaptation to enforce specific captioning styles for niche domains.
