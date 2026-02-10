import lancedb
import pandas as pd

def search_videos(query: str):
    # 1. Connect to the existing database
    db = lancedb.connect("data/vision_db")
    
    # 2. Open the table
    try:
        table = db.open_table("video_metadata")
    except Exception:
        print("❌ Table not found. Did you run main.py first?")
        return

    # 3. Perform a keyword search on the 'caption' column
    # We use a simple filter here; LanceDB also supports vector search!
    results = table.to_pandas()
    
    # Filter for the keyword (case-insensitive)
    matches = results[results['caption'].str.contains(query, case=False, na=False)]

    if not matches.empty:
        print(f"\n🔍 Found {len(matches)} matches for: '{query}'")
        print("-" * 50)
        for _, row in matches.iterrows():
            print(f"🎬 Video: {row['uri']}")
            print(f"🤖 AI Description: {row['caption']}")
            print("-" * 50)
    else:
        print(f"🤷 No videos found matching '{query}'.")

if __name__ == "__main__":
    search_term = input("What are you looking for? (e.g., 'city', 'water', 'leaf'): ")
    search_videos(search_term)