import lancedb
import modal

def semantic_search(query: str):
    db = lancedb.connect("data/vision_db")
    try:
        table = db.open_table("video_metadata")
    except Exception as e:
        print(f"❌ Actual Error: {e}") 
        return
    
    print(f"☁️ Asking Modal to embed: '{query}'...")
    # Lookup the deployed class by name
    Worker = modal.Cls.from_name("vision-query-moondream", "MoondreamWorker")
    worker = Worker()

    query_vector = worker.embed_text.remote(query)

    results = table.search(query_vector).limit(2).to_pandas()

    print(f"\n🧠 Semantic Results for: '{query}'")
    print(results[['uri', 'caption']])


def search_videos(query: str):
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
    #search_term = input("What are you looking for? (e.g., 'city', 'water', 'leaf'): ")
    #search_videos(search_term)

    term = input("Search by meaning: ")
    semantic_search(term)