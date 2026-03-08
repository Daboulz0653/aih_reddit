import json
from tqdm import tqdm

INPUT_FILE = "big_corpus_final_cleaned_with_deps.ndjson"
LIMIT = 20  # How many examples you want to see

def find_missing_inference(file_path, limit=20):
    missing_count = 0
    found_examples = []

    print(f"Searching for entries with no inferred model in {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                entry = json.loads(line)
                inferred = entry.get("model_inferred_temporal")
                
                # Check if it's None, an empty list, or an empty string
                if not inferred or (isinstance(inferred, list) and len(inferred) == 0):
                    missing_count += 1
                    
                    if len(found_examples) < limit:
                        # Store the ID and a snippet of the text for context
                        found_examples.append({
                            "id": entry.get("id"),
                            "date": entry.get("created_utc"), # or whatever your date field is
                            "text_snippet": entry.get("text", "")[:100] + "..."
                        })
            except Exception as e:
                continue

    print(f"\nTotal entries missing inference: {missing_count}")
    print(f"--- Top {len(found_examples)} Examples ---")
    for ex in found_examples:
        print(f"ID: {ex['id']} | Date: {ex['date']}")
        print(f"Snippet: {ex['text_snippet']}\n")

if __name__ == "__main__":
    find_missing_inference(INPUT_FILE, LIMIT)