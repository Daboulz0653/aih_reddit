import json

INPUT_FILE = "../raw_data/combined_corpus.ndjson"
CHAR_LIMIT = 20000

count_over_limit = 0
total_rows = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        total_rows += 1
        obj = json.loads(line)
        
        # Change "text" to the field you want to measure
        text = obj.get("body", "")
        
        if len(text) > CHAR_LIMIT:
            count_over_limit += 1

print(f"Total rows: {total_rows}")
print(f"Rows with > {CHAR_LIMIT} characters: {count_over_limit}")

