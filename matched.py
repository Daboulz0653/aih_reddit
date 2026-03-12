import json
import re

INPUT_FILE = "entweetewt/big_corpus/big_corpus_final_cleaned_with_deps_newregex5.ndjson"

uppercase_tokens = set()

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        usable_text = entry.get("dependency_parse", {}).get("usable_text", "")
        if not usable_text:
            continue
        tokens = re.findall(r'\b[A-Z][A-Z0-9.]+\b', usable_text)
        uppercase_tokens.update(tokens)

with open("uppercase_tokens6.txt", "w", encoding="utf-8") as f:
    for token in sorted(uppercase_tokens):
        f.write(token + "\n")

print(f"Unique uppercase tokens: {len(uppercase_tokens)}")
