import json
import re
from collections import Counter, defaultdict

# Regex for detecting ChatGPT mentions
pattern_chatgpt = re.compile(r"\bchat[-\s]?gpt(?:[-\s]?([345]|4\.5|4\.1|4o|5))?\b", re.IGNORECASE)
pattern_cgpt = re.compile(
    r"\bcgpt(?:[-\s]?([345]|4\.5|4\.1|4o|5))?\b",
    re.IGNORECASE
)
# Regex for URL and SUBREDDIT placeholders
pattern_url = re.compile(r"_URL_", re.IGNORECASE)
pattern_subreddit = re.compile(r"_SUBREDDIT_", re.IGNORECASE)
pattern_chatgpt_possessive = re.compile(r"\bchat[-\s]?gpt['’]s\b", re.IGNORECASE)

pattern_chatgpt_hyphen = re.compile(
    r"\bchat[-\s]?gpt(?:[-\s]?(?:[345]|4\.5|4\.1|4o|5))?-\w+",
    re.IGNORECASE
)

pattern_chatgpt_symbol = re.compile(
    r"\bchat[-\s]?gpt[+*/=]",
    re.IGNORECASE
)
# Counter for categories
counter = Counter()
# Store some examples for inspection
examples = defaultdict(list)

# Read skipped entries from NDJSON
with open("skipped_entries5.ndjson", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        text = entry.get("body", "")

        if pattern_url.search(text):
            counter["_URL_"] += 1
            if len(examples["_URL_"]) < 2:  # store up to 5 examples
                examples["_URL_"].append(text)
        elif pattern_subreddit.search(text):
            counter["_SUBREDDIT_"] += 1
            if len(examples["_SUBREDDIT_"]) < 2:
                examples["_SUBREDDIT_"].append(text)
        elif pattern_cgpt.search(text):
            counter["cgpt_mentions"] += 1
            if len(examples["cgpt_mentions"]) < 10:
                examples["cgpt_mentions"].append(text)
        elif pattern_chatgpt.search(text):
            counter["chatgpt_mentions"] += 1
            if len(examples["chatgpt_mentions"]) < 10:
                examples["chatgpt_mentions"].append(text)

            if pattern_chatgpt_possessive.search(text):
                counter["chatgpt_possessive"] += 1
                if len(examples["chatgpt_possessive"]) < 10:
                    examples["chatgpt_possessive"].append(text)

            elif pattern_chatgpt_hyphen.search(text):
                counter["chatgpt_hyphen"] += 1
                if len(examples["chatgpt_hyphen"]) < 10:
                    examples["chatgpt_hyphen"].append(text)

            elif pattern_chatgpt_symbol.search(text):
                counter["chatgpt_symbol"] += 1
                if len(examples["chatgpt_symbol"]) < 10:
                    examples["chatgpt_symbol"].append(text)

            else:
                counter["chatgpt_other"] += 1
                if len(examples["chatgpt_other"]) < 10:
                    examples["chatgpt_other"].append(text)
        else:
            counter["other"] += 1
            if len(examples["other"]) < 2:
                examples["other"].append(text)

# Print counts
print("Skipped entries breakdown:")
for k, v in counter.most_common():
    print(f"{k}: {v}")

# Print example entries
print("\nExample entries by category:")
for cat, texts in examples.items():
    print(f"\n=== {cat} ===")
    for t in texts:
        print(f"- {t}")
