import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
import csv
import nltk
from itertools import chain, groupby
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

#change these
# corpus_type = "combined_corpus" or "reddit_mini_corpus"
# ONE_HOP = true or false (if you want direct dependencies only)
corpus_type = "big_corpus"
ONE_HOP = False

dep_type = "onehop" if ONE_HOP else "direct"

INPUT_FILE = f"{corpus_type}_final_cleaned_with_deps.ndjson"
OUTPUT_FILE_SUMMARY = f"deps/{corpus_type}/{dep_type}/{corpus_type}_{dep_type}_cleaned_deps_mostfreqpos.ndjson"
OUTPUT_FILE_COMPLETE = f"deps/{corpus_type}/{dep_type}/{corpus_type}_{dep_type}_cleaned_deps_allposbreakdown.ndjson"
# OUTPUT_FILE_TOP = f"direct_deps/{corpus_type}/{dep_type}/{corpus_type}_{dep_type}_deps_top50_nostopwords_ver2_final6.ndjson"

# STOPWORDS = set(stopwords.words('english'))

NUM_PROCESSES = cpu_count() - 1
CHUNK_SIZE = 10000



logger = logging.getLogger()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

S = r"[- ]?"
MODEL_PATTERNS = {
    # 1. Specific Versions First (to prevent GPT-4 from 'eating' GPT-4o)
    "gpt-4o": re.compile(rf"\b(?:chat{S})?gpt{S}4o\S*", re.IGNORECASE),
    "gpt-4.5": re.compile(rf"\b(?:chat{S})?gpt{S}4\.5\S*", re.IGNORECASE),
    "gpt-4.1": re.compile(rf"\b(?:chat{S})?gpt{S}4\.1\S*", re.IGNORECASE),
    "gpt-4-turbo": re.compile(rf"\b(?:chat{S})?gpt{S}4{S}turbo\S*", re.IGNORECASE),

    # 2. GPT-4 (The Negative Lookahead ensures it ignores 4o, 4.5, etc.)
    "gpt-4": re.compile(rf"\b(?:chat{S})?gpt{S}4(?![o\.0-9]|{S}turbo)\S*", re.IGNORECASE),

    # 3. Other Versions
    "gpt-3.5": re.compile(rf"\b(?:chat{S})?gpt{S}3\.5\S*", re.IGNORECASE),
    "gpt-5": re.compile(rf"\b(?:chat{S})?gpt{S}5\S*", re.IGNORECASE),

    # 4. Generics Last
    "chatgpt": re.compile(r"\bchat[\s-]?gpt\S*", re.IGNORECASE),
    "gpt": re.compile(r"\bgpt\S*", re.IGNORECASE),
}

EXTRA_PATTERNS = {
    "model": re.compile(r"^gpt$", re.IGNORECASE),
    "tool": re.compile(r"^tool$", re.IGNORECASE),
    "agent": re.compile(r"^agent$", re.IGNORECASE),
    "bot": re.compile(r"^bot$", re.IGNORECASE),
    "assistant": re.compile(r"^assistant$", re.IGNORECASE),
    "chatbot": re.compile(r"^chatbot$", re.IGNORECASE),
    "LLM": re.compile(r"^LLM$", re.IGNORECASE)

}



def contains_model(sent_tree): 

    found = {}
    for token in sent_tree:
        text = token.get("text")
        token_id = token.get("id")
        for canonical_name, pattern in MODEL_PATTERNS.items():
            if pattern.search(text):
                found[token_id] = canonical_name
                break
    return found

def extract_deps(tree, models_found, deps, model_inferred=None):
    all_ids = set(models_found.keys())
    for model_id, model_name in models_found.items():  
        # if model_inferred and len(model_inferred) > 0:
        #     model_name = model_inferred[0]      

        entry = tree[model_id-1]

        #getting head --- what models points to
        head = tree[entry.get("head_idx") - 1] if entry.get("deprel") != "root" else None
        if head and head.get("id") not in all_ids:
            head_text = head.get("text")
            head_lemma = head.get("lemma")
            head_upos = head.get("upos")

            # deps[head_text][head_upos] += 1
            deps[head_lemma]["counts"][head_upos] += 1
            deps[head_lemma]["models"][model_name] += 1

            #if ONE_HOP == TRUE, gets head/other children of head node 
            if ONE_HOP:
                grandparent = tree[head.get("head_idx") -1] if head.get("deprel") != "root" else None

                if grandparent and grandparent.get("id") not in all_ids:
                    grandparent_text = grandparent.get("text").lower()
                    grandparent_lemma = grandparent.get("lemma").lower()
                    grandparent_upos = grandparent.get("upos")

                    # deps[grandparent_text][grandparent_upos] += 1

                    deps[grandparent_lemma]["counts"][grandparent_upos] += 1
                    deps[grandparent_lemma]["models"][model_name] += 1


                siblings = [c for c in head.get("children", []) if c != model_id]
                for sibling_id in siblings:
                    if sibling_id in all_ids:
                        continue
                    sibling = tree[sibling_id - 1]

                    sibling_text = sibling.get("text").lower()
                    sibling_lemma = sibling.get("lemma").lower()
                    sibling_upos = sibling.get("upos")


                    # deps[sibling_text][sibling_upos] += 1


                    deps[sibling_lemma]["counts"][sibling_upos] += 1
                    deps[sibling_lemma]["models"][model_name] += 1



        #getting children --- what points to model
        children = entry.get("children", [])

        for child_id in children:
            if child_id in all_ids:
                continue
            child = tree[child_id - 1]
            

            child_text = child.get("text").lower()
            child_lemma = child.get("lemma").lower()
            child_upos = child.get("upos")

            # deps[child_text][child_upos] += 1

            deps[child_lemma]["counts"][child_upos] += 1
            deps[child_lemma]["models"][model_name] += 1


            #if ONE_HOP == TRUE, get children of children node (head redundant)
            if ONE_HOP:
                grandchildren = child.get("children", [])

                for grandchild_id in grandchildren:
                    if grandchild_id in all_ids:
                        continue
                    grandchild = tree[grandchild_id - 1]

                    grandchild_text = grandchild.get("text").lower()
                    grandchild_lemma = grandchild.get("lemma").lower()
                    grandchild_upos = grandchild.get("upos")


                    # deps[grandchild_text][grandchild_upos] += 1


                    deps[grandchild_lemma]["counts"][grandchild_upos] += 1
                    deps[grandchild_lemma]["models"][model_name] += 1

    
                


def process_entry(line, chunk_deps):
    try:
        entry = json.loads(line)
        model_inferred = entry.get("model_inferred_temporal")

        dependency_parse = entry.get("dependency_parse")
        if not dependency_parse:
            logger.warning(f"No dependency_parse: {entry.get('id')}")
            return
        
        full_tree = dependency_parse.get("full_tree")
        if not full_tree:
            # logger.warning(f"No full_tree: {entry.get('id')}")
            return entry.get("id")


        full_tree_sorted = sorted(full_tree, key=lambda x: x.get("sent_id"))
        for s_id, tokens in groupby(full_tree_sorted, key=lambda x: x.get("sent_id")):
            sent_tree = list(tokens)

            #gets model mentions in text
            models_found = contains_model(sent_tree)
            
            if models_found:
                extract_deps(sent_tree, models_found, chunk_deps, model_inferred)

        return None
    except Exception as e:
        logger.warning(f"Error processing line: {e}")
        return None
        
def make_dep_entry():
    return {"counts": Counter(), "models": Counter()}

def process_chunk(chunk):
    chunk_deps = defaultdict(make_dep_entry)
    missing_tree_ids = []

    entries_processed = 0
    
    for line in chunk:
        failed_id= process_entry(line, chunk_deps)
        entries_processed += 1
        if failed_id:
            missing_tree_ids.append(failed_id)
    
    return chunk_deps, missing_tree_ids

def read_in_chunks(file_path, chunk_size):
    chunk = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main():
    # deps = defaultdict(lambda : {"counts": Counter(), "models": Counter()})
    #equivalent to above, but picklable and works with Pool 
    deps = defaultdict(make_dep_entry)
    all_missing_trees = []
    with Pool(NUM_PROCESSES) as pool:
            chunks = list(read_in_chunks(INPUT_FILE, CHUNK_SIZE))
            with tqdm(total=len(chunks), desc="processing chunks", unit="chunk") as progress_bar:
                for chunk_deps, missing_trees in pool.imap(process_chunk, chunks):
                    all_missing_trees.extend(missing_trees)
                    for word, counters in chunk_deps.items():
                        upos_counter, models_counter = counters["counts"], counters["models"]
                        deps[word]["counts"].update(upos_counter)
                        deps[word]["models"].update(models_counter)
                    progress_bar.update(1)

    #wtf
    if all_missing_trees:
        with open("missing_trees.txt", "w", encoding="utf-8") as f:
            for mid in all_missing_trees:
                f.write(f"{mid}\n")
        print(f"# of entries missing trees: {len(all_missing_trees)}")

    deps = {word: { "counts" :  counters["counts"], "models": counters["models"]}  for word, counters in deps.items() 
        if counters["counts"].most_common(1)[0][0] != "PUNCT"}
    # deps = {word: upos_counter for word, upos_counter in deps.items() 
    # if upos_counter.most_common(1)[0][0] != "PUNCT"}
    

    #------------------------- WRITING ALL WORDS ------------------------
    with open(OUTPUT_FILE_COMPLETE, "w", newline="", encoding="utf-8") as f:
        for word, counters in deps.items():
            upos_counter, models_counter = counters["counts"], counters["models"]
            row = {"word": word, "counts": dict(upos_counter), "models": dict(models_counter)}
            f.write(json.dumps(row) + "\n")
    
    #------------------------- WRITING MAX ------------------------
    with open(OUTPUT_FILE_SUMMARY, "w", newline="", encoding="utf-8") as f:
        for word, counters in deps.items():
            upos_counter, models_counter = counters["counts"], counters["models"]
            majority_upos, majority_count = upos_counter.most_common(1)[0]
            majority_model, majority_model_count = models_counter.most_common(1)[0]
            row = {"word": word,
                   "total_count" : sum(upos_counter.values()),
                   "majority_upos": majority_upos, 
                   "majority_count" : majority_count,
                   "majority_model": majority_model,
                   "majority_model_count": majority_model_count,
                   "model_freqs": models_counter
                   }
            f.write(json.dumps(row) + "\n")
    print(f"Done.")
            
              

if __name__ == "__main__":
    main()
