import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import chain, groupby

#change these
# corpus_type = "combined_corpus" or "reddit_mini_corpus"
# ONE_HOP = true or false (if you want direct dependencies only)
# INFERRED = false or true if you want to use the inferred model 
corpus_type = "big_corpus"
ONE_HOP = False
INFERRED = False

dep_type = "onehop" if ONE_HOP else "direct"
inferred_type = "_inferred" if INFERRED else ""

INPUT_FILE = f"{corpus_type}_final_cleaned_with_deps_newregex5.ndjson"
OUTPUT_FILE_SUMMARY = f"deps/{corpus_type}/{dep_type}/{corpus_type}_{dep_type}_cleaned_deps_mostfreqpos{inferred_type}.ndjson"
OUTPUT_FILE_COMPLETE = f"deps/{corpus_type}/{dep_type}/{corpus_type}_{dep_type}_cleaned_deps_allposbreakdown{inferred_type}.ndjson"

NUM_PROCESSES = cpu_count() - 1
CHUNK_SIZE = 10000


logger = logging.getLogger()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATTERNS = {
    "gpt-4o": re.compile(r"^(?:CHAT)?GPT4O\S*"),
    "gpt-4": re.compile(r"^(?:CHAT)?GPT4\S*"),
    "gpt-3.5": re.compile(r"^(?:CHAT)?GPT3\S*"),
    "gpt-5": re.compile(r"^(?:CHAT)?GPT5\S*"),
    "chatgpt": re.compile(r"^(?:CHAT)?GPT\S*"),
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



def contains_model(sent_tree, model_match_log): 

    found = {}
    for token in sent_tree:
        text = token.get("text")
        token_id = token.get("id")
        for canonical_name, pattern in MODEL_PATTERNS.items():
            if pattern.search(text):
                found[token_id] = canonical_name
                model_match_log[canonical_name][text] += 1
                break
    return found

def extract_deps(tree, models_found, deps, model_inferred=None):
    all_ids = set(models_found.keys())
    for model_id, model_name in models_found.items():

        if INFERRED and model_inferred and len(model_inferred) > 0:
            model_name = model_inferred[0]      

        entry = tree[model_id-1]

        #getting head --- what models points to
        head = tree[entry.get("head_id") - 1] if entry.get("deprel") != "root" else None

        if head and head.get("id") not in all_ids:
            head_lemma = head.get("lemma")
            head_upos = head.get("upos")

            deps[head_lemma]["counts"][head_upos] += 1
            deps[head_lemma]["models"][model_name] += 1

            #if ONE_HOP == TRUE, gets head/other children of head node 
            if ONE_HOP:
                grandparent = tree[head.get("head_id") -1] if head.get("deprel") != "root" else None

                if grandparent and grandparent.get("id") not in all_ids:
                    grandparent_lemma = grandparent.get("lemma").lower()
                    grandparent_upos = grandparent.get("upos")

                    deps[grandparent_lemma]["counts"][grandparent_upos] += 1
                    deps[grandparent_lemma]["models"][model_name] += 1


                siblings = [c for c in head.get("children", []) if c != model_id]
                for sibling_id in siblings:
                    if sibling_id in all_ids:
                        continue
                    sibling = tree[sibling_id - 1]

                    sibling_lemma = sibling.get("lemma").lower()
                    sibling_upos = sibling.get("upos")


                    deps[sibling_lemma]["counts"][sibling_upos] += 1
                    deps[sibling_lemma]["models"][model_name] += 1



        #getting children --- what points to model
        children = entry.get("children", [])

        for child_id in children:
            if child_id in all_ids:
                continue
            child = tree[child_id - 1]
            

            child_lemma = child.get("lemma").lower()
            child_upos = child.get("upos")

            deps[child_lemma]["counts"][child_upos] += 1
            deps[child_lemma]["models"][model_name] += 1


            #if ONE_HOP == TRUE, get children of children node (head redundant)
            if ONE_HOP:
                grandchildren = child.get("children", [])

                for grandchild_id in grandchildren:
                    if grandchild_id in all_ids:
                        continue
                    grandchild = tree[grandchild_id - 1]

                    grandchild_lemma = grandchild.get("lemma").lower()
                    grandchild_upos = grandchild.get("upos")


                    deps[grandchild_lemma]["counts"][grandchild_upos] += 1
                    deps[grandchild_lemma]["models"][model_name] += 1

    
                


def process_entry(line, chunk_deps, model_match_log):
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
        for _, tokens in groupby(full_tree_sorted, key=lambda x: x.get("sent_id")):
            sent_tree = list(tokens)

            #gets model mentions in text
            models_found = contains_model(sent_tree, model_match_log)
            
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
    model_match_log = defaultdict(Counter)
    missing_tree_ids = []

    entries_processed = 0
    
    for line in chunk:
        failed_id= process_entry(line, chunk_deps, model_match_log)
        entries_processed += 1
        if failed_id:
            missing_tree_ids.append(failed_id)
    
    return chunk_deps, model_match_log, missing_tree_ids


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
    all_matches = defaultdict(Counter)
    all_missing_trees = []
    with Pool(NUM_PROCESSES) as pool:
            chunks = list(read_in_chunks(INPUT_FILE, CHUNK_SIZE))

            with tqdm(total=len(chunks), desc="processing chunks", unit="chunk") as progress_bar:
                
                for chunk_deps, model_match_log, missing_trees in pool.imap(process_chunk, chunks):

                    for canonical, counts in model_match_log.items():
                        all_matches[canonical].update(counts)

                    all_missing_trees.extend(missing_trees)

                    for word, counters in chunk_deps.items():
                        upos_counter, models_counter = counters["counts"], counters["models"]
                        deps[word]["counts"].update(upos_counter)
                        deps[word]["models"].update(models_counter)
                    progress_bar.update(1)


    deps = {word: { "counts" :  counters["counts"], "models": counters["models"]}  for word, counters in deps.items() 
        if counters["counts"].most_common(1)[0][0] != "PUNCT"}
    

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

    #------------------- ENTRIES WITH MISSING TREES ------------------
    if all_missing_trees:
        with open("missing_trees.txt", "w", encoding="utf-8") as f:
            for mid in all_missing_trees:
                f.write(f"{mid}\n")
        print(f"# of entries missing trees: {len(all_missing_trees)}")

    #--------------------- WRITING MODEL MATCHES ---------------------
    with open(f"model_match_log.ndjson", "w", encoding="utf-8") as f:
        for canonical, counts in all_matches.items():
            row = {"model": canonical, "matches": dict(counts)}
            f.write(json.dumps(row) + "\n")

    
    total_words = len(deps)
    total_count = sum(sum(counters["counts"].values()) for counters in deps.values())
    total_mentions = sum(sum(counts.values()) for counts in all_matches.values())
    print(f"Unique words in deps: {total_words}, Total word occurrences: {total_count}")
    print(f"Number of model mentions detected: {total_mentions}")
    print(f"Done.")            
              

if __name__ == "__main__":
    main()
