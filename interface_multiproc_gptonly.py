import stanza
import json
# import pandas as pd
import logging
import sys
from tqdm import tqdm
from typing import Dict
import torch
logger = logging.getLogger()
from itertools import islice
import gc
from multiprocessing import Process, Queue
import os
import multiprocessing
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import re
import queue


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.setrecursionlimit(10000)  # Add this line


INPUT = "raw_data/big_corpus_final.ndjson"
OUTPUT = "entweetewt/big_corpus/big_corpus_final_cleaned_with_deps_newregex5.ndjson"
IDFILE = "processed_ids.txt"
SKIPPEDFILE = "skipped_entries5.ndjson"

TOTAL_ENTRIES = 1098440
BATCH_SIZE = 16
MAX_CHARS = 8000
DO_NOT_PROCESS = 20000


#can be set to NONE, GPT or EXTRA
KEYWORD = "GPT"
SUFFIX = r"(?:[-\s]?(?:turbo|mini|nano|preview|vision|audio|realtime|instruct))*"

VERSION = r"(?:3(?:\.5)?|4(?:\.1|\.5|o)?|5)"

#END = r"(?=$|\s|[.,!?;:](?=\s|$))"
END = r"(?:'s)?(?=$|\s|[.,!?;:](?=\s|$))"

MODEL_PATTERNS = {
    "gpt-4o": re.compile(rf"\bgpt[-\s]?4o{SUFFIX}{END}", re.IGNORECASE),
    "gpt-4.5": re.compile(rf"\bgpt[-\s]?4\.5{SUFFIX}{END}", re.IGNORECASE),
    "gpt-4.1": re.compile(rf"\bgpt[-\s]?4\.1{SUFFIX}{END}", re.IGNORECASE),
    "gpt-4": re.compile(rf"\bgpt[-\s]?4{SUFFIX}{END}", re.IGNORECASE),
    "gpt-3.5": re.compile(rf"\bgpt[-\s]?3\.?5{SUFFIX}{END}", re.IGNORECASE),
    "gpt-5": re.compile(rf"\bgpt[-\s]?5{SUFFIX}{END}", re.IGNORECASE),

    "chatgpt": re.compile(
        rf"\bchat[-\s]?gpt(?:[-\s]?{VERSION})?{SUFFIX}{END}",
        re.IGNORECASE
    ),

    "gpt": re.compile(
        rf"\bgpt(?:[-\s]?{VERSION})?{SUFFIX}{END}",
        re.IGNORECASE
    ),
}
#SUFFIX = r"(?:[-\s]?(?:turbo|mini|nano|preview|vision|audio|realtime|instruct))*"
#END = r"(?![a-z0-9\.])"
#END = r"(?=[\s]*[\.!?]?[\s]*$|[\s])"

#
#MODEL_PATTERNS = {
#    "gpt-4o": re.compile(rf"\bgpt[-\s]?4o{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4.5": re.compile(rf"\bgpt[-\s]?4\.5{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4.1": re.compile(rf"\bgpt[-\s]?4\.1{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4": re.compile(rf"\bgpt[-\s]?4{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-3.5": re.compile(rf"\bgpt[-\s]?3\.?5{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-5": re.compile(rf"\bgpt[-\s]?5{SUFFIX}{END}", re.IGNORECASE),
#    "chatgpt": re.compile(rf"\bchat[-\s]?gpt{SUFFIX}{END}", re.IGNORECASE),
#    "gpt": re.compile(rf"\bgpt{SUFFIX}{END}", re.IGNORECASE),
#}

#MODEL_PATTERNS = {
#    "gpt-4o": re.compile(rf"\bgpt[-\s]?4o{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4.5": re.compile(rf"\bgpt[-\s]?4\.5{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4.1": re.compile(rf"\bgpt[-\s]?4\.1{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4": re.compile(rf"\bgpt[-\s]?4{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-3.5": re.compile(rf"\bgpt[-\s]?3\.?5{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-5": re.compile(rf"\bgpt[-\s]?5{SUFFIX}{END}", re.IGNORECASE),
#    "chatgpt": re.compile(rf"\bchat[-\s]?gpt{SUFFIX}{END}", re.IGNORECASE),
 #   "gpt": re.compile(rf"\bgpt{SUFFIX}{END}", re.IGNORECASE),
#    "chatgpt": re.compile(
#        rf"\bchat[-\s]?gpt(?:[-\s]?([345]|4\.5|4\.1|4o|5))?{SUFFIX}{END}",
#        re.IGNORECASE
#    ),

    # Generic GPT catch-all (with optional version)
#    "gpt": re.compile(
#        rf"\bgpt(?:[-\s]?([345]|4\.5|4\.1|4o|5))?{SUFFIX}{END}",
#        re.IGNORECASE
#    ),
#}

#S = r"[- ]?"
#MODEL_PATTERNS = {
#    # 1. Specific Versions First (to prevent GPT-4 from 'eating' GPT-4o)
#   # "gpt-4o": re.compile(rf"\b(?:chat{S})?gpt{S}4o\S*", re.IGNORECASE),
#    "gpt-4o": re.compile(rf"\b(?:chat{S})?gpt{S}4{S}(?:o(?:{S}mini)?|omni)\b", re.IGNORECASE),
#    "gpt-4.5": re.compile(rf"\b(?:chat{S})?gpt{S}4\.5\S*", re.IGNORECASE),
#    "gpt-4.1": re.compile(rf"\b(?:chat{S})?gpt{S}4\.1\S*", re.IGNORECASE),
#    "gpt-4-turbo": re.compile(rf"\b(?:chat{S})?gpt{S}4{S}turbo\S*", re.IGNORECASE),
#
#    # 2. GPT-4 (The Negative Lookahead ensures it ignores 4o, 4.5, etc.)
#    "gpt-4": re.compile(rf"\b(?:chat{S})?gpt{S}4(?![o\.0-9]|{S}turbo|{S}omni)\b", re.IGNORECASE),
#    #"gpt-4": re.compile(rf"\b(?:chat{S})?gpt{S}4(?![o\.0-9]|{S}turbo)\S*", re.IGNORECASE),
#
#    # 3. Other Versions
#    "gpt-3.5": re.compile(rf"\b(?:chat{S})?gpt{S}3\.5\S*", re.IGNORECASE),
#    "gpt-5": re.compile(rf"\b(?:chat{S})?gpt{S}5\S*", re.IGNORECASE),
#
#    # 4. Generics Last
#    "chatgpt": re.compile(r"\bchat[\s-]?gpt\S*", re.IGNORECASE),
#    "gpt": re.compile(r"\bgpt\S*", re.IGNORECASE),
#}

EXTRA_PATTERNS = {
    "model": re.compile(r"^gpt$", re.IGNORECASE),
    "tool": re.compile(r"^tool$", re.IGNORECASE),
    "agent": re.compile(r"^agent$", re.IGNORECASE),
    "bot": re.compile(r"^bot$", re.IGNORECASE),
    "assistant": re.compile(r"^assistant$", re.IGNORECASE),
    "chatbot": re.compile(r"^chatbot$", re.IGNORECASE),
    "LLM": re.compile(r"^LLM$", re.IGNORECASE)

}

MODELS_TO_EXCLUDE = {
    "gemini": re.compile(r"^gemini$", re.IGNORECASE),
    "claude": re.compile(r"^claude$", re.IGNORECASE),
    "grok": re.compile(r"^grok$", re.IGNORECASE),
    "copilot": re.compile(r"^copilot$", re.IGNORECASE),
    "perplexity": re.compile(r"^perplexity$", re.IGNORECASE),
    "deepseek": re.compile(r"^deepseek$", re.IGNORECASE)

}


def configure_model():
    config = {
          'processors': 'tokenize,lemma,pos,depparse',
          'lang': 'en',
          'use_gpu': True,
          'device': 'cuda',
          
          'tokenize_pretokenized': False, #input not tokenzied 
          'tokenize_no_ssplit': True, #input already split with \n\n
          'tokenize_model_path': './TweebankNLP/twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt',

          'lemma_model_path': './TweebankNLP/twitter-stanza/saved_models/lemma/en_tweetewt_lemmatizer.pt',
          "pos_model_path": './TweebankNLP/twitter-stanza/saved_models/pos/en_tweetewt_tagger.pt',
          "depparse_model_path": './TweebankNLP/twitter-stanza/saved_models/depparse/en_tweetewt_parser.pt',
}
    # Initialize the pipeline using a configuration dict
    stanza.download("en")
    nlp = stanza.Pipeline(**config)

    return nlp


def serialize(doc):
    tree = []

    for sent_id, sentence in enumerate(doc.sentences):
        for word in sentence.words: #using words instead of tokens ---spacy does not take into account MWTs (?)
            word_info  = {
                'text': word.text,
                'lemma': word.lemma,
                'upos': word.upos,
                'xpos': word.xpos,
                'deprel': word.deprel,  # dependency relation,
                'feats':word.feats,
                'head_id': word.head, 
                'id': word.id,  
                'children': [child.id for child in sentence.words if child.head == word.id],  # indices of children
                'sent_id': sent_id
            }
            tree.append(word_info)

    return tree


def gpu_worker(gpu_id, input_queue, output_queue):
    
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        nlp = configure_model()
        tqdm.write(f"Worker {gpu_id} model loaded successfully")
        batch_count = 0
    
        while True:
            try:
                item = input_queue.get(timeout=300)  # 5 min timeout to detect hangs
                
                if item is None:
                    tqdm.write(f"Worker {gpu_id} received stop signal")
                    break
                
                batch_idx, texts = item
                batch_count += 1
                
                if batch_count % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    tqdm.write(f"Worker {gpu_id} periodic cleanup at batch {batch_count}")
                

                try:
                    in_docs = [stanza.Document([], text=d) for d in texts]

                    out_docs = nlp(in_docs)
                    results = []
                    for i, doc in enumerate(out_docs):
                            parsed_data = {
                                "usable_text": doc.text,
                                "full_tree": serialize(doc),
                     #           "triples": compact_structure(doc),
                                "num_tokens": doc.num_tokens,
                                "num_words": doc.num_words,
                                "num_sentences": len(doc.sentences)
                            }
                            results.append(parsed_data)
                        
                    del in_docs, out_docs
                
                except torch.cuda.OutOfMemoryError as e:
                    tqdm.write(f"Worker {gpu_id}: Out Of Memory on batch {batch_idx}: {e}")
                    results = [None] * len(texts)
                    gc.collect()
                    torch.cuda.empty_cache()
                
                except Exception as e:
                    tqdm.write(f"Worker {gpu_id}: Unexpected error on batch {batch_idx}: {e}")
                    tqdm.write(f"Worker {gpu_id}: Traceback:", exc_info=True)
                    results = [None] * len(texts)
                
                output_queue.put((batch_idx, results))
                
            except queue.Empty:
                tqdm.write(f"Worker {gpu_id} timeout waiting for work (5 min)")
                continue
                
    except Exception as e:
        tqdm.write(f"FATAL CRASH ON GPU {gpu_id}: {e}")
    
    finally:
        torch.cuda.empty_cache()
        tqdm.write(f"Worker {gpu_id} shutting down gracefully")



def collect_results(output_queue, pending_batches, next_id):
    while not output_queue.empty():
        results = output_queue.get()

        if isinstance(results, tuple) and results[0] == "ERROR":
            continue

        completed_idx, parsed_results = results
        
        pending_batches[f"res_{completed_idx}"] = parsed_results

        while f"res_{next_id}" in pending_batches:
            res = pending_batches.pop(f"res_{next_id}")
            raw = pending_batches.pop(next_id)
            yield raw, res
            next_id += 1

#writes batch entries to the output file and updates IDFILE and progress bar accordingly
def write_batch(raw, res, io):
    for entry, parsed_data in zip(raw, res):
        
        #text processed, but something went wrong
        if parsed_data is None:
            entry["dependency_parse"] = None
            io['failedfile'].write(json.dumps(entry) + '\n')
            tqdm.write(f"Failed to process entry {entry.get('id')}")

        else:    
            entry["dependency_parse"] = parsed_data
            io['outfile'].write(json.dumps(entry) + '\n')
        
        io['idfile'].write(str(entry.get("id")) + "\n")
        io['ids'].add(str(entry.get("id")))
    

#tokenizes by sentences, find sentences that mention model words specified by KEYWORD
#output --> sentences delineated by \n\n as desired by Stanza pipeline
def model_sentences(sentences, keyword):
    if keyword == "NONE":
        return "\n\n".join(sentences)
    elif keyword not in ["GPT", "EXTRA"]:
        tqdm.write("keyword chosen not allowed, defaulted to gpt only")
    
    matched = [s for s in sentences if any(token in s for token in ["GPT", "CHATGPT"])]

    if keyword == "EXTRA": 
        entry_has_excluded = any(pattern.search(s) for s in sentences for pattern in MODELS_TO_EXCLUDE.values())

        if not entry_has_excluded:
            models = {**MODEL_PATTERNS, **EXTRA_PATTERNS}
            matched = [s for s in sentences if any(pattern.search(s) for pattern in models.values())]
    
    
    return '\n\n'.join(matched)

#standerdizes quotation marks and removes single quotes
def clean(text):
    #weird utf8 chars
    text = re.sub(r'[\u2011\u2012\u2013\u2014\u2015\u2212]', ' ', text)
    text = re.sub(r'[\u2022\u2023\u2026\u2030\u2031]', ' ', text)
    text = re.sub(r'[\u060c\u061b\u061f\u3001\u3002\u300c\u300d\uff01\uff0c\uff1a\uff1b\uff1f]', ' ', text)
    
    #normalizing double and single quotes
    text = re.sub(r"[\'`´‘’‚‛′‵ʻʼʹ]", "'", text)
    text = re.sub(r'["“”„‟«»＂]', '"', text)
    
    #removing other meaningless symbols
    text = re.sub(r'[*#_~\\/|]', ' ', text)
    
    #normalizng ellipses and repeating characters
    #text = re.sub(r'([.!?,;:()"\'])', r' \1 ', text)
    #text = re.sub(r'([!?,;:()"])', r' \1 ', text)
    #padding brackets, etc... to allow for match
    #text = re.sub(r'([()\[\]{}"\'`])', r' \1 ', text)
    text = re.sub(r'([!?,;:()\[\]{}"])', r' \1 ', text)

    text = re.sub(r"'s", " 's", text)
    text  = " ".join(text.split())
    return text


#normalizes spelling of model names
# gpt 5, gpt-5, GPT5, gpt5 -------> GPT5
# chatgpt, chat Gpt ------> CHATGPT
def normalize_model_names(text):
    for model_name, pattern in MODEL_PATTERNS.items():
        clean_func = lambda m: m.group(0).replace(" ", "").replace("-", "").replace("_", "").upper()
        text = pattern.sub(clean_func, text)
    return text


def main():
    
    #loading in already processed ids
    processed_ids = set()
    with open(IDFILE, "r", encoding='utf-8') as pfile:
        processed_ids = set(line.strip() for line in pfile)
    
    processed_single = 0

    #initalizing workers
    input_queue = Queue(maxsize=20)
    output_queue = Queue()

    workers = []
    for gpu_id in [0, 1]:
        p = Process(target=gpu_worker, args=(gpu_id, input_queue, output_queue))
        p.daemon = True
        p.start()
        workers.append(p)

    tqdm.write("GPU workers started")
        
    with open(INPUT, 'r', encoding='utf-8') as infile, \
        open(OUTPUT, 'a', encoding='utf-8') as outfile, \
        open(IDFILE, "a", encoding='utf-8') as id_file, \
        open(SKIPPEDFILE, "a", encoding='utf-8') as failed_file:
        
        io = {
                'outfile': outfile,
                'failedfile': failed_file,
                'idfile' : id_file,
                'ids': processed_ids
                }

        errors = 0
        skipped = 0
        batch = []
        batch_entries = []
        batch_idx = 0
        pending_batches = {}
        next_batch_to_write = 0



        infile.seek(0)  # reset file ptr to beginning
        with tqdm(total=TOTAL_ENTRIES ,desc="processing", unit="entries", initial=len(processed_ids), position=0, leave=True) as progress_bar:

            for line in infile:
                try:
                    entry = json.loads(line)

                    #if entry already processed, skip
                    entry_id = entry.get("id")                    
                    if entry_id in processed_ids:
                        continue

                    if entry.get("type") == "submission":
                        # combine title and selftext for submissions
                        text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
                    else:  # comment
                        text = entry.get("body", "").strip()

                    #skip empty entries
                    if not text or text.strip() == "":
                        failed_file.write(json.dumps(entry) + '\n')
                        skipped += 1
                        continue
                    

                    #removing some puncts, lowering, and normalizing model name
                    #getting sentences with only models/words specified by KEYWORD, returned as \n\n separated sentences
                    sentences = sent_tokenize(text)
                    sentences = [s.replace('\n', ' ').replace('\t', ' ') for s in sentences]
                    sentences = [normalize_model_names(clean(s.lower())) for s in sentences]
                    entry["cleaned_text"] = ' '.join(sentences)
                    text = model_sentences(sentences, KEYWORD)

                    #shouldn't really do anything if scraping. this is if no matched sentences are returned 
                    if not text.strip():
                        io['failedfile'].write(json.dumps(entry) + '\n')
                        skipped += 1
                        continue

                    
                    #length check to not overload processor with large texts/batches
                    if len(text) >=  DO_NOT_PROCESS:
                        failed_file.write(json.dumps(entry), '\n')
                        skipped += 1 
                        continue
                    elif len(text) > MAX_CHARS: #if text long, put in queue individually
                        input_queue.put((batch_idx, [text]))
                        pending_batches[batch_idx] = [entry]
                        batch_idx += 1
                        processed_single += 1

                        for raw, res in collect_results(output_queue, pending_batches, next_batch_to_write):
                            write_batch(raw, res, io)
                            progress_bar.update(len(res))
                            next_batch_to_write += 1
                        continue
                    
                    #processing normal batch: adding to queue, updating next_batch_to_write
                    batch.append(text)
                    batch_entries.append(entry)


                    if len(batch) >= BATCH_SIZE:
                        input_queue.put((batch_idx, batch))
                        pending_batches[batch_idx] = batch_entries
                        batch_idx += 1

                        #collect_results(output_queue, pending_batches, next_batch_to_write, outfile, processed_file, processed_ids, progress_bar, failed_file)
                        for raw, res in collect_results(output_queue, pending_batches, next_batch_to_write):
                             write_batch(raw, res, io)
                             progress_bar.update(len(res))
                             next_batch_to_write += 1
                        batch = []
                        batch_entries = []
                        
                except json.JSONDecodeError as e:
                    tqdm.write(f"JSON decode error: {e}")
                    errors += 1
                    progress_bar.update(1)
                except Exception as e:
                    tqdm.write(f"error processing entry: {e}")
                    errors += 1
                    progress_bar.update(1)

            #---------------------LEFTOVERS IN BATCH FINAL RUN--------------------
            if batch: #essentially checks if batch is empty
                input_queue.put((batch_idx, batch))
                pending_batches[batch_idx] = batch_entries
                batch_idx += 1
            
            #sending done signal after submitting final batch
            for _ in workers:
                input_queue.put(None)

            total_batches = batch_idx
            while next_batch_to_write < total_batches:
                for raw, res in collect_results(output_queue, pending_batches, next_batch_to_write):
                    write_batch(raw, res, io)
                    progress_bar.update(len(res))
                    next_batch_to_write += 1

    # Wait for workers to finish
    for p in workers:
        p.join()




    tqdm.write(f"errors: {errors}")
    tqdm.write(f"lower bound on skipped: {skipped}")
    tqdm.write(f"total entries processed: {progress_bar.n}")
    tqdm.write(f"output saved to: {OUTPUT}")

if __name__ == "__main__":
    main()

