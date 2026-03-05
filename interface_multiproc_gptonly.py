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
from nltk.tokenize import sent_tokenize
import re


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
INPUT = "../raw_data/combined_corpus_cleaned_inferred_models.ndjson"
OUTPUT = "../entweetewt/combined_corpus/combined_corpus_cleaned_with_deps_entweetewt_tweettokenizer_gptonly.ndjson"
IDFILE = "../reddit_ids.txt"
SKIPPEDFILE = "../reddit_skipped_ids.txt"
TOTAL_ENTRIES = 870364
BATCH_SIZE = 16
MAX_CHARS = 8000
DO_NOT_PROCESS = 30000

sys.setrecursionlimit(10000)  # Add this line

#can be set to NONE, GPT or EXTRA
KEYWORD = "GPT"
MODEL_PATTERNS = {
    "gpt-3.5": re.compile(r"gpt[-\s]?3\.?5|chat[\s-]?gpt[-\s]?3\.?5", re.IGNORECASE),
    "gpt-4o": re.compile(r"gpt[-\s]?4o|chat[\s-]?gpt[-\s]?4o", re.IGNORECASE),
    "gpt-4": re.compile(r"gpt[-\s]?4(?!o)|chat[\s-]?gpt[-\s]?4(?!o)", re.IGNORECASE),
    "gpt-5": re.compile(r"gpt[-\s]?5(?!o)|chat[\s-]?gpt[-\s]?5(?!o)", re.IGNORECASE),
    "chatgpt": re.compile(r"\bchat[\s-]?gpt\b", re.IGNORECASE),
    "gpt": re.compile(r"^gpt$", re.IGNORECASE)
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

MODELS_TO_EXCLUDE = {
    "gemini": re.compile(r"^gemini$", re.IGNORECASE),
    "claude": re.compile(r"^claude$", re.IGNORECASE),
    "grok": re.compile(r"^grok$", re.IGNORECASE),
    "copilot": re.compile(r"^copilot$", re.IGNORECASE),
    "perplexity": re.compile(r"^perplexity$", re.IGNORECASE),
    "deepseek": re.compile(r"^deepseek$", re.IGNORECASE)

}


def configure_model():
    # config for the `en_tweet` models (models trained only on Tweebank)
    config = {
          'processors': 'tokenize,lemma,pos,depparse',
          'lang': 'en',
          'use_gpu': True,
          'device': 'cuda',
          
          'tokenize_pretokenized': False, # disable tokenization
          'tokenize_model_path': './twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt',
          'lemma_model_path': './twitter-stanza/saved_models/lemma/en_tweetewt_lemmatizer.pt',
          "pos_model_path": './twitter-stanza/saved_models/pos/en_tweetewt_tagger.pt',
          "depparse_model_path": './twitter-stanza/saved_models/depparse/en_tweetewt_parser.pt',
            # "ner_model_path": './twitter-stanza/saved_models/ner/en_tweet_nertagger.pt',
}
    # Initialize the pipeline using a configuration dict
    stanza.download("en")
    nlp = stanza.Pipeline(**config)

    return nlp


def serialize(doc):
    tree = []

    for sentence in doc.sentences:
        for word in sentence.words: #using words instead of tokens ---spacy does not take into account MWTs (?)
            word_info  = {
                'text': word.text,
                'lemma': word.lemma,
                'upos': word.upos,
                'xpos': word.xpos,
                "deprel": word.deprel,  # dependency relation,
                "feats":word.feats,
                "head_idx": word.head, 
                "id": word.id,  
                'children': [child.id for child in sentence.words if child.head == word.id]  # indices of children
            }
            tree.append(word_info)

    return tree

#do I need this for stanza? what is the purpose of this
def compact_structure(doc) -> Dict:
    sentences_triples = []

    for sent in doc.sentences:
        triples = []
        # root_indices = []

        for head, rel, dep in sent.dependencies:
            # if rel.lower() == "root" or dep.head == 0:
            #     root_indices.append(dep.id)
            #     continue

            triples.append({
                "head": head.id,
                "head_text": head.text,
                "rel": rel,
                "dep": dep.id,
                "dep_text": dep.text
            })

        sentences_triples.append(triples)
        # root_indices_per_sentence.append(root_indices)

    return  sentences_triples
#def compact_structure(doc) -> Dict:
#    sentences_triples = []
#    for sent in doc.sentences:
#        triples = []
#        # Build dependencies manually without relying on .parent or sent.dependencies
#        for word in sent.words:
#            if word.head == 0:  # Root node
#                continue
#            
#            # Find the head word by its ID
#            head_word = None
#            for potential_head in sent.words:
#                if potential_head.id == word.head:
#                    head_word = potential_head
#                    break
#            
#            if head_word is not None:
#                triples.append({
#                    "head": head_word.id,
#                    "head_text": head_word.text,
#                    "rel": word.deprel,
#                    "dep": word.id,
#                    "dep_text": word.text
#                })
#        
#        sentences_triples.append(triples)
#    
#    return {"sentences_triples": sentences_triples}

#def process_batch(batch, nlp0, nlp1):
#    results = []
#
#    in_docs = [stanza.Document([], text=d) for d in batch] # Wrap each document with a stanza.Document object
#    
#    mid = len(in_docs) //2
#    with ThreadPoolExecutor(max_workers=2) as executor:
#        future1 = executor.submit(nlp0, in_docs[:mid])
#        future2 = executor.submit(nlp1, in_docs[mid:])
#
#        out1 = future1.result()
#        out2 = future2.result()
#
#    out_docs = out1 + out2
#    for doc in out_docs:
#        parsed_data = {
#           "full_tree": serialize(doc),
#           "compact": compact_structure(doc),
#           "num_tokens" : doc.num_tokens,
#           "num_words" : doc.num_words,
#           "num_sentences": len(doc.sentences)
#        }
#        results.append(parsed_data)
#    
#    del in_docs
#    del out_docs
#
#    return results
#
def check_gpu_memory(gpu_id=0, threshold=0.85):
    """Check GPU memory and return usage percentage"""
    if not torch.cuda.is_available():
        return 0.0
    
    memory_allocated = torch.cuda.memory_allocated(gpu_id)
    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
    usage_pct = memory_allocated / memory_total
    
    if usage_pct > threshold:
        logger.warning(f"GPU {gpu_id} memory: {usage_pct:.1%} (allocated: {memory_allocated/1e9:.2f}GB / {memory_total/1e9:.2f}GB)")
        gc.collect()
        torch.cuda.empty_cache()
        # Check again after cleanup
        memory_allocated = torch.cuda.memory_allocated(gpu_id)
        usage_pct = memory_allocated / memory_total
        logger.info(f"GPU {gpu_id} after cleanup: {usage_pct:.1%}")
    
    return usage_pct

def gpu_worker(gpu_id, input_queue, output_queue):
    
    worker_log_file = f'worker_{gpu_id}.log'
    file_handler = logging.FileHandler(worker_log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    worker_logger = logging.getLogger(f'worker_{gpu_id}')
    worker_logger.addHandler(file_handler)
    worker_logger.addHandler(logging.StreamHandler())  # Also print to console
    worker_logger.setLevel(logging.INFO)


    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        nlp = configure_model()
        worker_logger.info(f"Worker {gpu_id} model loaded successfully")
        batch_count = 0
#        while True:
#            item = input_queue.get()
#            worker_logger.info(f"Worker {gpu_id} received stop signal")
#            if item is None: 
#                break
#
#            batch_idx, texts = item
#            if batch_count % 20 == 0:
#                check_gpu_memory(0, threshold=0.85)
#
#            try:
#                with torch.no_grad():
#                    worker_logger.debug(f"Worker {gpu_id} creating documents for batch {batch_idx}")
#                    in_docs = [stanza.Document([], text=d) for d in texts] # Wrap each document with a stanza.Document object
#                    worker_logger.debug(f"Worker {gpu_id} parsing batch {batch_idx}")
#                    out_docs = nlp(in_docs)
                    
        while True:
            try:
                item = input_queue.get(timeout=300)  # 5 min timeout to detect hangs
                
                if item is None:
                    worker_logger.info(f"Worker {gpu_id} received stop signal")
                    break
                
                batch_idx, texts = item
                batch_count += 1
                
                # Only check memory occasionally
                if batch_count % 20 == 0:
                    check_gpu_memory(0, threshold=0.85)
                
                try:
                    with torch.no_grad():
                        
                        in_docs = [stanza.Document([], text=d) for d in texts]

                        out_docs = nlp(in_docs)
                        results = []
                        for i, doc in enumerate(out_docs):
                            try:
                                parsed_data = {
                                    "usable_text": doc.text,
                                    "full_tree": serialize(doc),
                                    "triples": compact_structure(doc),
                                    "num_tokens": doc.num_tokens,
                                    "num_words": doc.num_words,
                                    "num_sentences": len(doc.sentences)
                                }
                                results.append(parsed_data)
                            except RecursionError:
                                worker_lgger.warning(f"Worker {gpu_id}: RecursionError serializing doc {i} in batch {batch_idx}")
                                results.append(None)
                            except Exception as e:
                                worker_logger.error(f"Worker {gpu_id}: Error serializing doc {i} in batch {batch_idx}: {e}")
                                results.append(None)
                        
             #           del out_docs
                        del in_docs, out_docs
                
                except torch.cuda.OutOfMemoryError as e:
                    worker_logger.error(f"Worker {gpu_id}: OOM on batch {batch_idx}: {e}")
                    results = [None] * len(texts)
                    gc.collect()
                    torch.cuda.empty_cache()
                
                except RecursionError as e:
                    worker_logger.error(f"Worker {gpu_id}: RecursionError during parsing batch {batch_idx}: {e}")
                    results = [None] * len(texts)
                
                except Exception as e:
                    worker_logger.error(f"Worker {gpu_id}: Unexpected error on batch {batch_idx}: {e}")
                    worker_logger.error(f"Worker {gpu_id}: Traceback:", exc_info=True)
                    results = [None] * len(texts)
                
                output_queue.put((batch_idx, results))
                
                # Light cleanup
                if batch_count % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    worker_logger.info(f"Worker {gpu_id} periodic cleanup at batch {batch_count}")
            
            except queue.Empty:
                worker_logger.warning(f"Worker {gpu_id} timeout waiting for work (5 min)")
                continue
            
            except Exception as e:
                worker_logger.error(f"Worker {gpu_id}: Error in main loop iteration: {e}", exc_info=True)
                # Don't break - try to continue
                
    except KeyboardInterrupt:
        worker_logger.info(f"Worker {gpu_id} received KeyboardInterrupt")
        
    except Exception as e:
        worker_logger.error(f"WORKER {gpu_id} FATAL CRASH: {e}")
        worker_logger.error(f"Worker {gpu_id} full traceback:", exc_info=True)
        
        # Try to send error signal
        try:
            output_queue.put(("ERROR", gpu_id, str(e)))
        except:
            worker_logger.error(f"Worker {gpu_id} could not send error signal to queue")
    
    finally:
        worker_logger.info(f"Worker {gpu_id} shutting down gracefully")
        file_handler.close()            



def collect_results(output_queue, pending_batches, next_batch_to_write, outfile, processed_file, processed_ids, progress_bar, failed_file):
    while not output_queue.empty():
        results = output_queue.get()

        if isinstance(results, tuple) and results[0] == "ERROR":
            continue

        completed_idx, parsed_results = results
        
        if completed_idx != next_batch_to_write:
            pending_batches[f"results_{completed_idx}"] = parsed_results
            continue
        
        write_batch(outfile, processed_file, processed_ids, pending_batches[completed_idx], parsed_results, progress_bar, failed_file)
            
        del pending_batches[completed_idx]
        next_batch_to_write += 1

        while f"results_{next_batch_to_write}" in pending_batches:
            parsed_results = pending_batches[f"results_{next_batch_to_write}"]
            batch_entries = pending_batches[next_batch_to_write]
            write_batch(outfile, processed_file, processed_ids,
                batch_entries, parsed_results, progress_bar, failed_file)
            del pending_batches[f"results_{next_batch_to_write}"]
            del pending_batches[next_batch_to_write]
            next_batch_to_write += 1
    
    return next_batch_to_write

def write_batch(outfile, processed_file, processed_ids, batch_entries, parsed_results, progress_bar, failed_file):
    """Helper function to write a batch of results"""
    for entry, parsed_data in zip(batch_entries, parsed_results):
        if parsed_data is None:
            entry["dependency_parse"] = None

            failed_file.write(json.dumps(entry) + '\n')
            logger.warning(f"Failed to process entry {entry.get('id')}")
        else:    
            entry["dependency_parse"] = parsed_data
        outfile.write(json.dumps(entry) + '\n')
        processed_file.write(str(entry.get("id")) + "\n")
        processed_ids.add(str(entry.get("id")))
    progress_bar.update(len(batch_entries))

def model_sentences(text, keyword):
    if keyword == "NONE":
        return text
    elif keyword not in ["GPT", "EXTRA"]:
        logger.warning("keyword chosen not allowed, defaulted to gpt only")
    
    sentences = sent_tokenize(text)

    matched = [s for s in sentences if any(pattern.search(s) for pattern in MODEL_PATTERNS.values())]

    if keyword == "EXTRA": 
        entry_has_excluded = any(pattern.search(s) for s in sentences for pattern in MODELS_TO_EXCLUDE.values())

        if not entry_has_excluded:
            models = {**MODEL_PATTERNS, **EXTRA_PATTERNS}
            matched = [s for s in sentences if any(pattern.search(s) for pattern in models.values())]
    
    
    return ' '.join(matched)
        

def main():

    processed_ids = set()
    with open(IDFILE, "r", encoding='utf-8') as pfile:
        processed_ids = set(line.strip() for line in pfile)
    
    processed_single = 0


    input_queue = Queue(maxsize=20)
    output_queue = Queue()

    workers = []
    for gpu_id in [0, 1]:
        p = Process(target=gpu_worker, args=(gpu_id, input_queue, output_queue))
        p.start()
        workers.append(p)

    logger.info("GPU workers started")
        
    with open(INPUT, 'r', encoding='utf-8') as infile, \
        open(OUTPUT, 'a', encoding='utf-8') as outfile, \
        open(IDFILE, "a", encoding='utf-8') as processed_file, \
        open(SKIPPEDFILE, "a", encoding='utf-8') as failed_file:

        errors = 0
        batch = []
        batch_entries = []
        batch_idx = 0
        pending_batches = {}
        next_batch_to_write = 0



        infile.seek(0)  # reset file ptr to beginning
        with tqdm(total=TOTAL_ENTRIES ,desc="processing", unit="entries", initial=len(processed_ids)) as progress_bar:

            for line in infile:
                try:
                    entry = json.loads(line)

                    entry_id = entry.get("id")
                    
                    if entry_id in processed_ids:
                        continue

                    if entry.get("type") == "submission":
                        # combine title and selftext for submissions
                        text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
                    else:  # comment
                        text = entry.get("body", "").strip()

                    if not text or text.strip() == "":
                        # skip empty entries
                        entry["dependency_parse"] = None
                        outfile.write(json.dumps(entry) + '\n')
                        progress_bar.update(1)
                        processed_file.write(str(entry.get("id")) + "\n")
                        processed_ids.add(str(entry.get("id")))
                        continue
                                        

                    text = text.replace('\n', ' ').strip()
                    
                    text = model_sentences(text, KEYWORD)

                    if len(text) >=  DO_NOT_PROCESS:
                        failed_file.write(json.dumps(entry), '\n')
                        continue
                    elif len(text) > MAX_CHARS:
                        input_queue.put((batch_idx, [text]))
                        pending_batches[batch_idx] = [entry]
                        batch_idx += 1
                        processed_single += 1

                        next_batch_to_write = collect_results(output_queue, pending_batches, next_batch_to_write, outfile, processed_file, processed_ids, progress_bar, failed_file)
                        continue
                    
                    batch.append(text)
                    batch_entries.append(entry)


                    if len(batch) >= BATCH_SIZE:
                        input_queue.put((batch_idx, batch))
                        pending_batches[batch_idx] = batch_entries
                        batch_idx += 1

                        batch = []
                        batch_entries = []
                        next_batch_to_write = collect_results(output_queue, pending_batches, next_batch_to_write, outfile, processed_file, processed_ids, progress_bar, failed_file)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")
                    errors += 1
                    progress_bar.update(1)
                except Exception as e:
                    logger.exception(f"error processing entry: {e}")
                    errors += 1
                    progress_bar.update(1)
                #    print(entry)

            #---------------------LEFTOVERS IN BATCH FINAL RUN--------------------
            if batch: #essentially checks if batch is empty
                input_queue.put((batch_idx, batch))
                pending_batches[batch_idx] = batch_entries
                batch_idx += 1

            for _ in workers:
                input_queue.put(None)

            total_batches = batch_idx
            while next_batch_to_write < total_batches:
                completed_idx, parsed_results = output_queue.get()
                
                if completed_idx != next_batch_to_write:
                    pending_batches[f"results_{completed_idx}"] = parsed_results
                else:
                    write_batch(outfile, processed_file, processed_ids,
                              pending_batches[completed_idx], parsed_results, progress_bar, failed_file)
                    del pending_batches[completed_idx]
                    next_batch_to_write += 1
                    
                    while f"results_{next_batch_to_write}" in pending_batches and next_batch_to_write < total_batches:
                        parsed_results = pending_batches[f"results_{next_batch_to_write}"]
                        batch_entries = pending_batches[next_batch_to_write]
                        write_batch(outfile, processed_file, processed_ids,
                                  batch_entries, parsed_results, progress_bar, failed_file)
                        del pending_batches[f"results_{next_batch_to_write}"]
                        del pending_batches[next_batch_to_write]
                        next_batch_to_write += 1
    
    # Wait for workers to finish
    for p in workers:
        p.join()




    logger.info(f"errors: {errors:,}")
    logger.info(f"total entries processed: {progress_bar.n}")
    logger.info(f"output saved to: {OUTPUT}")

if __name__ == "__main__":
    main()

