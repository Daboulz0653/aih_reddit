# AIH Reddit Dependency Parsing Pipeline
This is part of a larger endeavor by the AI Humanities Lab at Washington University in St. Louis to study the discourse around ChatGPT models online, with a specific focus on Reddit AI Communities. 

This repository contains the pipeline used to extract model mentions from 1098440 reddit posts and comments mentioning ChatGPT models from 2022-2024. See [here](https://github.com/starrothkopf/aihlab_reddit/tree/main) for more info on the collection process. 

Data is not currently included in this repo. 


## Scripts
There are two major scripts that drive this pipeline, and small scripts that were used in development and as as checks. 

### 🔗 [interface_multiproc_gptonly.py](https://github.com/Daboulz0653/aih_reddit/blob/main/interface_multiproc_gptonly.py)

#### Generates dependency parses for each entry

Using [TweebankNLP's](https://github.com/mit-ccc/TweebankNLP) pipeline for social media language (with some minor changes), this script processes the entries in four different ways: Tokenization, Lemmatization, POS tagging, and Dependency Parsing. 

What does it actually do?

1. cleans the text
2. extracts and normalizes model names
3. runs pipeline only on sentences with model mentions

The program as it currently is written runs parallel on two GPUs. 

Entries are output as is with 2 extra fields: 

```json
{
  "type": "",
  "id": "",
  "link_id": "",
  "parent_id": "",
  "body": "",
  "author": "",
  "author_flair_text": null,
  "created_utc": null,
  "created_date": null,
  "subreddit": "",
  "score": null,
  "edited": false,
  "distinguished": null,
  "stickied": false,
  "permalink": "",
  "is_submitter": false,
  "controversiality": 0,
  "gilded": 0,
  "name": "",
  "full_link": "",
  "models_detected": ["gpt"],
  "model_inferred_temporal": ["gpt-3.5"],
  "cleaned_text": "",       //text after its cleaned
  "dependency_parse": {
    "usable_text": "",      //sentences with model mentions, delineated by '\n\n' per stanza guidelines
    "full_tree": null,      //dependency parse
    "num_tokens": 0,
    "num_words": 0,
    "num_sentences": 0
  }
}
```

### 🔗 [extract_dependencies_inferred.py](https://github.com/Daboulz0653/aih_reddit/blob/main/extract_dependencies_inferred.py)

#### Extracts direct and one-hop dependencies from trees

This script aims to extract words used in relation to model mentions. This includes the following: 
1. Direct Dependencies: words which are directly tied to the model mentions, including
    1. The head, if it exists, which is the word that the model name points to
    2. Any children,  if they exist, which are words that point to the model name
2. One-Hop Dependencies: words which are one hop away from model mentions. This includes direct dependencies as well as
    1. The head of the head, if it exists, or the grandparent of the model name, which is the word that model’s head points to 
    2. Other children of the head, or the model’s siblings, which are words that similarly point to the head
    3. Children of the children, or the model’s grandchildren, which are words that point to the model’s children
    4. (The head of the children is the model mention itself, so not counted)

Here's an example dependency parse: 


![Dependency Parse Example](https://github.com/Daboulz0653/aih_reddit/blob/main/dependency_tree_example.png)

Model Mention: “GPT4.5PREVIEW”

Direct Dependencies:
- Head: “prompt”
- Children: None

One-Hop Dependencies: 
- Head: “prompt”
- Head of Head – Grandparent: “able”
- Children of Head – Siblings: “if”, “I”
- Children of Children – Grandchildren: None



What does it actually do?
1. finds model mention tokens in tree
2. gets dependencies, either direct or one-hop
3. stores words in dictionary, maintaining a count of:
     1. the different POS tags the words has appeared as
     2. the different models the words has been used in relations with 


It outputs two ndjsons:

allposbreakdown.ndjson

```json
{"word": "word", 
"counts": { "VERB": 0, 
            "NOUN": 0 },
"models": { "gpt-3.5": 0, 
            "gpt-4": 0, 
            "gpt-4o": 0, 
            "gpt-5": 0}}
```
mostfreq.ndjson

```json
{"word": "word", 
"total_count": 0, 
"majority_upos": "NOUN", 
"majority_count": 0, 
"majority_model": "chatgpt", "majority_model_count": 0, 
"model_freqs": {"gpt-3.5": 0, 
                "chatgpt": 0, 
                "gpt-4": 0, 
                "gpt-4o": 0, 
                "gpt-5": 0}}
```


### Other Scripts

🔗 [word_stats.py](https://github.com/Daboulz0653/aih_reddit/blob/main/word_stats.py): basic stats on how long entries are

🔗 [matched.py](https://github.com/Daboulz0653/aih_reddit/blob/main/matched.py): extracts all uppercase matched models from dependency parse. Used to check correctness of regex

🔗 [categorize_skipped.py](https://github.com/Daboulz0653/aih_reddit/blob/main/categorize_skipped.py): looks at skipped entries and groups them into different meaningful categories, including URLs, SUBREDDITs, and others. 

🔗 [sent_diff.py](https://github.com/Daboulz0653/aih_reddit/blob/main/sent_diff.py): tests different sentence tokenizers.

🔗 [regex_tester.py](https://github.com/Daboulz0653/aih_reddit/blob/main/regex_tester.py): tester script used in development of regex patterns. 












