import stanza
import torch
import numpy as np
# config for the `en_tweet` models (models trained only on Tweebank)
config = {
    'processors': 'tokenize,lemma,pos,depparse', # No 'mwt' here
    'lang': 'en',
    'use_gpu': True,
    'tokenize_model_path': './TweebankNLP/twitter-stanza/saved_models/tokenize/en_tweetewt_tokenizer.pt',
    'lemma_model_path': './TweebankNLP/twitter-stanza/saved_models/lemma/en_tweetewt_lemmatizer.pt',
    'pos_model_path': './TweebankNLP/twitter-stanza/saved_models/pos/en_tweetewt_tagger.pt',
    'depparse_model_path': './TweebankNLP/twitter-stanza/saved_models/depparse/en_tweetewt_parser.pt',
    'tokenize_no_ssplit': True,
}

# Initialize the pipeline using a configuration dict
stanza.download('en')
nlp = stanza.Pipeline(**config)
text = "omg i cant believe it lol anyway moving on this is wild #NLP  said 'its fine' but idk tbh"
doc = nlp(text)
print(doc)
