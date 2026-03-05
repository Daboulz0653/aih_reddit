import stanza
import torch
import numpy as np
# config for the `en_tweet` models (models trained only on Tweebank)
config = {
          'processors': 'tokenize,lemma, pos,depparse',
          'lang': 'en',
          'use_gpu': True,
          'tokenize_pretokenized': False, # disable tokenization
          'tokenize_model_path': './TweebankNLP/twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt',
          'lemma_model_path': './TweebankNLP/twitter-stanza/saved_models/lemma/en_tweetewt_lemmatizer.pt',
          'pos_model_path': './TweebankNLP/twitter-stanza/saved_models/pos/en_tweetewt_tagger.pt',
          'depparse_model_path': 'TweebankNLP//twitter-stanza/saved_models/depparse/en_tweetewt_parser.pt',
}

# Initialize the pipeline using a configuration dict
stanza.download('en')
nlp = stanza.Pipeline(**config)
doc = nlp("How goes it?")
print(doc) # Look at the result
