import nltk
import stanza
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

# Sample texts with known edge cases
test_texts = [
    # Missing periods, run-on sentences
    "i went to the store then i saw him he was just standing there i couldn't believe it",
    
    # Abbreviations + informal
    "met w/ Dr Smith at 3pm re: the study results vs last yr's data etc he said smth interesting",
    
    # Twitter-style, hashtags, @mentions
    "just saw @JohnDoe lol he was like 'wth' and i was like same tbh #relatable #mood fr fr",
    
    # Numbers and decimals mid-sentence
    "stock went up 3.5 then dropped to 2.1 then hit 4.0 crazy right it was wild",
    
    # Ellipses as sentence breaks
    "i don't know... maybe it's fine... but then again... who knows anymore",
    
    # Mixed case, no punctuation
    "She said NO then he said yes then she said FINE whatever i give up honestly",
    
    # URLs mid-text
    "check this out https://example.com/thing it's insane also see twitter.com/user for more",
    
    # Emoji as punctuation substitutes
    "it was amazing 😍 then it got weird 😬 now i don't know what to think 🤷",
    
    # Quote within sentence, ambiguous boundary
    "he said 'i'm done' she said 'no you're not' then walked out it was so dramatic",
    
    # Numbers that look like sentence starts
    "we had 3 options. 1. leave early 2. stay late 3. just not show up at all",
    
    # Slang abbreviations that look like sentence ends
    "went to the U.S. last summer tbh it was ok lol but the food was good ngl",
    
    # Mixed formal and informal
    "The results were significant (p<0.05) which is great but tbh the sample size was tiny so idk",
    
    # All lowercase, multiple thoughts
    "woke up late missed the bus got coffee spilled it on my shirt today is not it",
    
    # Trailing off, no ending punctuation
    "if only we had more time to actually look at the data properly because right now",

    "I want to know who these mental health experts are and hear exactly what their entire theory behind their processes even was in making these decisions. \n\nLet's say I'm someone actually worried about people's mental Health with chat GPT. I'm just supposed to trust that you found experts that know what they're talking about and then just assume it's going to be okay for the user? Bs. Don't trust this at all.\n\nI want to know who these experts are before I ever can trust what Chatgpt is doing.",
    "Yep I just fired them for trying to comfort me while I was telling it about a safety risk I faced from another person . It thought I was the problem . I’ve had enough . Whoever did this to ChatGPT ruined it and yes I’m not happy.",
    "you’d be very surprised. go check out r/artificialsentience multiple people over there argued that ChatGPT has a real emotional connection with them and that it is “alive” and has desires. I tried explaining that it is just an algorithm that spits out the best string of words given the user’s input, but they disagreed, even though that is an undeniable fact. It’s very concerning some of the things people have deluded themselves into believing & I think it’ll only get worse as the technology continues to advance.",
    "omg i cant believe it lol... anyway moving on RT @user: this is wild #NLP u/someone said 'its fine' but idk tbh"
    ]

# --- Stanza pipeline ---
config2 = {
          'processors': 'tokenize',
          'lang': 'en',
          'use_gpu': True,
          'tokenize_pretokenized': False, # disable tokenization
          'tokenize_model_path': './TweebankNLP/twitter-stanza/saved_models/tokenize/en_tweetewt_tokenizer.pt',
}
config1 = {
        'processors': 'tokenize',
          'lang': 'en',
          'use_gpu': True,
          'tokenize_pretokenized': False, # disable tokenization
          'tokenize_model_path': './TweebankNLP/twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt'}

# Initialize the pipeline using a configuration dict
stanza.download('en')
nlp3 = stanza.Pipeline(**config1)
nlp1 = stanza.Pipeline(**config2)
nlp2 = stanza.Pipeline(lang= 'en', processors='tokenize')

def stanza1_split(text):
    doc = nlp1(text)
    return [sent.text for sent in doc.sentences]
def stanza2_split(text):
    doc = nlp2(text)
    return [sent.text for sent in doc.sentences]
def stanza3_split(text):
    doc = nlp3(text)
    return [sent.text for sent in doc.sentences]

# --- NLTK ---
def nltk_split(text):
    return sent_tokenize(text)

# --- Compare ---
for text in test_texts:
    stanza1_sents = stanza1_split(text)
    stanza2_sents = stanza2_split(text)
    stanza3_sents = stanza3_split(text)
    nltk_sents = nltk_split(text)
    
    print(f"\nINPUT: {text}")
    print(f"\n  STANZA Tweebank EWT({len(stanza1_sents)} sentences):")
    for i, s in enumerate(stanza1_sents):
        print(f"    {i+1}: {s}")
    print(f"\n  STANZA Tweebank  ({len(stanza3_sents)} sentences):")
    for i, s in enumerate(stanza3_sents):
        print(f"    {i+1}: {s}")
    print(f"\n  STANZA Default ({len(stanza2_sents)} sentences):")
    for i, s in enumerate(stanza2_sents):
        print(f"    {i+1}: {s}")
    print(f"\n  NLTK ({len(nltk_sents)} sentences):")
    for i, s in enumerate(nltk_sents):
        print(f"    {i+1}: {s}")
    print("-" * 60)
