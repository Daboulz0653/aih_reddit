import re
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
#VERSION = r"(?:3(?:\.5)?|4(?:\.1|\.5|o)?|5)"
#MODEL_PATTERNS = {
#    "gpt-4o": re.compile(rf"\bgpt[-\s]?4o{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4.5": re.compile(rf"\bgpt[-\s]?4\.5{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-4.1": re.compile(rf"\bgpt[-\s]?4\.1{SUFFIX}{END}", re.IGNORECASE),
 #   "gpt-4": re.compile(rf"\bgpt[-\s]?4{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-3.5": re.compile(rf"\bgpt[-\s]?3\.?5{SUFFIX}{END}", re.IGNORECASE),
#    "gpt-5": re.compile(rf"\bgpt[-\s]?5{SUFFIX}{END}", re.IGNORECASE),
#    "chatgpt": re.compile(rf"\bchat[-\s]?gpt{SUFFIX}{END}", re.IGNORECASE),
 #   "gpt": re.compile(rf"\bgpt{SUFFIX}{END}", re.IGNORECASE),
   # "chatgpt": re.compile(
   #     rf"\bchat[-\s]?gpt(?:[-\s]?([345]|4\.5|4\.1|4o|5))?{SUFFIX}{END}",
   #     re.IGNORECASE
   # ),

    # Generic GPT catch-all (with optional version)
    #"gpt": re.compile(
    #    rf"\bgpt(?:[-\s]?([345]|4\.5|4\.1|4o|5))?{SUFFIX}{END}",
    #    re.IGNORECASE
    #),
#    "chatgpt": re.compile(
#        rf"\bchat[-\s]?gpt(?:[-\s]?{VERSION})?{SUFFIX}{END}",
#        re.IGNORECASE
#    ),
#
#    "gpt": re.compile(
#        rf"\bgpt(?:[-\s]?{VERSION})?{SUFFIX}{END}",
#        re.IGNORECASE
#    ),
#}
#SUFFIX = r"(?:[-\s]?(?:turbo|mini|nano|preview|vision|audio|realtime|instruct))*"
#MODEL_PATTERNS = {
#    "gpt-4o": re.compile(rf"\bgpt[-\s]?4o{SUFFIX}", re.IGNORECASE),
#    "gpt-4.5": re.compile(rf"\bgpt[-\s]?4\.5{SUFFIX}", re.IGNORECASE),
#    "gpt-4.1": re.compile(rf"\bgpt[-\s]?4\.1{SUFFIX}", re.IGNORECASE),
#    "gpt-4": re.compile(rf"\bgpt[-\s]?4{SUFFIX}", re.IGNORECASE),
#    "gpt-3.5": re.compile(rf"\bgpt[-\s]?3\.?5{SUFFIX}", re.IGNORECASE),
#    "gpt-5": re.compile(rf"\bgpt[-\s]?5{SUFFIX}", re.IGNORECASE),
#    "chatgpt": re.compile(rf"\bchat[-\s]?gpt{SUFFIX}", re.IGNORECASE),
#    "gpt": re.compile(rf"\bgpt{SUFFIX}", re.IGNORECASE),
#}

#MODEL_PATTERNS = {
#    "gpt-4o": re.compile(r"\bgpt[-\s]?4o(?![a-z])", re.IGNORECASE),
#    "gpt-4.5": re.compile(r"\bgpt[-\s]?4\.5(?![a-z])", re.IGNORECASE),
#    "gpt-4.1": re.compile(r"\bgpt[-\s]?4\.1(?![a-z])", re.IGNORECASE),
#    "gpt-4-turbo": re.compile(r"\bgpt[-\s]?4[-\s]?turbo(?![a-z])", re.IGNORECASE),
#    "gpt-4": re.compile(r"\bgpt[-\s]?4(?!o|\.5|\.1|[-\s]turbo|[a-z])", re.IGNORECASE),
#    "gpt-3.5": re.compile(r"\bgpt[-\s]?3\.?5(?![a-z])", re.IGNORECASE),
#    "gpt-5": re.compile(r"\bgpt[-\s]?5(?![a-z])", re.IGNORECASE),
#    "chatgpt": re.compile(r"\bchat[-\s]?gpt(?![a-z])", re.IGNORECASE),
#    "gpt": re.compile(r"\bgpt(?![a-z])", re.IGNORECASE),
#}

#MODEL_PATTERNS = {
#        "gpt-3.5": re.compile(r"gpt[-\s]?3\.?5", re.IGNORECASE),
#        "gpt-4": re.compile(r"gpt[-\s]?4(?!o|\.5|\.1|[-\s]turbo)\b", re.IGNORECASE),
#        "gpt-4-turbo": re.compile(r"gpt[-\s]?4[-\s]turbo|gpt[-\s]turbo", re.IGNORECASE),
#        "gpt-4o": re.compile(r"gpt[-\s]?4o", re.IGNORECASE),
#        "gpt-4.5": re.compile(r"gpt[-\s]?4\.5", re.IGNORECASE),
#        "gpt-4.1": re.compile(r"gpt[-\s]?4\.1", re.IGNORECASE),
#        "gpt-5": re.compile(r"gpt[-\s]?5\b", re.IGNORECASE),
#        "chatgpt": re.compile(r"\bchat[-\s]?gpt\b", re.IGNORECASE),
#        "gpt": re.compile(r"gpt", re.IGNORECASE),
#}
##S = r"[- ]?"
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
def clean(text):
    # Normalize dashes and bullets
    text = re.sub(r'[\u2011-\u2015\u2212\u2022\u2023\u2030\u2031]', ' ', text)

    # Remove nonstandard punctuation (Arabic/Chinese)
    text = re.sub(r'[\u060c\u061b\u061f\u3001\u3002\u300c\u300d\uff01\uff0c\uff1a\uff1b\uff1f]', ' ', text)

    # Normalize quotes
    text = re.sub(r"[‘’`´]", "'", text)
    text = re.sub(r'["“”„‟«»＂]', '"', text)

    # Keep underscores and dots — do not touch
    # Only remove markdown symbols that aren't part of model names
    text = re.sub(r'[*#~\\/|]', ' ', text)

    # Normalize ellipsis
    text = text.replace("…", "...")
    text = re.sub(r'([.!?]){2,}', r'\1', text)
# Pad parentheses, quotes, and backticks with spaces so regex can match
    text = re.sub(r'([()\[\]{}"\'`])', r' \1 ', text)
    # Separate "'s" from the model name
    text = re.sub(r"'s", " 's", text)
    # Collapse whitespace
    text = " ".join(text.split())

    return text
def test_regex():
    # These should NOT be normalized/changed (or should be ignored)
    junk_inputs = [
        "CHATGPT.PROMT", "CHATGPT.PY", "CHATGPT0", "CHATGPT001", 
        "GPT.WOLFRAM.COM", "GPT0", "GPT0314", "GPT10.0", "GPT1.5IMAGE",
    "chatgpt.py",
    "chatgpt.js",
    "chatgpt.java",
    "chatgpt.ai",
    "chatgpt.com",
    "chatgpt.org",
    "chatgpt.net",
    "gpt4.md",
    "gpt4.json",
    "gpt4.csv",
    "gpt.wolfram.com",
    "gpt.openai.com",
    "chatgpt123",
    "chatgpt001",
    "gpt100",
    "gpt10.0",
    "gpt4file",
    "gpt4script",
    "chatgpt_test",
    "chatgpt-build",
        ]

    # These SHOULD be normalized
    valid_inputs = {
        "I love gpt-4o": "I love GPT-4O",
    "using gpt-4.5 today": "using GPT-4.5 today",
    "chat-gpt is cool": "CHATGPT is cool",
    "standard gpt 4": "standard GPT-4",
    "i used gpt 3.5 turbo today": "i used GPT3.5TURBO today",
    "i think chatgpt 4 is the smartest": "i think CHATGPT4 is the smartest",
    "i love chatgpt.": "i love CHATGPT.",
    "chatgpt. its stupid": "CHATGPT. its stupid",
    "chatgpt...": "CHATGPT.",
    "chatgpt???": "CHATGPT?",
    "chatgpt!!!": "CHATGPT!",
    "chatgpt, honestly": "CHATGPT, honestly",
    "chatgpt: amazing": "CHATGPT: amazing",
    "chatgpt; seriously": "CHATGPT; seriously",
    "chatgpt)": "CHATGPT)",
    "(chatgpt)": "(CHATGPT)",
    "\"chatgpt\"": "\"CHATGPT\"",
    "'chatgpt'": "'CHATGPT'",
    "`chatgpt`": "`CHATGPT`",
    "**chatgpt**": "**CHATGPT**",
    "chatgpt/gpt4": "CHATGPT/GPT4",
    "gpt4??? this is crazy": "GPT4? this is crazy",
    "gpt3. its stupid": "GPT3. its stupid",
    "chatgpt. honestly": "CHATGPT. honestly",
    "gpt4! amazing": "GPT4! amazing",
    "gpt4: interesting": "GPT4: interesting",
    "gpt4; interesting": "GPT4; interesting",
    "chat-gpt 4 mini": "CHATGPT4MINI",
    "chatgpt4": "CHATGPT4",
    "gpt4mini": "GPT4MINI",
    "gpt4 turbo": "GPT4TURBO",
    "chatgpt's amazing": "CHATGPT's amazing",  # keep 's attached    
       # "I love gpt-4o": "I love GPT-4O",
       # "using gpt-4.5 today": "using GPT-4.5 today",
       # "chat-gpt is cool": "CHATGPT is cool",
       # "standard gpt 4": "standard GPT-4",
       # "i used gpt 3.5 turbo today" : "i used GPT3.5TURBO today",
       # " i think chatgpt 4 is the smartesT": "i think CHATGPT4 is the smartest",
       # " i love chatgpt.": "i love CHATGPT.",
       # "chatgpt. its stupid": "CHATGPT. its stupid",
       # "chatgpt...": "CHATGPT.",
       # "chatgpt???": "CHATGPT?",
       # "chatgpt!!!": "CHATGPT!",
       # "chatgpt, honestly": "CHATGPT, honestly",
       # "chatgpt: amazing": "CHATGPT: amazing",
       # "chatgpt; seriously": "CHATGPT; seriously",
       # "chatgpt)": "CHATGPT)",
       # "(chatgpt)": "(CHATGPT)",
       # "\"chatgpt\"": "\"CHATGPT\"",
       # "'chatgpt'": "'CHATGPT'",
       # "`chatgpt`": "`CHATGPT'",
       # "**chatgpt**": "**CHATGPT**",
       # "chatgpt/gpt4": "CHATGPT/GPT4",
       # "gpt4??? this is crazy": "GPT4? this is crazy",
       # "gpt3. its stupid": "GPT3. its stupid",
       # "chatgpt. honestly": "CHATGPT. honestly",
       # "gpt4! amazing": "GPT4! amazing",
       # "gpt4: interesting": "GPT4: interesting",
       # "gpt4; interesting": "GPT4; interesting",
       # "chat-gpt 4 mini": "CHATGPT4MINI",
       # "chatgpt4": "CHATGPT4",
       # "gpt4mini": "GPT4MINI",
       # "gpt4 turbo": "GPT4TURBO",
       # " chatgpt's amazing": "CHATGPT's amazing"
        }

    print("--- RUNNING STRICTURE TEST ---")
    
    # Test 1: False Positives (The Junk)
    failures_junk = []
    for text in junk_inputs:
        original = text
        for name, pattern in MODEL_PATTERNS.items():
            if pattern.search(clean(text)):
                failures_junk.append(f"FAILED: Pattern '{name}' incorrectly matched '{original}'")
    
    if not failures_junk:
        print("✅ SUCCESS: All junk strings were correctly ignored.")
    else:
        for f in failures_junk: print(f)
    
    failures_valid = []
    for text, expected in valid_inputs.items():
    
        found = False
        for name, pattern in MODEL_PATTERNS.items():
            if pattern.search(clean(text)):
                found = True
                break

        if not found:
            failures_valid.append(f"FAILED: Could not find model in '{text}'")
#    # Test 2: Valid Matches
 #   failures_valid = []
  #  for text, expected in valid_inputs.items():
   #     result = text
    #    for name, pattern in MODEL_PATTERNS.items():
     #       result = pattern.sub(name, result)
      #  
       # if result.lower() == text.lower(): # If no replacement happened
        #    failures_valid.append(f"FAILED: Could not find model in '{text}'")

    if not failures_valid:
        print("✅ SUCCESS: All valid model names were identified.")
    else:
        for f in failures_valid: print(f)

if __name__ == "__main__":
    test_regex()
