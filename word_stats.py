import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_corpus_lengths(input_file):
    """Analyze text lengths in your corpus"""
    char_lengths = []
    word_counts = []
    large = 0
    larger = 0
    print("Analyzing corpus...")
    with open(input_file, 'r', encoding='utf-8') as f, \
            open("long_entry_ids.txt", "w", encoding='utf-8') as ids:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"Processed {i} entries...")
            
            try:
                entry = json.loads(line)
                entry_id = entry.get("id")
                # Get text
                if entry.get("type") == "submission":
                    text = f"{entry.get('title', '')} {entry.get('selftext', '')}".strip()
                else:
                    text = entry.get("body", "").strip()
                
                if text:
                    char_lengths.append(len(text))
                    word_counts.append(len(text.split()))

                    if len(text) >= 20000:
                        ids.write(entry_id + '\n')
                        large +=1

                    if len(text) > 30000:
                        larger += 1
            except:
                continue
    
    char_lengths = np.array(char_lengths)
    word_counts = np.array(word_counts)
    
    print("\n" + "="*60)
    print("CHARACTER LENGTH STATISTICS")
    print("="*60)
    print(f"Total entries: {len(char_lengths):,}")
    print(f"Mean: {np.mean(char_lengths):,.0f}")
    print(f"Median: {np.median(char_lengths):,.0f}")
    print(f"Max: {np.max(char_lengths):,}")
    print(f"> 20000: {large}")
    print(f"> 30000: {larger}")
    print(f"\nPercentiles (characters):")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        val = np.percentile(char_lengths, p)
        count_over = np.sum(char_lengths > val)
        pct_over = (count_over / len(char_lengths)) * 100
        print(f"  {p:5.1f}%: {val:8,.0f} chars ({count_over:6,} entries or {pct_over:.2f}% above this)")
    
    print("\n" + "="*60)
    print("WORD COUNT STATISTICS")
    print("="*60)
    print(f"Mean: {np.mean(word_counts):,.0f}")
    print(f"Median: {np.median(word_counts):,.0f}")
    print(f"Max: {np.max(word_counts):,}")
    print(f"\nPercentiles (words):")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        val = np.percentile(word_counts, p)
        count_over = np.sum(word_counts > val)
        pct_over = (count_over / len(word_counts)) * 100
        print(f"  {p:5.1f}%: {val:8,.0f} words ({count_over:6,} entries or {pct_over:.2f}% above this)")
    
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLDS")
    print("="*60)
    
    # Conservative (99th percentile)
    conservative_chars = int(np.percentile(char_lengths, 99))
    conservative_words = int(np.percentile(word_counts, 99))
    skipped_conservative = np.sum(char_lengths > conservative_chars)
    
    # Moderate (99.5th percentile)
    moderate_chars = int(np.percentile(char_lengths, 99.5))
    moderate_words = int(np.percentile(word_counts, 99.5))
    skipped_moderate = np.sum(char_lengths > moderate_chars)
    
    # Aggressive (99.9th percentile)
    aggressive_chars = int(np.percentile(char_lengths, 99.9))
    aggressive_words = int(np.percentile(word_counts, 99.9))
    skipped_aggressive = np.sum(char_lengths > aggressive_chars)
    
    print(f"\n1. CONSERVATIVE (skip top 1%):")
    print(f"   MAX_CHARS = {conservative_chars:,}")
    print(f"   MAX_ESTIMATED_TOKENS = {int(conservative_words * 1.3):,}")
    print(f"   → Will skip {skipped_conservative:,} entries ({skipped_conservative/len(char_lengths)*100:.2f}%)")
    
    print(f"\n2. MODERATE (skip top 0.5%):")
    print(f"   MAX_CHARS = {moderate_chars:,}")
    print(f"   MAX_ESTIMATED_TOKENS = {int(moderate_words * 1.3):,}")
    print(f"   → Will skip {skipped_moderate:,} entries ({skipped_moderate/len(char_lengths)*100:.2f}%)")
    
    print(f"\n3. AGGRESSIVE (skip top 0.1%):")
    print(f"   MAX_CHARS = {aggressive_chars:,}")
    print(f"   MAX_ESTIMATED_TOKENS = {int(aggressive_words * 1.3):,}")
    print(f"   → Will skip {skipped_aggressive:,} entries ({skipped_aggressive/len(char_lengths)*100:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(char_lengths, bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(conservative_chars, color='r', linestyle='--', label=f'99th %ile: {conservative_chars:,}')
    plt.axvline(moderate_chars, color='orange', linestyle='--', label=f'99.5th %ile: {moderate_chars:,}')
    plt.xlabel('Character Length')
    plt.ylabel('Frequency')
    plt.title('Character Length Distribution')
    plt.legend()
    plt.xlim(0, min(np.percentile(char_lengths, 99.9) * 1.2, np.max(char_lengths)))
    
    plt.subplot(1, 2, 2)
    plt.hist(word_counts, bins=100, edgecolor='black', alpha=0.7, color='green')
    plt.axvline(conservative_words, color='r', linestyle='--', label=f'99th %ile: {conservative_words:,}')
    plt.axvline(moderate_words, color='orange', linestyle='--', label=f'99.5th %ile: {moderate_words:,}')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution')
    plt.legend()
    plt.xlim(0, min(np.percentile(word_counts, 99.9) * 1.2, np.max(word_counts)))
    
    plt.tight_layout()
    plt.savefig('corpus_length_distribution.png', dpi=150)
    print(f"\n📊 Distribution plot saved to: corpus_length_distribution.png")

if __name__ == "__main__":
    analyze_corpus_lengths("../raw_data/combined_corpus.ndjson")
