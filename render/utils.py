import numpy as np
import re
from scipy.ndimage import gaussian_filter1d

STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", 
             "to", "for", "of", "with", "by", "as", "is", "are", 
             "was", "were", "be", "been", "being", "this", "that", 
             "these", "those", "it", "its", "which", "who", "whom"}

def filter_for_barchart(word):
    cleaned = re.sub(r'[\W_]+', '', word.lower())
    if not cleaned: 
        return False
    if cleaned in STOPWORDS:
        return False
    return True

def simple_stem(word):
    """
    Very lightweight native suffix strip to avoid installing NLTK explicitly.
    Will map plurals/verbs to common roots protecting ranking distributions.
    """
    w = word.lower()
    if len(w) <= 3: return w
    if w.endswith('ies'): return w[:-3] + 'y'
    if w.endswith('es') and w[-3] in 'szxc': return w[:-2]
    if w.endswith('s') and w[-2] not in 'su': return w[:-1]
    if w.endswith('ing'): return w[:-3]
    if w.endswith('ed'): return w[:-2]
    return w

def group_by_stem_and_sort(word_weights):
    """
    Groups heavily weighted chronological occurrences back into standalone ranked 
    stems to eradicate bar chart duplicated identical strings.
    """
    stem_map = {} 
    # Chronological weight merging
    for word, weight in word_weights:
        cleaned = re.sub(r'[\W_]+', '', word.lower())
        if not cleaned or cleaned in STOPWORDS:
            continue
        stem = simple_stem(cleaned)
        
        if stem not in stem_map:
            stem_map[stem] = {'weight': weight, 'orig': word}
        else:
            if weight > stem_map[stem]['weight']:
                stem_map[stem]['weight'] = weight
            if len(word) < len(stem_map[stem]['orig']):
                stem_map[stem]['orig'] = word

    sorted_stems = sorted(stem_map.values(), key=lambda x: x['weight'], reverse=True)
    return [(item['orig'], item['weight']) for item in sorted_stems]

def stitch_and_smooth_saliency(full_context, chunks_data, sigma=2.0):
    char_probs = np.zeros(len(full_context), dtype=np.float32)
    
    for chunk in chunks_data:
        probs = chunk['probs']
        offsets = chunk['offsets']
        sequence_ids = chunk['sequence_ids']
        
        for p, (start, end), seq_id in zip(probs, offsets, sequence_ids):
            if seq_id == 1 and start < end:
                if start >= len(full_context):
                    continue
                safe_end = min(end, len(full_context))
                current_max = np.max(char_probs[start:safe_end]) if safe_end > start else 0
                if p > current_max:
                    char_probs[start:safe_end] = p
                    
    words = []
    word_probs = []
    
    for match in re.finditer(r'\S+', full_context):
        word = match.group()
        start, end = match.span()
        
        wp = np.max(char_probs[start:end])
        words.append(word)
        word_probs.append(wp)
        
    valid_probs = np.array(word_probs)
    if len(valid_probs) == 0:
        return []
        
    smoothed = gaussian_filter1d(valid_probs, sigma=sigma)
    
    min_p, max_p = np.min(smoothed), np.max(smoothed)
    if max_p > min_p:
        norm_weights = (smoothed - min_p) / (max_p - min_p)
    else:
        norm_weights = smoothed
        
    return list(zip(words, norm_weights))