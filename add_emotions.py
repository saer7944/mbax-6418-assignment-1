"""Add NRC word-count emotions to saved predictions, without model calls."""
import collections, html, json, re
from prepare_data import ROOT, save, sha
EMOTIONS=['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']
def load_lexicon(path):
    words=collections.defaultdict(set)
    for line in path.read_text(encoding='utf-8').splitlines():
        parts=line.split('\t')
        if len(parts)==3 and parts[1] in EMOTIONS and parts[2]=='1': words[parts[0]].add(parts[1])
    if len(words)<1000: raise ValueError('NRC lexicon appears incomplete')
    return words
def emotion_scores(title,text,lexicon):
    clean=re.sub(r'<[^>]*>',' ',html.unescape((title+' '+text).replace('[rating omitted]',' '))).lower()
    tokens=re.findall(r"[a-z]+(?:'[a-z]+)?",clean)
    scores={e:0 for e in EMOTIONS}
    for word in tokens:
        for e in lexicon.get(word,()): scores[e]+=1
    peak=max(scores.values()); tied=[e for e in EMOTIONS if scores[e]==peak] if peak else []
    # Explicit alphabetical tie rule; no-match remains none, never arbitrary anger.
    return {'nrc_emotion':tied[0] if tied else 'none','nrc_scores':scores,'nrc_tied_emotions':tied,'nrc_has_tie':len(tied)>1,'nrc_matched_tokens':sum(bool(lexicon.get(w)) for w in tokens)}
def main():
    lexpath=ROOT/'data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'; lexicon=load_lexicon(lexpath)
    b=json.loads((ROOT/'results/balanced_raw.json').read_text())
    for r in b['reviews']:
        r.update(emotion_scores(r['model_input']['title'],r['model_input']['text'],lexicon));r['emotion_agrees']=r['emotion']==r['nrc_emotion']
    rows=b['reviews']; n=len(rows); agreements=sum(r['emotion_agrees'] for r in rows)
    covered=[r for r in rows if r['nrc_emotion']!='none']
    b['emotion_metrics']={'n':n,'agreements':agreements,'agreement_rate':agreements/n,'disagreements':n-agreements,'no_match':n-len(covered),'tie_count':sum(r['nrc_has_tie'] for r in rows),'covered_agreements':sum(r['emotion_agrees'] for r in covered),'covered_n':len(covered),'llm_counts':dict(collections.Counter(r['emotion'] for r in rows)),'nrc_counts':dict(collections.Counter(r['nrc_emotion'] for r in rows)),'lexicon_sha256':sha(lexpath),'tie_rule':'alphabetically first highest-scoring emotion; none if all zero','tokenization':'HTML removed, lowercased English tokens, exact matches; repeated words count; no stemming or negation adjustment'}
    save(ROOT/'results/balanced_results.json',b); print(json.dumps(b['emotion_metrics'],indent=2))
if __name__=='__main__': main()
