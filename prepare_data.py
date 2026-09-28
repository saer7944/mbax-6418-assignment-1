"""Download and reproducibly sample the complete source; standard library only."""
import argparse, collections, gzip, hashlib, json, random, urllib.request, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz'
NRC_URL = 'https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip'
CLASSES = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
def label(rating):
    return 'POSITIVE' if rating >= 4 else 'NEUTRAL' if rating == 3 else 'NEGATIVE'
def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seed', type=int, default=6418); args=parser.parse_args()
    data=ROOT/'data'; data.mkdir(exist_ok=True)
    for name,url in [('Gift_Cards.jsonl.gz',DATA_URL),('NRC-Emotion-Lexicon.zip',NRC_URL)]:
        target=data/name
        if not target.exists(): urllib.request.urlretrieve(url,target)
    with zipfile.ZipFile(data/'NRC-Emotion-Lexicon.zip') as z:
        names=[n for n in z.namelist() if n.endswith('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt') and '__MACOSX' not in n]
        if len(names)!=1: raise ValueError(f'Expected one English word lexicon, found {names}')
        (data/'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt').write_bytes(z.read(names[0]))
    rng=random.Random(args.seed); seen=collections.Counter(); stars=collections.Counter(); pools={c:[] for c in CLASSES}; first=[]
    with gzip.open(data/'Gift_Cards.jsonl.gz','rt',encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            r=json.loads(line); rating=r['rating']; assert rating in [1,2,3,4,5]
            # No reviewer identifiers, image URLs, or unrelated metadata in saved samples.
            row={k:r.get(k) for k in ['title','text','rating','asin','parent_asin','verified_purchase','helpful_vote','timestamp']}
            row['source_line']=i; row['review_id']=f'GC-{i:06d}'
            if i<=100: first.append(row)
            c=label(rating); seen[c]+=1; stars[int(rating)]+=1
            # Stratified reservoir sampling gives every row within its class equal chance.
            if len(pools[c])<50: pools[c].append(row)
            else:
                j=rng.randrange(seen[c])
                if j<50: pools[c][j]=row
    balanced=sorted([r for c in CLASSES for r in pools[c]],key=lambda r:r['source_line'])
    assert len(balanced)==150
    save(ROOT/'results/first100_sample.json',first); save(ROOT/'results/balanced_sample.json',balanced)
    save(ROOT/'results/data_manifest.json',{'source_url':DATA_URL,'dataset_page':'https://amazon-reviews-2023.github.io/','source_sha256':sha(data/'Gift_Cards.jsonl.gz'),'total_reviews':sum(seen.values()),'rating_counts':dict(sorted(stars.items())),'class_counts':dict(seen),'seed':args.seed,'sampling':'single-pass stratified reservoir, 50 per class, sorted by source line','nrc_url':NRC_URL,'nrc_sha256':sha(data/'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')})
    print(json.dumps({'total':sum(seen.values()),'class_counts':seen,'stars':stars,'balanced':len(balanced)},indent=2))
if __name__=='__main__': main()
