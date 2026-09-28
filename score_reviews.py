"""Classify title/text only, checkpoint raw API responses, then score by rating."""
import argparse, concurrent.futures, datetime, hashlib, json, os, re, time, urllib.request
from pathlib import Path
from prepare_data import ROOT, CLASSES, label, save
EMOTIONS=['anger','anticipation','disgust','fear','joy','sadness','surprise','trust','none']
def metrics(rows, classes):
    matrix=[[sum(r['actual']==a and r['sentiment']==p for r in rows) for p in classes] for a in classes]
    per={}
    for i,c in enumerate(classes):
        tp=matrix[i][i]; support=sum(matrix[i]); predicted=sum(row[i] for row in matrix)
        precision=tp/predicted if predicted else 0; recall=tp/support if support else 0
        per[c]={'support':support,'predicted':predicted,'correct':tp,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0}
    n=len(rows); correct=sum(matrix[i][i] for i in range(len(classes)))
    return {'n':n,'correct':correct,'mismatches':n-correct,'accuracy':correct/n,'majority_baseline':max(sum(row) for row in matrix)/n,'balanced_accuracy':sum(per[c]['recall'] for c in classes)/len(classes),'macro_f1':sum(per[c]['f1'] for c in classes)/len(classes),'classes':classes,'confusion_matrix':matrix,'per_class':per}
def model_input(row):
    # Remove explicit star-rating phrases in titles/text as well as withholding metadata.
    pattern=r"\b(?:[1-5](?:\.0)?|one|two|three|four|five)\s*[- ]?\s*stars?\b|\b[1-5](?:\.0)?\s*(?:out of|/)\s*5\b|[★☆]+"
    return {k:re.sub(pattern,'[rating omitted]',row[k],flags=re.I) for k in ['title','text']}
def request_payload(row,prompt,model,seed):
    # Whitelist: rating and labels can never enter the API request.
    return {'model':model,'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(model_input(row),ensure_ascii=False)}],'temperature':0,'seed':seed,'max_tokens':256,'response_format':{'type':'json_schema','json_schema':{'name':'review_classification','strict':True,'schema':{'type':'object','properties':{'sentiment':{'type':'string','enum':CLASSES if 'NEUTRAL' in prompt else ['NEGATIVE','POSITIVE']}, **({'emotion':{'type':'string','enum':EMOTIONS}} if 'NEUTRAL' in prompt else {})},'required':['sentiment','emotion'] if 'NEUTRAL' in prompt else ['sentiment'],'additionalProperties':False}}},'chat_template_kwargs':{'enable_thinking':False}}
def classify(row,prompt,config,key,three):
    payload=request_payload(row,prompt,config['model'],config['seed'])
    for attempt in range(4):
        try:
            req=urllib.request.Request(config['base_url']+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            with urllib.request.urlopen(req,timeout=120) as resp: raw=json.load(resp)
            choice=raw['choices'][0]
            if choice.get('finish_reason')!='stop': raise ValueError('Incomplete model response')
            pred=json.loads(choice['message']['content'])
            allowed=CLASSES if three else ['NEGATIVE','POSITIVE']
            assert set(pred)==({'sentiment','emotion'} if three else {'sentiment'})
            assert pred['sentiment'] in allowed
            if three: assert pred['emotion'] in EMOTIONS
            actual=label(row['rating']) if three else ('POSITIVE' if row['rating']>=4 else 'NEGATIVE')
            return {**row,**pred,'model_input':model_input(row),'actual':actual,'correct':actual==pred['sentiment'],'api_response':raw}
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['binary','balanced','spotcheck'],required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    key=os.environ.get('OPENAI_API_KEY')
    if not key: raise SystemExit('Set OPENAI_API_KEY to your class endpoint key. Do not commit it.')
    three=a.mode=='balanced';prompt=(ROOT/'prompts'/('three_class.txt' if three else 'binary.txt')).read_text()
    if a.mode=='spotcheck':
        rows=[{'review_id':'SPOT-P','title':'Wonderful gift','text':'It arrived quickly and my sister loved it. I would happily buy it again.','rating':5},{'review_id':'SPOT-N','title':'Useless','text':'The card would not redeem and support refused to help. A total waste of money.','rating':1}]
    else: rows=json.loads((ROOT/'results'/('balanced_sample.json' if three else 'first100_sample.json')).read_text())
    config={'base_url':os.environ.get('OPENAI_BASE_URL','http://dobolyi.com:9001/v1').rstrip('/'),'model':os.environ.get('OPENAI_MODEL','cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit'),'temperature':0,'seed':6418,'max_tokens':256,'enable_thinking':False,'response_constraint':'json_schema_enums_v1','input_preprocessing':'explicit_star_phrases_redacted_v1','prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'sample_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
    out=ROOT/'results'/f'{a.mode}_raw.json'; checkpoint=ROOT/'results'/f'{a.mode}_checkpoint.json'
    cached={}
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text())
        if previous['config']!=config: raise SystemExit('Checkpoint configuration differs; rename old checkpoint to begin a new run.')
        cached={r['review_id']:r for r in previous['reviews']}
    pending=[r for r in rows if r['review_id'] not in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        errors=[]
        futures={pool.submit(classify,r,prompt,config,key,three):r for r in pending}
        for f in concurrent.futures.as_completed(futures):
            try: result=f.result()
            except Exception as exc:
                errors.append({'review_id':futures[f]['review_id'],'error':repr(exc)});continue
            cached[result['review_id']]=result
            save(checkpoint,{'config':config,'reviews':list(cached.values())})
            if len(cached)%10==0 or len(rows)<10: print(f'{a.mode}: {len(cached)}/{len(rows)}',flush=True)
    if errors:
        save(ROOT/'results'/f'{a.mode}_errors.json',errors)
        raise SystemExit(f'{len(errors)} requests failed; successful results checkpointed. Rerun to retry.')
    ordered=[cached[r['review_id']] for r in rows]
    bundle={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':a.mode,'config':config,'prompt':prompt,'metrics':metrics(ordered,CLASSES if three else ['NEGATIVE','POSITIVE']),'reviews':ordered}
    save(out,bundle);checkpoint.unlink(missing_ok=True)
    print(json.dumps(bundle['metrics'],indent=2))
if __name__=='__main__': main()
