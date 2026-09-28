"""Offline invariants and independent checks of saved output."""
import json, unittest
from prepare_data import ROOT, label
from score_reviews import metrics, request_payload, model_input
from add_emotions import emotion_scores
class PipelineTests(unittest.TestCase):
    def test_rating_boundaries(self):
        self.assertEqual([label(x) for x in [1,2,3,4,5]],['NEGATIVE','NEGATIVE','NEUTRAL','POSITIVE','POSITIVE'])
    def test_rating_never_sent(self):
        row={'title':'Nice','text':'Useful gift','rating':1,'actual':'NEGATIVE','user_id':'PRIVATE'}
        payload=request_payload(row,'NEUTRAL','model',6418)
        self.assertEqual(json.loads(payload['messages'][1]['content']),{'title':'Nice','text':'Useful gift'})
        self.assertNotIn('PRIVATE',json.dumps(payload))
    def test_explicit_rating_text_removed(self):
        self.assertEqual(model_input({'title':'Three Stars','text':'Great: 5 stars, 1star, 2 out of 5, ★★★'})['title'],'[rating omitted]')
        self.assertNotIn('5 stars',model_input({'title':'','text':'Great: 5 stars'})['text'])
    def test_nrc_zero_tie_negation(self):
        lex={'happy':{'joy','trust'},'sad':{'sadness'}}
        self.assertEqual(emotion_scores('','xyz',lex)['nrc_emotion'],'none')
        self.assertEqual(emotion_scores('[rating omitted]','',{'omitted':{'anger'}})['nrc_emotion'],'none')
        r=emotion_scores('happy','happy sad',lex)
        self.assertEqual(r['nrc_scores']['joy'],2);self.assertEqual(r['nrc_emotion'],'joy');self.assertTrue(r['nrc_has_tie'])
        # Document baseline behavior rather than pretending it understands negation.
        self.assertEqual(emotion_scores('','not happy',lex)['nrc_scores']['joy'],1)
    def test_saved_results_independently(self):
        for mode in ['binary','balanced']:
            run=json.loads((ROOT/'results'/f'{mode}_raw.json').read_text());rows=run['reviews'];m=run['metrics']
            self.assertEqual(len(rows),100 if mode=='binary' else 150)
            self.assertEqual(len({r['review_id'] for r in rows}),len(rows))
            self.assertEqual(sum(r['actual']==r['sentiment'] for r in rows),m['correct'])
            self.assertEqual(metrics(rows,m['classes']),m)
            for i,a in enumerate(m['classes']):
                for j,p in enumerate(m['classes']):
                    self.assertEqual(m['confusion_matrix'][i][j],len([r for r in rows if r['actual']==a and r['sentiment']==p]))
            for r in rows:
                expected=label(r['rating']) if mode=='balanced' else ('POSITIVE' if r['rating']>=4 else 'NEGATIVE')
                self.assertEqual(r['actual'],expected)
                self.assertEqual(r['model_input'],model_input(r))
                self.assertEqual(r['sentiment'],json.loads(r['api_response']['choices'][0]['message']['content'])['sentiment'])
            if mode=='balanced':self.assertEqual([m['per_class'][c]['support'] for c in m['classes']],[50,50,50])
    def test_emotion_agreement(self):
        b=json.loads((ROOT/'results/balanced_results.json').read_text());rows=b['reviews'];e=b['emotion_metrics']
        self.assertEqual(e['agreements'],sum(r['emotion']==r['nrc_emotion'] for r in rows))
        self.assertEqual(sum(e['llm_counts'].values()),150);self.assertEqual(sum(e['nrc_counts'].values()),150)
        self.assertEqual(e['no_match'],sum(max(r['nrc_scores'].values())==0 for r in rows))
if __name__=='__main__':unittest.main()
