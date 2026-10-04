"""Frozen local BGE question-similarity audit; no training or QA evaluation."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.insurerag_vlm.holdout_quality import normalize

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--candidates',type=Path,nargs='+',required=True)
 p.add_argument('--model',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve previous audit')
 torch.set_num_threads(2);started=time.perf_counter()
 history=[json.loads(l) for l in a.history.read_text(encoding='utf8').splitlines()]
 rows=[json.loads(l) for f in a.candidates for l in f.read_text(encoding='utf8').splitlines() if l.strip()]
 manifest={'history_sha256':sha(a.history),'model_weights_sha256':sha(a.model/'model.safetensors'),
  'model_config_sha256':sha(a.model/'config.json'),'pooling':'normalized CLS','max_length':256,'symmetric_no_query_instruction':True}
 tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 model=AutoModel.from_pretrained(a.model,local_files_only=True,use_safetensors=True).to(a.device).eval()
 if a.device=='cuda':model=model.half()
 def encode(texts,label):
  vectors=[];truncated=[]
  for start in range(0,len(texts),a.batch_size):
   batch=[normalize(t) for t in texts[start:start+a.batch_size]]
   lengths=tokenizer(batch,add_special_tokens=True,truncation=False)['input_ids']
   truncated.extend(start+i for i,tokens in enumerate(lengths) if len(tokens)>256)
   encoded=tokenizer(batch,padding=True,truncation=True,max_length=256,return_tensors='pt').to(a.device)
   with torch.inference_mode():values=torch.nn.functional.normalize(model(**encoded).last_hidden_state[:,0].float(),p=2,dim=1)
   vectors.append(values.cpu().numpy())
   if start%(a.batch_size*100)==0:print(json.dumps({'phase':label,'processed':start+len(batch),'total':len(texts)}),flush=True)
  return np.concatenate(vectors),truncated
 a.cache.mkdir(parents=True,exist_ok=True);metadata=a.cache/'manifest.json';embedding_path=a.cache/'history_embeddings.npy'
 if metadata.exists():
  cache=json.loads(metadata.read_text());assert cache['configuration']==manifest,'History/model cache mismatch'
  assert sha(embedding_path)==cache['embeddings_sha256'],'Corrupt embedding cache'
  historical=np.load(embedding_path);historical_truncated=cache['truncated_indices']
 else:
  historical,historical_truncated=encode([r['text'] for r in history],'historical_questions')
  np.save(embedding_path,historical);metadata.write_text(json.dumps({'configuration':manifest,'embeddings_sha256':sha(embedding_path),'truncated_indices':historical_truncated},indent=2)+'\n',encoding='utf8')
 candidate,candidate_truncated=encode([r['question'] for r in rows],'candidate_questions')
 result=[]
 for start in range(0,len(rows),128):
  sims=candidate[start:start+128]@historical.T
  for i,values in enumerate(sims):
   indices=np.argpartition(values,-3)[-3:];indices=indices[np.argsort(values[indices])[::-1]];idx=start+i
   result.append({'id':rows[idx]['id'],'publisher':rows[idx]['publisher'],'candidate_truncated':idx in candidate_truncated,
    'flag_for_review':float(values[indices[0]])>=.9,'nearest':[{'similarity':float(values[j]),'origin':history[j]['origin'],
    'historical_text_sha256':hashlib.sha256(history[j]['text'].encode()).hexdigest(),'historical_index':int(j),'historical_truncated':int(j) in historical_truncated} for j in indices]})
 sims=candidate@candidate.T;pairs=[]
 for i,j in zip(*np.where(np.tril(sims,k=-1)>=.9)):
  pairs.append({'first':rows[int(i)]['id'],'second':rows[int(j)]['id'],'similarity':float(sims[i,j])})
 # Expose the strongest cross-document alternatives even below the flag threshold.
 # A low score must never be treated as proof of semantic independence.
 for i,row in enumerate(rows):
  values=sims[i].copy()
  for j,other in enumerate(rows):
   if i==j or row.get('document_group')==other.get('document_group'):values[j]=-np.inf
  indices=np.argsort(values)[-3:][::-1]
  result[i]['nearest_other_document_candidates']=[{'id':rows[int(j)]['id'],'similarity':float(values[j])} for j in indices if np.isfinite(values[j])]
 report={'method':'Local frozen BAAI/bge-small-en-v1.5 normalized CLS symmetric question similarity; flags need content disposition',
  'threshold':.9,'historical_questions':len(history),'candidate_questions':len(rows),'historical_truncated':len(historical_truncated),
  'candidate_truncated':len(candidate_truncated),'flagged':sum(r['flag_for_review'] for r in result),'candidate_pair_flags':len(pairs),
  'configuration':manifest,'candidate_file_sha256':{str(f):sha(f) for f in a.candidates},'rows':result,'pairs':pairs,
  'seconds':time.perf_counter()-started,'accepted_items':0,
  'limitations':['This audits questions, not full historical evidence; complement the full-text lexical audit.','Truncated text flags require additional review.','Semantic cosine is a heuristic, not proof of contamination or independence.','Known denied-path and foundation-pretraining gaps remain.']}
 a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(json.dumps({k:v for k,v in report.items() if k not in ['rows','pairs']},indent=2))
if __name__=='__main__':main()
