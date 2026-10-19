"""Fixed-threshold detection, subgroup, calibration and bootstrap analyses."""
from pathlib import Path
import json, csv
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation'
NAMES=['Longitudinal crack','Transverse crack','Alligator crack','Other corruption','Pothole']

def ious(a,b):
    a=np.asarray(a).reshape(-1,4); b=np.asarray(b).reshape(-1,4)
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    area=np.maximum(rb-lt,0).prod(2)
    return area/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-area,1e-12)

def gt_boxes(row):
    a=np.array(row['boxes']).reshape(-1,5); h,w=row['shape']
    xy=a[:,1:3]*[w,h]; wh=a[:,3:]*[w,h]
    return np.concatenate([xy-wh/2,xy+wh/2],axis=1),a[:,0].astype(int)

def match(row,pred,threshold=.25):
    gt,gc=gt_boxes(row)
    confidence=np.asarray(pred['conf']); keep=np.flatnonzero(confidence>=threshold)
    keep=keep[np.argsort(-confidence[keep],kind='stable')]
    pb=np.asarray(pred['boxes']).reshape(-1,4)[keep]; pc=np.asarray(pred['cls'],dtype=int)[keep]
    ov=ious(pb,gt); used=set(); tp=np.zeros(len(keep),dtype=bool); matched=[]
    for j,c in enumerate(pc):
        candidates=[k for k in range(len(gc)) if gc[k]==c and k not in used and ov[j,k]>=.5]
        if candidates:
            k=max(candidates,key=lambda k:ov[j,k]); used.add(k); tp[j]=True; matched.append(float(ov[j,k]))
    counts=np.zeros((5,3),dtype=int)
    for c in range(5):
        counts[c]=[sum(tp & (pc==c)),sum(~tp & (pc==c)),sum(gc==c)-sum(tp & (pc==c))]
    return counts,confidence[keep],tp,matched

def ratios(counts):
    t,f,n=np.asarray(counts).sum(axis=0) if np.asarray(counts).ndim==2 else counts
    p=t/(t+f) if t+f else 0.; r=t/(t+n) if t+n else 0.
    return {'tp':int(t),'fp':int(f),'fn':int(n),'precision':float(p),'recall':float(r),'f1':float(2*p*r/(p+r)) if p+r else 0.}

def binary_auc(y,s):
    y=np.array(y); s=np.array(s); order=np.argsort(s); ranks=np.empty(len(s),float)
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2; i=j
    n1=y.sum(); n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def main():
    records=json.loads((OUT/'data_records.json').read_text())['test']
    predictions={p['name']:p for p in json.loads((OUT/'baseline_predictions.json').read_text())}
    rows=[]; confs=[]; correct=[]; overlap=[]; counts=[]; scores=[]; ys=[]
    for row in records:
        p=predictions[row['name']]; c,s,t,ov=match(row,p)
        counts.append(c); confs.extend(s); correct.extend(t); overlap.extend(ov)
        score=max(p['conf'],default=0.); y=int(len(row['boxes'])>0)
        scores.append(score); ys.append(y)
        rows.append({'name':row['name'],'country':row['country'],'objects':len(row['boxes']),
                     **ratios(c),'damage_present':y,'damage_score':score,'predicted_damage':int(score>=.25)})
    a=np.array(counts); result={'fixed_threshold':.25,'matching_iou':.5,'overall':ratios(a.sum(0)),
        'per_class':{NAMES[i]:ratios(a[:,i,:].sum(0)) for i in range(5)},'subgroups':{}}
    for country in sorted(set(x['country'] for x in records)):
        idx=[i for i,x in enumerate(records) if x['country']==country]
        result['subgroups'][country]={'images':len(idx),'objects':sum(len(records[i]['boxes']) for i in idx),**ratios(a[idx].sum((0,1)))}
    y=np.array(ys); s=np.array(scores); pred=s>=.25
    binary={'tp':int(sum(pred & (y==1))),'fp':int(sum(pred & (y==0))),'tn':int(sum(~pred & (y==0))),'fn':int(sum(~pred & (y==1)))}
    binary.update(accuracy=float(np.mean(pred==y)),precision=float(sum(pred & (y==1))/sum(pred)),
        recall=float(sum(pred & (y==1))/sum(y)),specificity=float(sum(~pred & (y==0))/sum(y==0)),
        roc_auc=binary_auc(y,s),brier_score=float(np.mean((s-y)**2)),background_false_alarm_rate=float(sum(pred & (y==0))/sum(y==0)))
    result['image_level']=binary
    cs=np.array(confs); ct=np.array(correct,dtype=float); bins=[]; ece=0
    for lo in np.arange(0,1,.1):
        ix=(cs>=lo)&(cs<lo+.1 if lo<.9 else cs<=1)
        if ix.any():
            b={'lower':float(lo),'upper':float(lo+.1),'n':int(ix.sum()),'confidence':float(cs[ix].mean()),'precision':float(ct[ix].mean())}
            ece+=ix.mean()*abs(b['confidence']-b['precision']); bins.append(b)
    result['detection_calibration']={'bins':bins,'ece_10_equal_width_bins':float(ece),
        'brier_score':float(np.mean((cs-ct)**2)),'predictions':len(cs),'mean_matched_iou':float(np.mean(overlap))}
    rng=np.random.default_rng(23022); dist=[]; binary_dist=[]; totals=a.sum(1)
    for _ in range(2000):
        ix=rng.integers(0,len(records),len(records)); m=ratios(totals[ix].sum(0))
        dist.append([m['precision'],m['recall'],m['f1']]); binary_dist.append(np.mean(pred[ix]==y[ix]))
    result['bootstrap_95ci']={k:np.percentile(np.array(dist)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['precision','recall','f1'])}
    result['bootstrap_95ci']['image_accuracy']=np.percentile(binary_dist,[2.5,97.5]).tolist()
    result['threshold_sweep']=[]
    for threshold in [.1,.25,.5,.75]:
        c=sum((match(row,predictions[row['name']],threshold)[0] for row in records),np.zeros((5,3),int))
        result['threshold_sweep'].append({'confidence':threshold,**ratios(c)})
    with (OUT/'per_image_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
