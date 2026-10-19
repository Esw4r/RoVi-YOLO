"""Seeded corruption, metamorphic, latency, resilience and local sensitivity probes."""
from pathlib import Path
import json, time, hashlib
import cv2, numpy as np, torch
from ultralytics import YOLO
from analyze_results import match, ratios, ious

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output/evaluation'
SEED=23022

def pack(r):
    b=r.boxes.cpu()
    return {'boxes':b.xyxy.tolist(),'conf':b.conf.tolist(),'cls':b.cls.tolist()}

def summary(records,preds):
    c=sum((match(r,p)[0] for r,p in zip(records,preds)),np.zeros((5,3),int))
    return {'images':len(records),'objects':sum(len(r['boxes']) for r in records),**ratios(c)}

def main():
    torch.set_num_threads(4)
    rows=json.loads((OUT/'data_records.json').read_text())['test']
    rng=np.random.default_rng(SEED)
    # Uniform sampling without replacement; report realized group counts.
    subset=[rows[i] for i in sorted(rng.choice(len(rows),300,replace=False))]
    images=[]; records=[]
    for row in subset:
        im=cv2.imread(str(ROOT/row['file'])); h,w=im.shape[:2]
        im=cv2.resize(im,(round(w*512/max(h,w)),round(h*512/max(h,w))),interpolation=cv2.INTER_AREA)
        images.append(im); records.append({**row,'shape':list(im.shape[:2])})
    model=YOLO(str(ROOT/'gopher-ai_v1.pt'))
    kw=dict(imgsz=512,device=0,half=False,conf=.001,iou=.7,max_det=300,verbose=False,rect=False)
    def predict(ims):
        output=[]
        for start in range(0,len(ims),8):
            output.extend(pack(r) for r in model.predict(ims[start:start+8],**kw))
        return output
    result={'seed':SEED,'sample_size':len(records),'sample_names':[x['name'] for x in records],
            'corruptions':{},'preprocessing':'Longest-side resize to 512 before corruption; square 512 letterbox; predict single-label NMS; FP32',
            'sample_countries':{c:sum(r['country']==c for r in records) for c in sorted(set(r['country'] for r in records))}}
    clean=predict(images); result['corruptions']['clean']=summary(records,clean)
    print('clean',result['corruptions']['clean'],flush=True)
    variants={
        'dark_0.75':lambda im,rng:np.clip(im.astype(float)*.75,0,255).astype('uint8'),
        'dark_0.50':lambda im,rng:np.clip(im.astype(float)*.5,0,255).astype('uint8'),
        'bright_1.50':lambda im,rng:np.clip(im.astype(float)*1.5,0,255).astype('uint8'),
        'gaussian_blur_5':lambda im,rng:cv2.GaussianBlur(im,(5,5),1.5),
        'gaussian_noise_15':lambda im,rng:np.clip(im.astype(float)+rng.normal(0,15,im.shape),0,255).astype('uint8'),
        'jpeg_quality_30':lambda im,rng:cv2.imdecode(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,30])[1],1),
        'dark_0.50_and_blur_5':lambda im,rng:cv2.GaussianBlur((im.astype(float)*.5).astype('uint8'),(5,5),1.5),
        'horizontal_flip':lambda im,rng:np.ascontiguousarray(im[:,::-1])}
    all_predictions={'clean':clean}
    for name,transform in variants.items():
        changed=[transform(im,rng) for im in images]
        rr=records
        if name=='horizontal_flip':
            rr=[{**r,'boxes':[[b[0],1-b[1],b[2],b[3],b[4]] for b in r['boxes']]} for r in records]
        pred=predict(changed); all_predictions[name]=pred
        result['corruptions'][name]=summary(rr,pred)
        result['corruptions'][name]['f1_delta_percentage_points']=100*(result['corruptions'][name]['f1']-result['corruptions']['clean']['f1'])
        print(name,result['corruptions'][name],flush=True)
    # Metamorphic relation: class/box consistency after flipping back.
    matches=0; original_count=0; flipped_count=0; exact=0
    for im,p,f in zip(images,clean,all_predictions['horizontal_flip']):
        pb=np.array(p['boxes']).reshape(-1,4); pc=np.array(p['cls']); ps=np.array(p['conf'])
        fb=np.array(f['boxes']).reshape(-1,4); fc=np.array(f['cls']); fs=np.array(f['conf'])
        fb[:,[0,2]]=im.shape[1]-fb[:,[2,0]]
        pb=pb[ps>=.25]; pc=pc[ps>=.25]; fb=fb[fs>=.25]; fc=fc[fs>=.25]
        ov=ious(pb,fb); used=set(); n=0
        for j,c in enumerate(pc):
            options=[k for k in range(len(fc)) if c==fc[k] and k not in used and ov[j,k]>=.5]
            if options: used.add(max(options,key=lambda k:ov[j,k])); n+=1
        matches+=n; original_count+=len(pb); flipped_count+=len(fb); exact+=int(n==len(pb)==len(fb))
    result['metamorphic']={'matched_boxes':matches,'clean_boxes':original_count,'flipped_boxes':flipped_count,
        'clean_detection_retention':matches/original_count,'symmetric_consistency_f1':2*matches/(original_count+flipped_count),
        'images_with_complete_consistency':exact,'images':len(images)}
    # Batch-one warmed latency, excluding image decode and resize, including API/pre/postprocess.
    latkw={**kw,'conf':.25}
    for im in images[:10]: model.predict(im,**latkw)
    times=[]; stages=[]
    torch.cuda.reset_peak_memory_stats()
    for im in images[:100]:
        torch.cuda.synchronize(); t=time.perf_counter(); r=model.predict(im,**latkw)[0]; torch.cuda.synchronize()
        times.append((time.perf_counter()-t)*1000); stages.append(r.speed)
    result['latency']={'images':100,'warmup_images':10,'batch':1,'mean_ms':float(np.mean(times)),
        'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'p99_ms':float(np.percentile(times,99)),
        'min_ms':float(min(times)),'max_ms':float(max(times)),'serial_images_per_second':1000/float(np.mean(times)),
        'mean_stages_ms':{k:float(np.mean([r[k] for r in stages])) for k in stages[0]},
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
    # Feed-forward model: clean / corrupted / restored input sequence.
    restored=[]
    for im in images[:20]:
        before=pack(model.predict(im,**latkw)[0]); model.predict((im*.5).astype('uint8'),**latkw)
        torch.cuda.synchronize(); t=time.perf_counter(); after=pack(model.predict(im,**latkw)[0]); torch.cuda.synchronize()
        same=before['cls']==after['cls'] and np.allclose(before['conf'],after['conf'],atol=1e-6) and np.allclose(before['boxes'],after['boxes'],atol=1e-4)
        restored.append({'restored':bool(same),'recovery_call_ms':(time.perf_counter()-t)*1000})
    result['resilience']={'trials':20,'restored_output_trials':sum(x['restored'] for x in restored),
        'mean_recovery_call_ms':float(np.mean([x['recovery_call_ms'] for x in restored]))}
    # Local post-hoc sensitivity: highest-confidence correct pothole in the random sample.
    choices=[]
    for i,(row,p) in enumerate(zip(records,clean)):
        _,scores,tps,_=match(row,p)
        keep=np.flatnonzero(np.asarray(p['conf'])>=.25)
        keep=keep[np.argsort(-np.array(p['conf'])[keep],kind='stable')]
        for j,tp in zip(keep,tps):
            if tp and int(p['cls'][j])==4: choices.append((p['conf'][j],i,j))
    if choices:
        score,i,j=max(choices); im=images[i]; box=np.array(clean[i]['boxes'][j]); h,w=im.shape[:2]; occluded=[]
        for yy in range(6):
            for xx in range(6):
                copy=im.copy(); copy[yy*h//6:(yy+1)*h//6,xx*w//6:(xx+1)*w//6]=114; occluded.append(copy)
        ps=predict(occluded); values=[]
        for p in ps:
            ov=ious([box],p['boxes'])[0]
            values.append(max([s for s,c,o in zip(p['conf'],p['cls'],ov) if int(c)==4 and o>=.5],default=0.))
        drops=(score-np.array(values)).reshape(6,6)
        np.save(OUT/'occlusion_map.npy',drops)
        cv2.imwrite(str(OUT/'occlusion_source.jpg'),im)
        result['interpretability']={'image':records[i]['name'],'class':'Pothole','baseline_confidence':score,
            'box':box.tolist(),'grid':[6,6],'mask_value':114,'max_confidence_drop':float(drops.max()),
            'mean_confidence_drop':float(drops.mean()),'min_confidence_drop':float(drops.min()),
            'selection':'highest-confidence correctly detected pothole in the random 300-image subset'}
    (OUT/'extended_results.json').write_text(json.dumps(result,indent=2))
    (OUT/'stress_predictions.json').write_text(json.dumps(all_predictions))
    print('Extended tests complete',flush=True)

if __name__=='__main__': main()
