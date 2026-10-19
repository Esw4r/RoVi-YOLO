"""Fresh full-test-set validation; preserve per-image predictions for audit."""
from pathlib import Path
import json, platform, sys, time
import numpy as np
import torch, ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/evaluation'

class CaptureValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.captured = []
        CaptureValidator.latest = self

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            b = self._prepare_batch(si, batch)
            p = self.scale_preds(self._prepare_pred(pred), b)
            self.captured.append({'name':Path(b['im_file']).name,
                'boxes':p['bboxes'].cpu().tolist(), 'conf':p['conf'].cpu().tolist(),
                'cls':p['cls'].cpu().tolist()})
        super().update_metrics(preds, batch)

def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    cfg = OUT/'data.yaml'
    cfg.write_text('path: '+(ROOT/'rdd2022/RDD_SPLIT').as_posix()+'\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [longitudinal crack, transverse crack, alligator crack, other corruption, Pothole]\n')
    model = YOLO(str(ROOT/'gopher-ai_v1.pt'))
    meta = {'python':sys.version, 'torch':torch.__version__, 'ultralytics':ultralytics.__version__,
            'platform':platform.platform(), 'device':torch.cuda.get_device_name(0),
            'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'parameter_count':sum(p.numel() for p in model.model.parameters()),
            'model_file_bytes':(ROOT/'gopher-ai_v1.pt').stat().st_size,
            'checkpoint_epoch_zero_based':model.ckpt.get('epoch'), 'checkpoint_date':model.ckpt.get('date'),
            'checkpoint_ultralytics':model.ckpt.get('version'), 'train_args':model.ckpt.get('train_args'),
            'names':model.names}
    (OUT/'model_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    start=time.perf_counter()
    metrics=model.val(validator=CaptureValidator,data=str(cfg),split='test',imgsz=512,
        batch=8,device=0,half=False,workers=0,conf=0.001,iou=0.7,max_det=300,
        augment=False,plots=True,save_json=False,project=str(OUT),name='baseline',exist_ok=True,
        cache=False,verbose=True,seed=23022)
    # Persist raw evidence before converting NumPy scalar metrics to JSON.
    (OUT/'baseline_predictions.json').write_text(json.dumps(CaptureValidator.latest.captured),encoding='utf-8')
    result={'metrics':metrics.results_dict,'per_class':metrics.summary(), 'speed_ms':metrics.speed,
            'wall_seconds':time.perf_counter()-start,'images':len(CaptureValidator.latest.captured)}
    def numeric(value):
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,np.ndarray): return value.tolist()
        raise TypeError(type(value).__name__)
    (OUT/'baseline_metrics.json').write_text(json.dumps(result,indent=2,default=numeric),encoding='utf-8')
    print(json.dumps(result,indent=2,default=numeric),flush=True)

if __name__=='__main__': main()
