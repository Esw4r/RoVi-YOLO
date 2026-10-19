"""Prepare actual video sequences, distill YOLO pseudo-labels, train/evaluate an RNN.

Run from Gopher-AI: python -m vanilla_rnn.train
No human labels are supplied: all metrics measure agreement with the YOLO teacher.
"""
import argparse
import copy
import json
import random
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import CLASSES, ROOT, DEFAULT_CNN, CNNEncoder, VanillaRNN, FrameBaseline, sha256


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def prepare(args, out):
    paths = sorted(args.videos.glob('*.mp4'))
    if len(paths) != 8 or not all(p.name.startswith(('NO20221110-', 'NO20221119-')) for p in paths):
        raise ValueError('Default split expects the eight supplied recordings. Define/review a new split before changing the dataset.')
    # Label audit found no positives in the first three nighttime recordings.
    # Hold out complete daytime clips, and assess the separate nighttime session separately.
    splits = {p.name: ('night_test' if i < 4 else 'train' if i < 6 else 'val' if i == 6 else 'test')
              for i, p in enumerate(paths)}
    encoder = CNNEncoder(args.cnn)
    records = []
    cache = out / 'features'
    cache.mkdir(exist_ok=True)
    for p in paths:
        fingerprint = {'video_sha256': sha256(p), 'cnn_sha256': encoder.sha256,
                       'sample_fps': args.sample_fps, 'imgsz': 512, 'layer': 10,
                       'teacher_confidence': 0.25, 'teacher_iou': 0.7}
        dest = cache / (p.stem + '.npz')
        meta_path = dest.with_suffix('.json')
        if dest.exists() and meta_path.exists() and json.loads(meta_path.read_text())['fingerprint'] == fingerprint:
            meta = json.loads(meta_path.read_text())
            print('Using cached features:', p.name, flush=True)
        else:
            cap = cv2.VideoCapture(str(p))
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or fps < args.sample_fps:
                cap.release()
                raise ValueError(f'Cannot sample {p} at {args.sample_fps} FPS.')
            stride = fps / args.sample_fps
            if abs(stride - round(stride)) > 1e-6:
                cap.release()
                raise ValueError('Use a sample FPS that divides the source FPS exactly.')
            stride = round(stride)
            vectors, labels, timestamps, pending = [], [], [], []
            def flush():
                if not pending:
                    return
                results, features = encoder.predict(pending)
                vectors.extend(features.numpy())
                for r in results:
                    target = np.zeros(5, dtype=np.float32)
                    for cls in r.boxes.cls.cpu().numpy().astype(int):
                        target[cls] = 1
                    labels.append(target)
                pending.clear()
            try:
                for index in range(count):
                    if not cap.grab():
                        raise ValueError(f'Video decode ended early: {p.name}, frame {index}/{count}')
                    if index % stride:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise ValueError(f'Video frame decode failed: {p.name}:{index}')
                    pending.append(frame)
                    timestamps.append(index / fps)
                    if len(pending) == 8:
                        flush()
                    if len(timestamps) % 200 == 0:
                        print(f'{p.name}: {len(timestamps)} sampled frames', flush=True)
                flush()
            finally:
                cap.release()
            features, targets = np.asarray(vectors), np.asarray(labels)
            np.savez_compressed(dest, features=features, targets=targets, timestamps=np.asarray(timestamps))
            meta = {'fingerprint': fingerprint, 'source_fps': fps, 'source_frames': count,
                    'duration_seconds': count / fps, 'sampled_frames': len(labels),
                    'pseudo_positive_frames': targets.sum(0).astype(int).tolist()}
            write_json(meta_path, meta)
            print('Prepared', p.name, meta['pseudo_positive_frames'], flush=True)
        records.append({'video': p.name, 'split': splits[p.name], 'cache': str(dest.resolve()), **meta})
    manifest = {'label_source': 'unreviewed YOLO pseudo-labels; not ground truth', 'records': records,
                'split_notes': 'Chosen after auditing label availability: train first two Nov 19 clips, validate third, test fourth; all Nov 10 clips form a separate night_test. Daytime splits are from the same trip and may share road/scene context. No windows cross files. Independent-trip damage generalization is NOT established.'}
    write_json(out / 'manifest.json', manifest)
    return manifest, encoder.sha256


def windows(records, split, length):
    xs, ys = [], []
    for row in records:
        if row['split'] != split:
            continue
        with np.load(row['cache']) as data:
            f, targets = data['features'], data['targets']
            for end in range(length - 1, len(f)):
                xs.append(f[end-length+1:end+1])
                ys.append(targets[end])
    if not xs:
        raise ValueError(f'No usable {split} sequences.')
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


def metrics(logits, y):
    pred, truth = logits.sigmoid() >= 0.5, y.bool()
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    present = truth.any(0)
    return {'samples': len(y), 'macro_f1_present_classes': f1[present].mean().item() if present.any() else 0.0,
            'micro_f1': (2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()).clamp_min(1)).item(),
            'per_class': {name: {'support': int(truth[:, i].sum()), 'predicted_positive': int(pred[:, i].sum()),
                                'precision': precision[i].item(), 'recall': recall[i].item(), 'f1': f1[i].item()}
                          for i, name in enumerate(CLASSES)}}


@torch.inference_mode()
def evaluate(model, dataset):
    model.eval()
    outputs = torch.cat([model(x) for x, _ in DataLoader(TensorDataset(*dataset), batch_size=256)])
    return outputs, metrics(outputs, dataset[1])


def fit(model, train, val, args, name, out):
    torch.manual_seed(args.seed)
    positives = train[1].sum(0)
    supported = positives > 0
    if not supported.any():
        raise ValueError('YOLO found no positive training labels. Human annotation is required.')
    model.supported.copy_(supported)
    pos_weight = ((len(train[1]) - positives) / positives.clamp_min(1)).clamp(1, 100)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(*train), batch_size=128, shuffle=True)
    best_loss, best_f1, stale, history, best_state, best_epoch = float('inf'), -1.0, 0, [], None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)[:, supported].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(x)
        logits, m = evaluate(model, val)
        val_loss = loss_fn(logits, val[1])[:, supported].mean().item()
        entry = {'epoch': epoch, 'train_loss': loss_total/len(train[0]), 'val_loss': val_loss,
                 'val_teacher_agreement_macro_f1': m['macro_f1_present_classes']}
        history.append(entry)
        print(name, json.dumps(entry), flush=True)
        # Select for rare-damage agreement instead of an all-background loss minimum.
        score = m['macro_f1_present_classes']
        if score > best_f1 + 1e-6 or (abs(score - best_f1) <= 1e-6 and val_loss < best_loss - 1e-5):
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), val_loss, epoch, 0
            best_f1 = score
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    write_json(out / f'{name}_history.json', history)
    return best_epoch, supported.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=ROOT/'dashcam_recordings')
    parser.add_argument('--cnn', type=Path, default=DEFAULT_CNN)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'artifacts')
    parser.add_argument('--sample-fps', type=int, default=5)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.sample_fps, args.sequence_length, args.hidden_size, args.epochs, args.patience) < 1:
        parser.error('Sizes, rates, epochs, and patience must be positive.')
    if args.sequence_length < 2:
        parser.error('Temporal training requires at least two frames.')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    start = time.time()
    manifest, cnn_hash = prepare(args, args.output)
    train, val, test, night = [windows(manifest['records'], s, args.sequence_length)
                               for s in ('train', 'val', 'test', 'night_test')]
    # Each training frame contributes once to feature normalization, not once per overlapping window.
    raw_train = np.concatenate([np.load(r['cache'])['features'] for r in manifest['records'] if r['split'] == 'train'])
    mean, std = torch.from_numpy(raw_train.mean(0)), torch.from_numpy(raw_train.std(0).clip(1e-4))
    model = VanillaRNN(train[0].shape[-1], args.hidden_size)
    model.mean.copy_(mean)
    model.std.copy_(std)
    best_epoch, supported = fit(model, train, val, args, 'rnn', args.output)
    ckpt = {'format_version': 1, 'architecture': 'vanilla_rnn', 'state_dict': model.state_dict(),
            'classes': CLASSES, 'input_size': train[0].shape[-1], 'hidden_size': args.hidden_size,
            'sequence_length': args.sequence_length, 'sample_fps': args.sample_fps,
            'cnn_sha256': cnn_hash, 'cnn_layer': 10, 'imgsz': 512, 'supported_classes': supported,
            'label_source': 'yolo_pseudo_labels', 'best_epoch': best_epoch, 'seed': args.seed}
    torch.save(ckpt, args.output/'best.pt')
    torch.manual_seed(args.seed)
    baseline = FrameBaseline(train[0].shape[-1])
    baseline.mean.copy_(mean)
    baseline.std.copy_(std)
    baseline_epoch, _ = fit(baseline, train, val, args, 'frame_baseline', args.output)
    torch.save({'state_dict': baseline.state_dict(), 'input_size': train[0].shape[-1]}, args.output/'frame_baseline.pt')
    report = {'status': 'experimental_pseudo_label_training', 'metrics_meaning': 'Agreement with unreviewed YOLO labels, NOT independently measured damage accuracy.',
              'split_notes': manifest['split_notes'], 'training_sequences': len(train[0]),
              'training_positive_sequences': train[1].sum(0).int().tolist(), 'classes': CLASSES,
              'rnn_best_epoch': best_epoch, 'baseline_best_epoch': baseline_epoch,
              'selection': 'Validation macro F1 over positive-support classes; weighted BCE breaks ties. Positive weights capped at 100. Unsupported classes masked in both models.',
              'validation': evaluate(model, val)[1], 'test': evaluate(model, test)[1],
              'frame_only_baseline_test': evaluate(baseline, test)[1],
              'night_test': evaluate(model, night)[1],
              'elapsed_seconds': time.time()-start, 'cnn_sha256': cnn_hash,
              'checkpoint_sha256': sha256(args.output/'best.pt')}
    write_json(args.output/'metrics.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
