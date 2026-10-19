"""Shared training/inference contract. Temporal scores do not replace YOLO boxes."""
from collections import deque
from pathlib import Path
import hashlib
import math
import threading

import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

CLASSES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'other corruption', 'Pothole']
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CNN = ROOT / 'YOLO/gopher-ai_v1.pt'
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / 'artifacts/best.pt'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class CNNEncoder:
    """Capture pooled backbone features and detections in the same forward pass."""
    def __init__(self, weights=DEFAULT_CNN, device=None, imgsz=512, layer=10):
        self.path = Path(weights).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256(self.path)
        self.yolo = YOLO(str(self.path))
        if list(self.yolo.names.values()) != CLASSES:
            raise ValueError('CNN checkpoint class order does not match Gopher AI.')
        self.device = device or ('0' if torch.cuda.is_available() else 'cpu')
        self.imgsz, self.layer = imgsz, layer
        self.lock = threading.RLock()
        self._features = None
        self._hook = self.yolo.model.model[layer].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._features = output.detach().float().mean(dim=(-2, -1)).cpu()

    @torch.inference_mode()
    def predict(self, frames):
        with self.lock:
            self._features = None
            results = self.yolo.predict(frames, imgsz=self.imgsz, rect=False,
                                        conf=0.25, iou=0.7, device=self.device,
                                        quantize=32, verbose=False)
            if self._features is None or len(self._features) != len(results):
                raise RuntimeError('CNN feature capture failed.')
            return results, self._features.clone()


class VanillaRNN(nn.Module):
    def __init__(self, input_size=256, hidden_size=64, num_classes=5):
        super().__init__()
        self.register_buffer('mean', torch.zeros(input_size))
        self.register_buffer('std', torch.ones(input_size))
        self.register_buffer('supported', torch.ones(num_classes, dtype=torch.bool))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1,
                          nonlinearity='tanh', batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        output, _ = self.rnn(x)  # Fresh hidden state for each fixed-length window.
        return self.head(output[:, -1]).masked_fill(~self.supported, -20.0)


class FrameBaseline(nn.Module):
    """Same CNN features, only the final frame; measures whether history helps."""
    def __init__(self, input_size=256):
        super().__init__()
        self.register_buffer('mean', torch.zeros(input_size))
        self.register_buffer('std', torch.ones(input_size))
        self.register_buffer('supported', torch.ones(5, dtype=torch.bool))
        self.head = nn.Linear(input_size, 5)

    def forward(self, x):
        return self.head((x[:, -1] - self.mean) / self.std).masked_fill(~self.supported, -20.0)


class TemporalPredictor:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, cnn_hash=None):
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if ckpt.get('format_version') != 1 or ckpt.get('architecture') != 'vanilla_rnn':
            raise ValueError('Unsupported temporal checkpoint.')
        if cnn_hash and ckpt['cnn_sha256'] != cnn_hash:
            raise ValueError('RNN checkpoint requires the CNN used during training.')
        if ckpt['classes'] != CLASSES:
            raise ValueError('RNN class order mismatch.')
        self.config = ckpt
        self.model = VanillaRNN(ckpt['input_size'], ckpt['hidden_size'])
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def new_window(self):
        return TemporalWindow(self)


class TemporalWindow:
    """One window per stream. Never share temporal history across connections."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.length = predictor.config['sequence_length']
        self.interval = 1.0 / predictor.config['sample_fps']
        self.frames = deque(maxlen=self.length)
        self.last_timestamp = None
        self.latest = None

    def reset(self):
        self.frames.clear()
        self.last_timestamp = None
        self.latest = None

    @torch.inference_mode()
    def push(self, feature, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Frame timestamp must be finite.')
        if self.last_timestamp is not None:
            elapsed = timestamp - self.last_timestamp
            if elapsed < 0 or elapsed > self.interval * 2.5:
                self.reset()
            elif elapsed < self.interval - 1e-4:
                return self.latest or self.status()
        self.frames.append(torch.as_tensor(feature).float().cpu())
        self.last_timestamp = timestamp
        if len(self.frames) < self.length:
            return self.status()
        scores = self.predictor.model(torch.stack(list(self.frames))[None]).sigmoid()[0].tolist()
        supported = self.predictor.config['supported_classes']
        self.latest = {**self.status(), 'ready': True, 'scores': [
            {'class_id': i, 'class_name': name, 'score': scores[i] if supported[i] else None,
             'supported': supported[i], 'active': bool(supported[i] and scores[i] >= 0.5)}
            for i, name in enumerate(CLASSES)]}
        return self.latest

    def status(self):
        return {'ready': False, 'frames_collected': len(self.frames), 'frames_required': self.length,
                'label_source': 'yolo_pseudo_labels', 'experimental': True, 'scores': []}
