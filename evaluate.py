"""Objective paired evaluation; subjective scores are collected separately."""
import csv
import json
import os
from pathlib import Path
import re
import librosa
import numpy as np
import onnxruntime as ort
import pyworld
import sherpa_onnx
import soundfile as sf
from src.control import ROOT


def normalize(text):
    return ''.join(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]', text)).lower()


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def read16(path):
    samples, sr = sf.read(path, dtype='float32')
    return librosa.resample(samples, orig_sr=sr, target_sr=16000)


class Evaluator:
    def __init__(self):
        models = Path('models')
        asr_dir = models / 'sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03'
        self.asr = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=str(asr_dir / 'model.int8.onnx'), tokens=str(asr_dir / 'tokens.txt'),
            num_threads=4, decoding_method='greedy_search')
        self.speaker = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(models / 'wespeaker_zh_cnceleb_resnet34.onnx'), num_threads=4))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.quality = ort.InferenceSession(str(models / 'sig_bak_ovr.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])

    def transcribe(self, samples):
        stream = self.asr.create_stream()
        stream.accept_waveform(16000, samples)
        self.asr.decode_stream(stream)
        return normalize(stream.result.text)

    def embedding(self, samples):
        stream = self.speaker.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=np.ascontiguousarray(samples))
        stream.input_finished()
        if not self.speaker.is_ready(stream):
            raise ValueError('Audio too short for speaker embedding')
        embedding = np.asarray(self.speaker.compute(stream))
        return embedding / max(np.linalg.norm(embedding), 1e-12)

    def quality_score(self, samples):
        # DNSMOS official non-personalized calibration and 9.01 s windows.
        window = 144160
        while len(samples) < window:
            samples = np.concatenate([samples, samples])
        values = []
        for start in range(0, len(samples) - window + 1, 16000):
            raw = self.quality.run(None, {'input_1': samples[None, start:start + window]})[0][0]
            values.append([np.polyval([-0.08397278, 1.22083953, 0.0052439], raw[0]),
                           np.polyval([-0.13166888, 1.60915514, -0.39604546], raw[1]),
                           np.polyval([-0.06766283, 1.11546468, 0.04602535], raw[2])])
        return np.mean(values, axis=0)


def acoustic_features(samples, sr):
    f0, _ = pyworld.harvest(samples.astype(np.float64), sr, f0_floor=65, f0_ceil=600)
    voiced = f0[f0 > 0]
    return dict(f0_median_hz=float(np.median(voiced)) if len(voiced) else None,
                rms_db=float(20 * np.log10(max(np.sqrt(np.mean(samples ** 2)), 1e-10))))


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    os.chdir(ROOT)
    output = ROOT / 'results'
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    evaluator = Evaluator()
    references = {item['id']: evaluator.embedding(read16(output / item['audio']))
                  for item in manifest if item['condition'] == 'whole'}
    metrics, segments = [], []
    for item in manifest:
        samples = read16(output / item['audio'])
        hypothesis = evaluator.transcribe(samples)
        reference = normalize(item['text'])
        distance = edit_distance(reference, hypothesis)
        sig, bak, ovr = evaluator.quality_score(samples)
        metrics.append(dict(id=item['id'], condition=item['condition'], text=item['text'],
                            asr_text=hypothesis, edit_distance=distance, characters=len(reference),
                            cer=distance / len(reference), dnsmos_sig=float(sig), dnsmos_bak=float(bak),
                            dnsmos_ovrl=float(ovr),
                            speaker_cosine=float(evaluator.embedding(samples) @ references[item['id']]),
                            seconds=item['seconds'], clip_ratio=float(np.mean(np.abs(samples) >= 0.999))))
        waveform, sr = sf.read(output / item['audio'], dtype='float32')
        for index, span in enumerate(item['spans']):
            clip = waveform[span['start_sample']:span['end_sample']]
            segments.append(dict(id=item['id'], condition=item['condition'], index=index,
                                 text=span['text'], target=span['target'], emotion=span['emotion'],
                                 requested_factor=span['requested_factor'], seconds=len(clip) / sr,
                                 base_seconds=span['base_samples'] / sr,
                                 measured_ratio=len(clip) / span['base_samples'],
                                 **acoustic_features(clip, sr)))
        print(item['id'], item['condition'], 'CER', round(distance / len(reference), 3), flush=True)
    write_csv(output / 'objective_metrics.csv', metrics)
    write_csv(output / 'segment_metrics.csv', segments)
    summary = {}
    for condition in ['whole', 'baseline', 'emotion', 'duration', 'combined']:
        rows = [r for r in metrics if r['condition'] == condition]
        summary[condition] = dict(n=len(rows),
            corpus_cer=sum(r['edit_distance'] for r in rows) / sum(r['characters'] for r in rows),
            **{key: float(np.mean([r[key] for r in rows]))
               for key in ['dnsmos_sig', 'dnsmos_ovrl', 'speaker_cosine', 'clip_ratio']})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
