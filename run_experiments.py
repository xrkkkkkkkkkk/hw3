"""Generate paired conditions and sample-accurate segment annotations."""
import argparse
import json
import os
from pathlib import Path
import time
import soundfile as sf
from src.control import ROOT, SR, Synthesizer, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'cases.json')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    config_path = args.config.resolve()
    os.chdir(ROOT)
    cases = json.loads(config_path.read_text(encoding='utf-8'))
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases need unique nonempty IDs')
    synth = Synthesizer(args.threads)
    output = ROOT / 'results'
    (output / 'audio').mkdir(exist_ok=True)
    manifest = []
    for case in cases:
        if not case['segments'] or not all(s['text'].strip() for s in case['segments']):
            raise ValueError('All segments must have text')
        text = ''.join(s['text'] for s in case['segments'])
        for condition in ['whole', 'baseline', 'emotion', 'duration', 'combined']:
            start = time.perf_counter()
            if condition == 'whole':
                samples = synth.generate(text) * 0.55
                spans = []
            else:
                samples, spans = render(synth, case, condition)
            relative = 'audio/' + case['id'] + '_' + condition + '.wav'
            sf.write(output / relative, samples, SR, subtype='PCM_16')
            manifest.append(dict(id=case['id'], condition=condition, text=text, audio=relative,
                                 sample_rate=SR, seconds=len(samples) / SR, spans=spans,
                                 elapsed_seconds=time.perf_counter() - start))
            print(case['id'], condition, round(len(samples) / SR, 3), flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
