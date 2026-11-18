"""Check generated audio, duration controls, and paired non-target samples."""
import json
import numpy as np
import soundfile as sf
from src.control import ROOT, PROFILES, SR


def main():
    result = ROOT / 'results'
    manifest = json.loads((result / 'manifest.json').read_text(encoding='utf-8'))
    lookup = {(r['id'], r['condition']): r for r in manifest}
    maximum_error, compared_segments, peaks = 0.0, 0, []
    for row in manifest:
        samples, sr = sf.read(result / row['audio'])
        assert sr == SR and samples.ndim == 1 and np.isfinite(samples).all()
        assert abs(len(samples) / sr - row['seconds']) < 1 / sr
        peaks.append(float(np.max(np.abs(samples))))
        assert peaks[-1] < 0.999, 'Clipping: ' + row['audio']
        if row['condition'] == 'whole':
            continue
        base = lookup[(row['id'], 'baseline')]
        base_samples, _ = sf.read(result / base['audio'])
        for span, original in zip(row['spans'], base['spans']):
            length = span['end_sample'] - span['start_sample']
            expected = span['requested_factor'] * PROFILES[span['emotion']]['duration']
            error = abs(length / span['base_samples'] - expected)
            maximum_error = max(maximum_error, error)
            assert error <= 1 / span['base_samples']
            if span['emotion'] == 'neutral' and span['requested_factor'] == 1:
                actual = samples[span['start_sample']:span['end_sample']]
                reference = base_samples[original['start_sample']:original['end_sample']]
                assert np.array_equal(actual, reference), 'Unexpected modification outside target'
                compared_segments += 1
    check = dict(audio_files=len(manifest), sample_rate=SR, maximum_peak=max(peaks),
                 maximum_duration_ratio_error=maximum_error,
                 identical_unmodified_segments=compared_segments,
                 passed=True)
    (result / 'verification.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    main()
