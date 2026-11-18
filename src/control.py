"""Chinese VITS baseline with explicit segment-level prosody controls."""
from pathlib import Path
import hashlib
import librosa
import numpy as np
import pyworld
import sherpa_onnx

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'neutral': dict(semitones=0.0, energy_db=0.0, duration=1.0, pitch_range=1.0),
    'excited': dict(semitones=2.0, energy_db=2.0, duration=0.90, pitch_range=1.20),
    'sad': dict(semitones=-1.5, energy_db=-2.0, duration=1.15, pitch_range=0.80),
}
SR = 22050


class Synthesizer:
    def __init__(self, threads=4):
        folder = Path('models') / 'vits-melo-tts-zh_en'
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(folder / 'model.onnx'),
                    lexicon=str(folder / 'lexicon.txt'),
                    tokens=str(folder / 'tokens.txt'),
                    noise_scale=0.0, noise_scale_w=0.0),
                num_threads=threads, provider='cpu'),
            rule_fsts=','.join(str(folder / f) for f in ['date.fst', 'number.fst', 'phone.fst']))
        if not config.validate():
            raise RuntimeError('TTS model configuration invalid. Run download_models.py first.')
        self.tts = sherpa_onnx.OfflineTts(config)
        self.cache = ROOT / 'results' / 'cache'
        self.cache.mkdir(parents=True, exist_ok=True)

    def generate(self, text):
        # Cache floating point samples to keep paired experiments sample-identical.
        key = hashlib.sha256(('melo-v1-noise0-sid0-sr22050|' + text).encode()).hexdigest()
        path = self.cache / (key + '.npy')
        if path.exists():
            return np.load(path)
        audio = self.tts.generate(text, sid=0, speed=1.0)
        samples = librosa.resample(np.asarray(audio.samples), orig_sr=audio.sample_rate, target_sr=SR)
        # Remove only outer low-energy padding. Internal pauses are retained.
        samples, _ = librosa.effects.trim(samples, top_db=38)
        if len(samples) < SR // 20:
            raise ValueError('Empty/too short synthesized text: ' + text)
        samples = samples.astype(np.float32)
        np.save(path, samples)
        return samples


def prosody(samples, profile, duration_factor=1.0):
    if profile not in PROFILES:
        raise ValueError('Unknown emotion: ' + profile)
    if not 0.5 <= duration_factor <= 2.0:
        raise ValueError('duration_factor must be between 0.5 and 2.0')
    p = PROFILES[profile]
    output = samples.copy()
    if p['semitones']:
        waveform = samples.astype(np.float64)
        f0, times = pyworld.harvest(waveform, SR, f0_floor=65, f0_ceil=600)
        envelope = pyworld.cheaptrick(waveform, f0, times, SR)
        aperiodicity = pyworld.d4c(waveform, f0, times, SR)
        voiced = f0 > 0
        if voiced.any():
            center = np.median(np.log(f0[voiced]))
            f0[voiced] = np.exp(center + p['pitch_range'] * (np.log(f0[voiced]) - center))
            f0[voiced] *= 2 ** (p['semitones'] / 12)
        output = pyworld.synthesize(f0, envelope, aperiodicity, SR).astype(np.float32)
        output = librosa.util.fix_length(output, size=len(samples))
    factor = p['duration'] * duration_factor
    if abs(factor - 1) > 1e-8:
        output = librosa.effects.time_stretch(output, rate=1 / factor)
    output = librosa.util.fix_length(output, size=round(len(samples) * factor))
    return output * (10 ** (p['energy_db'] / 20))


def assemble(segments, gap_ms=45):
    """Fixed-length boundary fades preserve the annotated target durations."""
    parts, spans = [], []
    position = 0
    gap = np.zeros(round(SR * gap_ms / 1000), dtype=np.float32)
    for index, segment in enumerate(segments):
        samples = segment['samples'].copy()
        fade = min(round(SR * 0.008), len(samples) // 4)
        samples[:fade] *= np.linspace(0, 1, fade)
        samples[-fade:] *= np.linspace(1, 0, fade)
        span = {k: v for k, v in segment.items() if k != 'samples'}
        span.update(start_sample=position, end_sample=position + len(samples),
                    start=position / SR, end=(position + len(samples)) / SR)
        spans.append(span)
        parts.append(samples)
        position += len(samples)
        if index < len(segments) - 1:
            parts.append(gap)
            position += len(gap)
    waveform = np.concatenate(parts)
    # Fixed shared gain is used in all conditions. No per-file normalization.
    return waveform * 0.55, spans


def render(synthesizer, case, condition):
    segments = []
    for item in case['segments']:
        emotion = item.get('emotion', 'neutral') if condition in ('emotion', 'combined') else 'neutral'
        factor = case.get('duration_factor', 1.0) if item.get('target') and condition in ('duration', 'combined') else 1.0
        original = synthesizer.generate(item['text'])
        controlled = prosody(original, emotion, factor)
        segments.append(dict(text=item['text'], target=item.get('target', False), emotion=emotion,
                             requested_factor=factor, base_samples=len(original), samples=controlled))
    return assemble(segments)
