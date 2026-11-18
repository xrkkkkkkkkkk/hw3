"""Download the public models, retain licenses, and reject unsafe archive paths."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
SOURCES = {
    'tts': ('https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2', 'vits-melo-tts-zh_en/model.onnx'),
    'asr': ('https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2', 'sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx'),
    'speaker': ('https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx', 'wespeaker_zh_cnceleb_resnet34.onnx'),
    'quality': ('https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx', 'sig_bak_ovr.onnx'),
}


def download(url, target):
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out, 1024 * 1024)
            partial.replace(target)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def check_member(name):
    target = (MODELS / name).resolve()
    if MODELS.resolve() not in target.parents:
        raise ValueError('Unsafe archive member: ' + name)


def main():
    MODELS.mkdir(exist_ok=True)
    records = []
    for name, (url, expected) in SOURCES.items():
        archive = MODELS / url.rsplit('/', 1)[1]
        if not (MODELS / expected).exists():
            print('Downloading ' + name, flush=True)
            if not archive.exists():
                download(url, archive)
            if archive.name.endswith('.tar.bz2'):
                with tarfile.open(archive) as package:
                    for member in package.getmembers():
                        check_member(member.name)
                        if member.issym() or member.islnk():
                            raise ValueError('Archive links are unsupported')
                    package.extractall(MODELS)
            elif archive.suffix == '.zip':
                with zipfile.ZipFile(archive) as package:
                    for member in package.namelist():
                        check_member(member)
                    package.extractall(MODELS)
        model = MODELS / expected
        if not model.is_file():
            raise RuntimeError('Missing model: ' + str(model))
        records.append(dict(name=name, url=url, file=expected,
                            sha256=hashlib.sha256(model.read_bytes()).hexdigest()))
        print('Ready: ' + expected, flush=True)
    (MODELS / 'sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
