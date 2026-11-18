"""Build an offline blind listening page and experiment figures."""
import json
from pathlib import Path
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from src.control import ROOT


def main():
    output = ROOT / 'results'
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    references = {r['id']: r['audio'] for r in manifest if r['condition'] == 'whole'}
    trials = [dict(r) for r in manifest if r['condition'] != 'whole']
    random.Random(2026).shuffle(trials)
    key, blocks = [], []
    for index, row in enumerate(trials, 1):
        trial = f'T{index:02d}'
        key.append(dict(trial=trial, id=row['id'], condition=row['condition']))
        controls = ''.join(f'<label>{label}<select data-field="{field}" required><option value="">请选择</option>' +
                           ''.join(f'<option value="{n}">{n}</option>' for n in range(1, 6)) + '</select></label>'
                           for field, label in [('intelligibility', '可懂度'), ('naturalness', '自然度'), ('similarity', '音色相似度')])
        blocks.append(f'<section data-trial="{trial}"><h2>{trial}</h2><p>参考音频</p>'
                      f'<audio controls preload="none" src="{references[row["id"]]}"></audio><p>待评音频</p>'
                      f'<audio controls preload="none" src="{row["audio"]}"></audio><div>{controls}</div></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>语音主观试听</title>
<style>body{font:17px system-ui;max-width:900px;margin:36px auto;padding:0 20px;background:#f5f6f8;color:#172337}section{background:white;padding:22px;margin:20px 0;border-radius:12px}label{display:inline-block;margin:16px 24px 0 0}select,input,button{font:inherit;padding:7px}audio{width:100%}button{cursor:pointer}#status{color:#a33}table{border-collapse:collapse;background:white;width:100%;font-size:15px}th,td{border:1px solid #cbd0d8;padding:10px;text-align:left}li{margin:8px 0}</style>
<h1>语音主观试听</h1>
<p>请使用耳机，在安静环境中试听，并在整轮评测中保持相同的设备音量。</p>
<ol><li>先听待评音频，分别评价可懂度和自然度。可懂度关注词句是否听清，自然度关注声音、节奏、停顿和衔接是否像自然说话。</li>
<li>再交替播放参考音频和待评音频，评价是否像同一个人在说话。综合判断声音的厚薄、明亮或沙哑等音色特征。同一个人也可以改变语速、音高和情绪。</li>
<li>对照下表独立选择三项分数。例如，词句完全听清，但带有明显机械感，可以给可懂度 5 分、自然度 3 分。</li></ol>
<table><thead><tr><th>分数</th><th>可懂度</th><th>自然度</th><th>音色相似度</th></tr></thead><tbody>
<tr><td>5</td><td>每个词基本都能清楚辨认，理解毫不费力</td><td>接近自然人声，发音、节奏和衔接顺畅</td><td>非常像同一个人，音色特征高度一致</td></tr>
<tr><td>4</td><td>绝大部分听清，少量含糊，理解基本轻松</td><td>总体自然，有轻微机械感或局部不顺</td><td>比较像同一个人，有轻微音色变化</td></tr>
<tr><td>3</td><td>能懂大意，部分词需重听或猜测</td><td>机械感、停顿或衔接问题较明显，仍可接受</td><td>有一些相似特征，但音色变化明显</td></tr>
<tr><td>2</td><td>较多词听不清，理解比较费力</td><td>发音或节奏明显僵硬，衔接突兀或失真较重</td><td>相似程度较低，大部分声音特征有差异</td></tr>
<tr><td>1</td><td>几乎无法辨认词句或理解内容</td><td>声音严重失真、破碎，听感很不自然</td><td>基本不像同一个人</td></tr>
</tbody></table>
<p>评分按实际听感选择，可以重复播放。请用相同标准评完全部音频。每项均为整数 1–5 分，可以分多次填写，同一浏览器会保存进度。请完成全部 32 条后导出评分文件。</p>
<button type="button" id="reset">清空本页评分，重新评</button><p>此按钮清空当前页面保存的分数。已导出的评分文件仍可留作历史记录。</p>
<label>匿名听众编号 <input id="listener" placeholder="例如 L01" required></label>
''' + ''.join(blocks) + '''<button id="export">导出评分 JSON</button><p id="status"></p>
<script>
const storageKey='speech-study-v2';
function collect(){return {listener:document.querySelector('#listener').value.trim(),ratings:[...document.querySelectorAll('section')].map(s=>({trial:s.dataset.trial,...Object.fromEntries([...s.querySelectorAll('select')].map(e=>[e.dataset.field,Number(e.value)||null]))}))}}
function save(){localStorage.setItem(storageKey,JSON.stringify(collect()))}
document.addEventListener('change',save);
document.querySelector('#reset').onclick=()=>{for(const e of document.querySelectorAll('select'))e.value='';localStorage.removeItem(storageKey);document.querySelector('#status').textContent='已清空本页评分，可以开始新一轮。';};
try{const old=JSON.parse(localStorage.getItem(storageKey)||'null');if(old){document.querySelector('#listener').value=old.listener;for(const r of old.ratings){const s=document.querySelector('[data-trial="'+r.trial+'"]');if(s)for(const e of s.querySelectorAll('select'))e.value=r[e.dataset.field]||'';}}}catch(e){document.querySelector('#status').textContent='浏览器无法读取保存的进度，请在本次完成。'}
document.querySelector('#export').onclick=()=>{const data=collect();if(!data.listener||data.ratings.some(r=>['intelligibility','naturalness','similarity'].some(k=>!r[k]))){document.querySelector('#status').textContent='请填写听众编号并完成全部评分。';return;}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='ratings_'+data.listener+'.json';a.click();URL.revokeObjectURL(a.href);document.querySelector('#status').textContent='评分文件已导出。';};
</script></html>'''
    page = page.replace('全部 32 条', f'全部 {len(trials)} 条')
    (output / 'listening.html').write_text(page, encoding='utf-8')
    (output / 'blind_key.json').write_text(json.dumps(key, indent=2), encoding='utf-8')
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    # Show the complete utterance, including the annotated boundaries.
    example_id = 's05' if any(r['id'] == 's05' for r in manifest) else manifest[0]['id']
    selected = [r for r in manifest if r['id'] == example_id and r['condition'] != 'whole']
    limit = max(float(np.max(np.abs(sf.read(output / row['audio'])[0]))) for row in selected) * 1.2
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), constrained_layout=True)
    for ax, row in zip(axes, selected):
        samples, sr = sf.read(output / row['audio'])
        time = np.arange(len(samples)) / sr
        ax.plot(time, samples, linewidth=0.35, color='#286280')
        for span in row['spans']:
            if span['target']:
                ax.axvspan(span['start'], span['end'], color='#e2a53b', alpha=0.25)
        ax.set(ylabel=row['condition'], ylim=(-limit, limit))
    axes[-1].set_xlabel('Time (s), shaded area = controlled phrase')
    fig.savefig(figures / 'waveform_example.png', dpi=160)
    plt.close(fig)
    if not (output / 'segment_metrics.csv').exists() or not (output / 'summary.json').exists():
        return
    rows = list(csv.DictReader((output / 'segment_metrics.csv').open(encoding='utf-8-sig')))
    target = [r for r in rows if r['condition'] == 'duration' and r['target'] == 'True']
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    positions = np.arange(len(target))
    ax.bar(positions - 0.18, [float(r['requested_factor']) for r in target], width=0.36, label='Requested')
    ax.bar(positions + 0.18, [float(r['measured_ratio']) for r in target], width=0.36, label='Measured')
    ax.set(xticks=positions, xticklabels=[r['id'] for r in target], ylabel='Target duration / baseline duration')
    ax.legend()
    fig.savefig(figures / 'duration_control.png', dpi=160)
    plt.close(fig)
    summary = json.loads((output / 'summary.json').read_text())
    names = list(summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ['corpus_cer', 'dnsmos_ovrl', 'speaker_cosine'],
                                ['ASR character error rate', 'DNSMOS overall quality proxy', 'Speaker cosine vs whole utterance']):
        ax.bar(names, [summary[n][metric] for n in names], color='#286280')
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', rotation=35)
    fig.savefig(figures / 'objective_comparison.png', dpi=160)
    plt.close(fig)
    subjective_path = output / 'subjective_summary.csv'
    if subjective_path.exists():
        ratings = list(csv.DictReader(subjective_path.open(encoding='utf-8-sig')))
        conditions = ['baseline', 'emotion', 'duration', 'combined']
        positions = np.arange(len(conditions))
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for offset, metric in zip([-0.25, 0, 0.25], ['intelligibility', 'naturalness', 'similarity']):
            lookup = {r['condition']: float(r['mean']) for r in ratings if r['metric'] == metric}
            ax.bar(positions + offset, [lookup[c] for c in conditions], width=0.25, label=metric)
        ax.set(xticks=positions, xticklabels=conditions, ylabel='Mean listener score (1-5)', ylim=(0, 5.5))
        ax.legend(loc='upper right', fontsize=8)
        fig.savefig(figures / 'subjective_comparison.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
