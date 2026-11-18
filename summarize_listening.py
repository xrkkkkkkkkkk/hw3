"""Aggregate real listening responses only. No synthetic ratings are generated."""
import argparse
import json
from collections import defaultdict
import numpy as np
from evaluate import write_csv
from src.control import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='Exported listener JSON files')
    args = parser.parse_args()
    keys = {r['trial']: r for r in json.loads((ROOT / 'results/blind_key.json').read_text())}
    data = defaultdict(list)
    listeners = set()
    for path in args.files:
        with open(path, encoding='utf-8') as stream:
            result = json.load(stream)
        listener = result['listener'].strip()
        if not listener or listener in listeners:
            raise ValueError('Empty or duplicate listener ID')
        listeners.add(listener)
        if {r['trial'] for r in result['ratings']} != set(keys) or len(result['ratings']) != len(keys):
            raise ValueError('Each listener must complete all trials exactly once')
        for row in result['ratings']:
            for field in ['intelligibility', 'naturalness', 'similarity']:
                score = row[field]
                if type(score) is not int or not 1 <= score <= 5:
                    raise ValueError('Scores must be integers from 1 to 5')
                data[(keys[row['trial']]['condition'], field, listener)].append(score)
    output = []
    for condition in ['baseline', 'emotion', 'duration', 'combined']:
        for field in ['intelligibility', 'naturalness', 'similarity']:
            means = [float(np.mean(data[(condition, field, name)])) for name in sorted(listeners)]
            output.append(dict(condition=condition, metric=field, listeners=len(listeners),
                               mean=float(np.mean(means)),
                               listener_mean_std=float(np.std(means, ddof=1)) if len(means) > 1 else ''))
    write_csv(ROOT / 'results/subjective_summary.csv', output)
    from make_review import main as make_review
    make_review()
    from build_report import main as build_report
    build_report()
    print('Saved real listener scores for', len(listeners), 'listeners')


if __name__ == '__main__':
    main()
