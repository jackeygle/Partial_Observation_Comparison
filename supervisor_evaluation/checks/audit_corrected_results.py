"""Check final corrected results against the thesis and record artifact provenance."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'supervisor_evaluation/outputs/full'
PAPER = ROOT.parent / 'Thesis_writing/Thesis_paper20260219'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    thesis = (PAPER / 'thesis.tex').read_text()
    accuracy = list(csv.DictReader((OUT / 'accuracy.csv').open()))
    uncertainty = list(csv.DictReader((OUT / 'uncertainty.csv').open()))
    channels = list(csv.DictReader((OUT / 'uncertainty_by_channel.csv').open()))
    timing = list(csv.DictReader((OUT / 'inference_time_controlled.csv').open()))
    selection = json.loads((OUT / 'corrected_model_selection.json').read_text())
    summary = json.loads((OUT / 'breakdown/summary.json').read_text())
    differences = json.loads((OUT / 'breakdown/accuracy_consistency.json').read_text())
    checks = []
    for key, epoch in [('varnet_mse', 76), ('varnet_aughead', 80)]:
        x = selection[key]
        assert x['selected_epoch'] == epoch and len(x['days']) == 7
        assert x['selected_mse'] == min(x['mse_by_epoch'].values())
        cp = Path(x['run_dir']) / x['selected_ckpt']
        assert sha(cp) == x['checkpoint_sha256']
    manifest = json.loads((ROOT / 'supervisor_evaluation/models/manifest.json').read_text())
    assert manifest == json.loads((OUT / 'model_manifest.json').read_text())
    for name, entry in manifest['files'].items():
        assert sha(ROOT / 'supervisor_evaluation/models' / name) == entry['sha256'], name
    for key, name in [('varnet_mse', 'varnet_mse_fold_s3.pt'),
                      ('varnet_aughead', 'varnet_aughead_fold_s0.pt')]:
        assert manifest['files'][name]['sha256'] == selection[key]['checkpoint_sha256']
    checks.append('validation-only selection, frozen winners and all eight packaged artifact checksums')
    accuracy_names = ['Senseiver-A', 'Senseiver-G (ours)', 'DINCAE', '4DVarNet',
                      r'4DVarNet aug.\ var.\ (ours)', 'EnKF']
    fields = ['blind_rmse', 'density_rmse', 'vx_rmse', 'vy_rmse', 'variance_rmse',
              'all_walkable_rmse']
    block = thesis[thesis.index(r'\label{tab:accuracy}'):]
    block = block[:block.index(r'\end{table}')]
    for name, row in zip(accuracy_names, accuracy):
        line = next(l for l in block.splitlines() if l.strip().startswith(name + ' '))
        values = re.findall(r'(?<!\d)0\.\d{3}(?!\d)', line)
        assert values == [f'{float(row[k]):.3f}' for k in fields], (name, values)
    checks.append('all six accuracy table rows match accuracy.csv')
    block = thesis[thesis.index(r'\label{tab:uncertainty}'):]
    block = block[:block.index(r'\end{table}')]
    by_method = {r['method']: r for r in uncertainty}
    order = ['DINCAE', '4DVarNet aughead_obs (single)', 'Localized EnKF (100 members)']
    for metric, ndigits in [('crps', 3), ('spread_rmse', 2)]:
        line = next(l for l in block.splitlines() if l.strip().startswith(
            'CRPS' if metric == 'crps' else 'Spread/RMSE'))
        values = re.findall(r'(?<!\d)\d+\.\d+(?!\d)', line)
        expected = [f'{float(by_method[m][k]):.{ndigits}f}'
                    for m in order for k in [metric, metric+'_lo', metric+'_hi']]
        assert values == expected, (metric, values, expected)
    for metric, ndigits in [('crps', 3), ('spread_rmse', 2)]:
        start = block.index('CRPS' if metric == 'crps' else 'Spread/RMSE')
        part = block[start:]
        for channel, label in [('density', r'$\rho$'), ('vx', '$v_x$'),
                               ('vy', '$v_y$'), ('variance', r'$\nu$')]:
            line = next(l for l in part.splitlines() if label in l)
            values = re.findall(r'(?<!\d)\d+\.\d+(?!\d)', line)
            key = 'crps' if metric == 'crps' else 'spread_rmse'
            expected = [f"{float(next(r for r in channels if r['method']==m and r['channel']==channel)[key]):.{ndigits}f}"
                        for m in order]
            assert values == expected, (metric, channel, values, expected)
    checks.append('pooled uncertainty intervals and all channel table rows match CSV')
    block = thesis[thesis.index(r'\label{tab:occupancy}'):]
    block = block[:block.index(r'\end{table}')]
    for name, key in zip(accuracy_names, summary['methods']):
        line = next(l for l in block.splitlines() if l.strip().startswith(name+' '))
        values = re.findall(r'(?<!\d)0\.\d{3}(?!\d)', line)
        assert values == [f'{v:.3f}' for v in summary['density_rmse_empty_occupied'][key]]
        share = int(re.search(r'(\d+)\\%', line).group(1))
        assert share == round(100*summary['density_se_share_occupied'][key])
    checks.append('all occupancy table rows match common-support breakdown')
    assert all(abs(d) <= 5e-4 for d in differences.values()), differences
    checks.append('independent native-support breakdown reproduces accuracy within 0.0005 RMSE')
    assert len({r['gpu'] for r in timing}) == 1
    assert all(int(r['repeats']) == 3 for r in timing)
    checks.append('six-method timing uses one GPU and three repetitions')
    clean = re.sub(r'(?<!\\)%[^\n]*', '', thesis)
    labels = re.findall(r'\\label\{([^}]+)\}', clean)
    refs = re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', clean)
    assert not (set(refs)-set(labels)), sorted(set(refs)-set(labels))
    assert all(n==1 for n in Counter(labels).values())
    begins = Counter(re.findall(r'\\begin\{([^}]+)\}', clean))
    ends = Counter(re.findall(r'\\end\{([^}]+)\}', clean))
    assert begins == ends
    depth = 0
    for ch in re.sub(r'\\[{}]', '', clean):
        if ch=='{': depth+=1
        if ch=='}': depth-=1
        assert depth>=0
    assert depth==0
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', clean)
    assert all((PAPER / p).is_file() for p in figures)
    checks.append('LaTeX references, labels, environment counts, braces and 17 figure paths')
    asset_map = {
        'accuracy.pdf': OUT/'figures/accuracy_rmse.pdf',
        'uncertainty.pdf': OUT/'figures/uncertainty_summary.pdf',
        'latency.pdf': OUT/'figures/inference_latency.pdf',
        'breakdown.pdf': ROOT/'thesis_figures/out/breakdown.pdf',
        'comparison.pdf': ROOT/'thesis_figures/out/comparison_atc-20130811_32778.pdf',
        'uncertainty_maps.pdf': ROOT/'thesis_figures/out/uncertainty_maps_atc-20130811_32778.pdf',
        'varnet.pdf': ROOT/'thesis_figures/out/varnet.pdf',
        'dincae.pdf': ROOT/'thesis_figures/out/dincae.pdf',
    }
    hashes = {}
    for name, source in asset_map.items():
        assert sha(source) == sha(PAPER/'figures'/name), name
        hashes[name] = sha(source)
    checks.append(f'all {len(asset_map)} refreshed thesis PDFs match regenerated source assets')
    report = dict(checks=checks, validation_selection=selection,
                  accuracy=accuracy, uncertainty=uncertainty, timing=timing,
                  breakdown_accuracy_differences=differences,
                  refreshed_figure_sha256=hashes, n_thesis_figures=len(figures),
                  pdf_compiled=False, new_training_performed=False,
                  new_baseline_or_jitter_experiments=False)
    target = OUT/'corrected_adoption_audit.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS: {len(checks)} final checks -> {target}')


if __name__=='__main__':
    main()
