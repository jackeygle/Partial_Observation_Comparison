"""Validation selection and batch=1 test inference for the new 200-frame run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from crowdcore import frame_reliability as fr, navigation as nav, observation_model as om
from methods.varnet.train import build_windows
from methods.varnet.checks.model_io import load_solver
from methods.varnet.checks.select_checkpoint import score_day
from methods.varnet.checks.eval_comprehensive import clip_bounds


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('GPU required')
    a.out.mkdir(parents=True, exist_ok=True)
    walk = nav.build_valid_mask_from_config()
    valid = om.split_files('valid')
    print(f'[frames] {fr.describe()}', flush=True)
    cp = a.out / ('validation_cache_all_walkable' + ('_gapmask' if fr.ENABLED else '') + '.json')
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    candidates = sorted(set(a.run_dir.glob('ckpt_*.pt')) |
                        {a.run_dir / 'varnet_last.pt', a.run_dir / 'varnet_best.pt'})
    scores = []
    for path in candidates:
        if not path.exists():
            continue
        ck = torch.load(path, map_location='cpu')
        if ck.get('n_iter_eff') != 20:
            continue
        args = ck['args']
        assert args['dT'] == 200 and args['kt'] == 3
        assert args['normg_scope'] == 'window' and args['fold_axes'] == 'ct'
        sha = digest(path)
        epoch = ck['epoch']
        del ck
        solver, args, _ = load_solver(str(path), 'cuda')
        solver.requires_grad_(False)
        for fp in valid:
            key = sha + ':' + Path(fp).name
            if key not in cache:
                se, n = score_day({epoch: solver}, fp, {str(path): args}, walk, 'cuda', 1,
                                  'all_walkable')[epoch]
                cache[key] = [se, n]
                save(cp, cache)
            print(f'[valid] epoch={epoch} day={Path(fp).name}', flush=True)
        se = sum(cache[sha + ':' + Path(fp).name][0] for fp in valid)
        n = sum(cache[sha + ':' + Path(fp).name][1] for fp in valid)
        scores.append(dict(checkpoint=str(path), sha256=sha, epoch=epoch, mse=se/n))
        del solver
        torch.cuda.empty_cache()
    if not scores:
        raise SystemExit('No trained K=20 candidate: refusing premature evaluation')
    winner = min(scores, key=lambda x: (x['mse'], x['epoch'], x['checkpoint']))
    save(a.out / 'selection.json', dict(split='valid', scope='all_walkable', batch=1, window_frames=200,
         days=[Path(fp).name for fp in valid], frames=fr.describe(), candidates=scores, selected=winner))
    print(f'[selected] {winner}', flush=True)
    solver, args, _ = load_solver(winner['checkpoint'], 'cuda')
    solver.requires_grad_(False)
    daily = []
    for fp in om.split_files('test'):
        day = Path(fp).stem
        result = a.out / f'{day}_metrics{"_gapmask" if fr.ENABLED else ""}.json'
        if result.exists():
            row = json.loads(result.read_text())
            if row['checkpoint_sha256'] == winner['sha256']:
                daily.append(row)
                continue
        seed = args['seed'] if args.get('data_seed') is None else args['data_seed']
        X, Y, M, X0 = build_windows([fp], 200, args['sensing_range'], args['num_agents'],
            seed, 1, add_noise=False if args.get('no_noise') else None,
            obs_every_k=args.get('obs_every_k'), trajectory_mode=args['trajectory_mode'])
        se = np.zeros(4, dtype=np.float64)
        n = np.zeros(4, dtype=np.int64)
        all_se = np.zeros(4, dtype=np.float64)
        all_n = np.zeros(4, dtype=np.int64)
        assert fr.BLOCK == 200                     # a 4DVarNet window is one block
        keep_w = fr.scorable_frames(fp, len(X) * 200).reshape(len(X), 200)[:, 0]
        example = int(np.flatnonzero(keep_w)[keep_w.sum() // 2])
        for i in range(len(X)):
            if not keep_w[i]:
                continue
            with torch.enable_grad():
                p = solver(*(t[i:i+1].to('cuda') for t in (X0, Y, M))).detach().cpu().numpy()
            if not np.isfinite(p).all():
                raise RuntimeError(f'Nonfinite reconstruction: {day} window {i}')
            p = clip_bounds(p)
            truth = X[i:i+1].numpy()
            sq = (p.astype(np.float64) - truth) ** 2
            blind = (M[i:i+1].numpy() < .5) & walk[None, None, None]
            for c in range(4):
                se[c] += sq[:, c][blind[:, c]].sum()
                n[c] += blind[:, c].sum()
                all_se[c] += sq[:, c][:, :, walk].sum()
                all_n[c] += 200 * int(walk.sum())
            if i == example:
                np.savez_compressed(a.out / f'{day}_example.npz', prediction=p[0],
                    truth=truth[0], observations=Y[i].numpy(), mask=M[i].numpy())
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(8, 7))
                for ax, field, title in zip(axs, [truth[0,0,100], p[0,0,100],
                    np.abs(p[0,0,100]-truth[0,0,100])], ['Truth', '4DVarNet', 'Absolute error']):
                    im = ax.imshow(field, origin='lower', vmin=0)
                    ax.set_title(title)
                    fig.colorbar(im, ax=ax, shrink=.6)
                fig.tight_layout()
                fig.savefig(a.out / f'{day}_example.png', dpi=150)
                plt.close(fig)
            if i % 50 == 0:
                print(f'[test] {day} {i+1}/{len(X)} windows; batch=1', flush=True)
        row = dict(day=day, checkpoint_sha256=winner['sha256'], frames=int(keep_w.sum())*200,
                   squared_error=se.tolist(), counts=n.tolist(),
                   all_squared_error=all_se.tolist(), all_counts=all_n.tolist())
        save(result, row)
        daily.append(row)
        del X, Y, M, X0
    se = np.sum([r['squared_error'] for r in daily], axis=0)
    n = np.sum([r['counts'] for r in daily], axis=0)
    ase = np.sum([r['all_squared_error'] for r in daily], axis=0)
    an = np.sum([r['all_counts'] for r in daily], axis=0)
    metrics = dict(batch=1, window_frames=200, selected=winner, per_day=daily,
        blind_rmse=np.sqrt(se/n).tolist(), pooled_blind_rmse=float(np.sqrt(se.sum()/n.sum())),
        pooled_all_walkable_rmse=float(np.sqrt(ase.sum()/an.sum())))
    save(a.out / 'test_summary.json', metrics)
    with (a.out / 'accuracy.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['method', 'All', 'density', 'vx', 'vy', 'variance', 'All walkable'])
        w.writerow(['4DVarNet kt3 window-norm batch1', metrics['pooled_blind_rmse'],
                    *metrics['blind_rmse'], metrics['pooled_all_walkable_rmse']])
    save(a.out / 'status.json', dict(status='complete', **metrics))
    print('[complete]', metrics['pooled_blind_rmse'], flush=True)


if __name__ == '__main__':
    main()
