"""Reselect saved reconstruction models on all walkable validation cells.

Writes separate results; never replaces the paper's frozen assets or selections.
With the gap mask on (crowdcore.frame_reliability; default under DATA_VARIANT=all92) only
scorable frames are scored, into their own output directory (suffix _gapmask).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from crowdcore import config, frame_reliability as fr, navigation as nav, observation_model as om
from supervisor_evaluation.checks.final_wink3 import save

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'supervisor_evaluation/outputs/reselect_all_walkable_20261010'
RUNS = {
    'senseiver_a': 'methods/senseiver/runs/senseiver_A_pd',
    'senseiver_g': 'methods/senseiver/runs/sv_G_k16_pd',
    'dincae': 'methods/dincae/runs/dincae_ff_pd',
    'varnet': 'methods/varnet/runs/varnet_mse5_h96_pdfold_s3',
    # DINCAE without the log1p of `var`; run with DINCAE_VAR_TRANSFORM=none (load_models checks)
    'dincae_varnone': 'methods/dincae/runs/dincae_ff_pd_varnone_all150',
}
# DATA_VARIANT=all92 (route 2): its own runs, its own output directory, 14 validation days;
# nothing of the Sunday-only selection above is read or overwritten.
EXPECTED_VALID_DAYS = 7
if config.DATA_VARIANT:
    OUT = ROOT / f'supervisor_evaluation/outputs/reselect_all_walkable{config.VARIANT_SUFFIX}'
    RUNS = {
        'senseiver_a': 'methods/senseiver/runs/senseiver_A_pd_all92',
        'senseiver_g': 'methods/senseiver/runs/sv_G_k16_pd_all92',
        'dincae_varnone': 'methods/dincae/runs/dincae_ff_pd_varnone_all150_all92',
    }
    EXPECTED_VALID_DAYS = 14
if fr.ENABLED:
    OUT = OUT.with_name(OUT.name + '_gapmask')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def score(pred, truth, walk, keep=None):
    """Pooled squared error over the walkable cells of the frames with keep=True (all frames
    when keep is None)."""
    if keep is not None:
        pred, truth = pred[keep], truth[keep]
    sq = (pred.astype(np.float64) - truth) ** 2
    if not np.isfinite(sq).all():
        raise RuntimeError('Nonfinite validation prediction')
    return float(sq[:, :, walk].sum()), len(pred) * 4 * int(walk.sum())


def enkf():
    # These are full posterior predictions, not forecast-loss approximations.
    source = ROOT / 'supervisor_evaluation/outputs/reselect/enkf'
    previous = json.loads((source.parent / 'enkf_validation.json').read_text())
    rows = []
    for name in ['E1', 'E2', 'E3', 'E4']:
        se = n = 0
        for day in previous['days']:
            with np.load(source / f'{name}_{day}.npz') as z:
                s, c = score(z['mean'], z['truth'], z['walkable'].astype(bool))
                se += s; n += c
        rows.append(dict(candidate=name, mse=se/n, rmse=float(np.sqrt(se/n)),
                         config=previous['candidates'][name]['config']))
    save(OUT / 'enkf_selection.json', dict(split='valid', scope='all_walkable',
        days=previous['days'], criterion='pooled RMSE', candidates=rows,
        selected=min(rows, key=lambda r: r['mse']),
        forecast_checkpoint_unchanged=True, previous_winner=previous['winner']))
    print('[EnKF selected]', min(rows, key=lambda r: r['mse']), flush=True)


def score_senseiver_shard(paths, days, files, walk, cache, cache_path, progress_path):
    """All of a shard's Senseiver checkpoints, day by day: each batch's input tokens are built
    once and fed to every checkpoint still missing that day (they share the token builder's
    inputs -- positional encoding, input mean/std, time window -- which is checked). The tokens,
    batches (64 frames), clipping and scoring are those of select_checkpoint.predict, so the
    scores equal the one-checkpoint-at-a-time loop."""
    from methods.senseiver.checks.select_checkpoint import LO, HI
    from methods.senseiver.network import Senseiver
    from methods.senseiver import sensors
    models = []
    for path in paths:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model = Senseiver(**ck['hparams']).to('cuda')
        model.load_state_dict(ck['model']); model.eval()
        models.append((path, sha(path), int(ck['epoch']), model))
        del ck
    ref = models[0][3]
    for path, _, _, m in models[1:]:
        if m.time_window != ref.time_window or not all(
                torch.equal(getattr(m, k), getattr(ref, k)) for k in ('pos_enc', 'in_mean', 'in_std')):
            raise RuntimeError(f'{path}: token builder inputs differ from {models[0][0]}')
    pe = ref.pos_enc.detach().cpu().numpy()
    mean = ref.in_mean.detach().cpu().numpy()
    std = ref.in_std.detach().cpu().numpy()
    L = ref.time_window
    # SV_VERIFY=1: on the first day that has cached checkpoints, rescore those too and require
    # the cached (one-checkpoint-at-a-time) values back exactly.
    verify = os.environ.get('SV_VERIFY') == '1'
    for fp in files:
        key = lambda f: f + ':' + Path(fp).name
        cached = [m for m in models if key(m[1]) in cache]
        check = verify and bool(cached)
        todo = models if check else [m for m in models if key(m[1]) not in cache]
        if todo:
            _, X, Yf, Of = next(d for d in days if d[0] == fp)
            keep_day = fr.scorable_frames(fp, len(X))
            se = {m[1]: 0.0 for m in todo}; n = {m[1]: 0 for m in todo}
            for first in range(1, len(X)-1, 64):
                idx = list(range(first, min(first+64, len(X)-1)))
                if L > 1:
                    windows = [(Yf[max(0, t - L + 1):t + 1], Of[max(0, t - L + 1):t + 1]) for t in idx]
                    tok, pad, dt, _, cell = sensors.build_batch_temporal(windows, pe, mean, std,
                                                                         return_cell_idx=True)
                    inputs = (tok.to('cuda'), pad.to('cuda'), dt.to('cuda'), cell.to('cuda'))
                else:
                    tok, pad, _ = sensors.build_batch(Yf[idx], Of[idx], pe, mean, std)
                    inputs = (tok.to('cuda'), pad.to('cuda'))
                for _, f, _, model in todo:
                    with torch.no_grad():
                        p = np.clip(model.reconstruct(*inputs).cpu().numpy(), LO, HI)
                    s_, c_ = score(p, X[idx], walk, keep_day[idx]); se[f] += s_; n[f] += c_
            for _, f, _, _ in todo:
                if key(f) in cache:
                    if cache[key(f)] != [se[f], n[f]]:
                        raise RuntimeError(f'verify failed on {key(f)}: {cache[key(f)]} vs {[se[f], n[f]]}')
                    print(f'[verify] {key(f)[:12]}: shared-token score == cached score {cache[key(f)]}', flush=True)
                cache[key(f)] = [se[f], n[f]]
            save(cache_path, cache)
            verify = verify and not check
        print(f'[valid] {len(todo)} senseiver checkpoints scored on {Path(fp).name}', flush=True)
    rows = []
    for path, f, epoch, _ in models:
        tot = sum(cache[f + ':' + Path(fp).name][0] for fp in files)
        cnt = sum(cache[f + ':' + Path(fp).name][1] for fp in files)
        rows.append(dict(epoch=epoch, checkpoint=str(path), sha256=f, mse=tot/cnt, rmse=float(np.sqrt(tot/cnt))))
    save(progress_path, dict(candidates=rows))
    del models
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('method', choices=[*RUNS, 'enkf'])
    ap.add_argument('--shard', default='0/1',
                    help='k/n: score every n-th candidate starting at k, with its own cache; '
                         'the shard that completes the set writes the selection')
    a = ap.parse_args()
    k_shard, n_shard = (int(x) for x in a.shard.split('/'))
    tag = '' if n_shard == 1 else f'_shard{k_shard}of{n_shard}'
    is_dincae = a.method.startswith('dincae')
    OUT.mkdir(parents=True, exist_ok=True)
    if a.method == 'enkf':
        enkf()
        return
    if not torch.cuda.is_available():
        raise SystemExit('GPU required')
    run = ROOT / RUNS[a.method]
    files = om.split_files('valid')
    if len(files) != EXPECTED_VALID_DAYS:
        raise RuntimeError(f'Expected {EXPECTED_VALID_DAYS} validation days, got {len(files)}')
    walk = nav.build_valid_mask_from_config().astype(bool)
    print(f'[frames] {fr.describe()}', flush=True)
    if fr.ENABLED and a.method == 'varnet':
        raise SystemExit('the old 4DVarNet path (score_day) has no frame mask')
    cache_path = OUT / f'{a.method}_cache{tag}.json'
    # Read every cache of this method (any earlier sharding): a (checkpoint sha, day) score is
    # the same whichever shard computed it, so re-sharding never recomputes finished days.
    cache = {}
    for other in sorted(OUT.glob(f'{a.method}_cache*.json')):
        cache.update(json.loads(other.read_text()))
    dest = OUT / f'{a.method}_selection.json'
    if dest.exists():
        print('[already complete]', dest, flush=True)
        return
    paths = (sorted(run.glob('epoch_*.pt')) if a.method.startswith('senseiver')
             else sorted(run.glob('ckpt_*.pt')))
    if is_dincae:
        paths += [run / 'last.pt']
    all_paths = paths
    paths = paths[k_shard::n_shard]
    if a.method == 'varnet':
        paths = sorted((run / 'frozen_candidates_20261009').glob('epoch_*.pt'))
    rows = []
    if a.method.startswith('senseiver'):
        from methods.senseiver.checks.select_checkpoint import predict, LO, HI
        from methods.senseiver.network import Senseiver
        from methods.senseiver import dataset as ds
        args = torch.load(paths[0], map_location='cpu', weights_only=False)['args']
        mode = args.get('trajectory_mode', 'fixed')
        base = args['seed'] if args.get('valid_obs_seed') is None else args['valid_obs_seed']
        if args.get('obs_seed') is not None and args.get('valid_obs_seed') is None:
            base = args['obs_seed']
        cfg = ds.obs_config()
        days = []
        for fp in files:
            X = np.asarray(om.load_state(fp)[0], np.float32)
            obs = om.generate_observations(X, sensing_range=cfg['sensing_range'],
                num_agents=cfg['num_agents'], add_noise=cfg['add_noise'],
                seed=om.day_seed(fp, base, mode), valid_mask=nav.build_valid_mask_from_config(X),
                obs_every_k=cfg['obs_every_k'])
            days.append((fp, X, obs['Y'].astype(np.float32).reshape(len(X), 4, -1),
                         obs['Omega'].astype(bool).reshape(len(X), -1)))
    if a.method.startswith('senseiver'):
        rows = score_senseiver_shard(paths, days, files, walk, cache, cache_path,
                                     OUT / f'{a.method}_progress{tag}.json')
        paths = []                     # scored above; the generic loop below is for the others
    for path in paths:
        fingerprint = sha(path)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        epoch = int(ck['epoch'])
        if a.method.startswith('senseiver'):
            model = Senseiver(**ck['hparams']).to('cuda')
            model.load_state_dict(ck['model']); model.eval()
        elif is_dincae:
            from methods.dincae.checks.evaluate import load_models, predict_day, clip_bounds
            from methods.dincae.state import StateStats
            model, _, _ = load_models(str(run), str(path), 'cuda')
            stats = StateStats()
            mode = ck.get('args', {}).get('trajectory_mode', 'fixed')
        else:
            from methods.varnet.checks.model_io import load_solver
            from methods.varnet.checks.select_checkpoint import score_day
            model, args, _ = load_solver(str(path), 'cuda')
            model.requires_grad_(False)
            if ck['n_iter_eff'] != 20:
                raise RuntimeError('Old 4DVarNet must use trained K20 candidates')
        del ck
        for fp in files:
            key = fingerprint + ':' + Path(fp).name
            if key not in cache:
                if a.method.startswith('senseiver'):
                    _, X, Yf, Of = next(d for d in days if d[0] == fp)
                    keep_day = fr.scorable_frames(fp, len(X))
                    se = n = 0
                    for first in range(1, len(X)-1, 64):
                        idx = list(range(first, min(first+64, len(X)-1)))
                        p = np.clip(predict(model, Yf, Of, idx, 'cuda'), LO, HI)
                        s, c = score(p, X[idx], walk, keep_day[idx]); se += s; n += c
                elif is_dincae:
                    truth, rec, _, _, _ = predict_day(model, stats, fp, torch.device('cuda'), 0, 256,
                                                    seed=om.day_seed(fp, 0, mode))
                    # rows are original frames -min(FRESH_OFFSETS) .. n-1-max(FRESH_OFFSETS)
                    from methods.dincae.checks.evaluate import FRESH_OFFSETS
                    lo_f, hi_f = -min(FRESH_OFFSETS), max(FRESH_OFFSETS)
                    keep_day = fr.scorable_frames(fp, len(truth) + lo_f + hi_f)[lo_f:lo_f + len(truth)]
                    se, n = score(clip_bounds(rec), truth, walk, keep_day)
                    del truth, rec
                else:
                    # Old weights retain their original global-gradient normalization.
                    # Keep B8, as in their previous checkpoint selection.
                    se, n = score_day({epoch: model}, fp, {str(path): args}, walk,
                                      'cuda', 8, 'all_walkable')[epoch]
                cache[key] = [se, n]
                save(cache_path, cache)
            print(f'[valid] {a.method} epoch {epoch}: {Path(fp).name}', flush=True)
        se = sum(cache[fingerprint + ':' + Path(fp).name][0] for fp in files)
        n = sum(cache[fingerprint + ':' + Path(fp).name][1] for fp in files)
        rows.append(dict(epoch=epoch, checkpoint=str(path), sha256=fingerprint,
                         mse=se/n, rmse=float(np.sqrt(se/n))))
        save(OUT / f'{a.method}_progress{tag}.json', dict(candidates=rows))
        del model
        torch.cuda.empty_cache()
    if n_shard > 1:                    # merge once every shard's candidates are scored
        rows = []
        for k in range(n_shard):
            pf = OUT / f'{a.method}_progress_shard{k}of{n_shard}.json'
            if pf.exists():
                rows += json.loads(pf.read_text())['candidates']
        if len(rows) < len(all_paths):
            print(f'[shard {k_shard}/{n_shard}] done; {len(rows)}/{len(all_paths)} scored so far',
                  flush=True)
            return
    if not rows or not all(np.isfinite(r['mse']) for r in rows):
        raise RuntimeError('No finite candidate scores')
    save(dest, dict(split='valid', scope='all_walkable', criterion='pooled RMSE',
        clipping='density [0,5], velocities [-5,5], variance [0,2]',
        days=[Path(fp).name for fp in files], frames=fr.describe(), candidates=rows,
        selected=min(rows, key=lambda r: (r['mse'], r['epoch'])),
        varnet_batch=8 if a.method == 'varnet' else None))
    print('[selected]', json.loads(dest.read_text())['selected'], flush=True)


if __name__ == '__main__':
    main()
