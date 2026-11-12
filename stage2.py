"""Second-stage experiment driver; reuses run.py and the original two models."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'stage2'


def audit_prepared(cfg):
    import joblib
    records = []
    for seed in cfg['seeds']:
        path = OUT / f'prepared_seed{seed}'
        frame = pd.read_csv(path / 'sample_manifest.csv', dtype={'original_user': str})
        assert frame['index'].is_unique
        assert (frame.stars == frame.label+1).all()
        data = joblib.load(path / 'subsets.joblib')
        user_sets = {s: set(g.original_user) for s, g in frame.groupby('split')}
        hashes = {s: {hashlib.sha256(t.encode()).hexdigest() for t in item['texts']} for s, item in data.items()}
        for a in data:
            for b in data:
                if a >= b:
                    continue
                shared = len(user_sets[a] & user_sets[b])
                records.append(dict(seed=seed, split_a=a, split_b=b,
                    reviewer_intersection=shared, exact_text_intersection=len(hashes[a]&hashes[b])))
                if (a in ['val', 'test'] and b not in ['val', 'test']) or (b in ['val', 'test'] and a not in ['val', 'test']):
                    assert shared == 0
    pd.DataFrame(records).to_csv(OUT / 'leakage_audit.csv', index=False)


def prepare(cfg):
    from wilds import get_dataset
    from src.data import load_data
    paths = [OUT / f'prepared_seed{s}' for s in cfg['seeds']]
    if all((p / 'dataset_summary.json').exists() and (p / 'subsets.joblib').exists() for p in paths):
        print('Reusing prepared stage2 subsets (no dataset reload/download)', flush=True)
        return
    ds = get_dataset(dataset='amazon', version=cfg['dataset_version'], root_dir=str(ROOT / cfg['data_dir']), download=False)
    previous = pd.concat([pd.read_csv(ROOT / 'results' / p / 'sample_manifest.csv') for p in ['smoke', 'full']])
    rng = np.random.default_rng(cfg['evaluation_seed'])
    fixed, excluded = {}, {}
    for split in ['id_test', 'test', 'val']:
        prior = previous.loc[previous.split == split, 'index'].unique() if split != 'val' else []
        eligible = np.setdiff1d(ds.get_subset(split).indices, prior)
        fixed[split] = rng.choice(eligible, cfg['limits'][split], replace=False)
        excluded[split] = len(prior)
    users = ds.metadata_array[:, ds.metadata_fields.index('user')].numpy()
    official_users = {s: set(users[ds.get_subset(s).indices].tolist()) for s in ds.split_dict}
    overlaps = {f'{a}:{b}': len(official_users[a] & official_users[b])
                for a in official_users for b in official_users if a < b}
    assert not (official_users['train'] & official_users['test'])
    assert not (official_users['train'] & official_users['val'])
    assert not (official_users['id_val'] & official_users['test'])
    assert not (official_users['val'] & official_users['test'])
    for seed, path in zip(cfg['seeds'], paths):
        run_cfg = copy.deepcopy(cfg)
        run_cfg['seed'] = seed
        generator = np.random.default_rng(seed)
        indices = dict(fixed)
        for split in ['train', 'id_val']:
            indices[split] = generator.choice(ds.get_subset(split).indices, cfg['limits'][split], replace=False)
        load_data(run_cfg, path, dataset=ds, selected_indices=indices)
    audit = {'phase1_used_final_tests': True, 'excluded_previously_evaluated_rows': excluded,
        'evaluation_seed': cfg['evaluation_seed'], 'official_reviewer_intersections': overlaps,
        'fixed_final_test_samples_across_seeds': True,
        'note': 'ID reviewer overlap with train/calibration is intentional in official split; row indices disjoint. Fresh rows do not make the entire research process pristine confirmatory.'}
    (OUT / 'split_audit.json').write_text(json.dumps(audit, indent=2))


def run_one(cfg, tag, train_size, mode, development=False):
    run_cfg = copy.deepcopy(cfg)
    run_cfg['limits']['train'] = train_size
    run_cfg['prepared_data'] = f'results/stage2/prepared_seed{cfg["seed"]}'
    run_cfg['evaluation_splits'] = ['val'] if development else ['id_test', 'val', 'test']
    config_path = OUT / 'configs' / f'{tag}.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(run_cfg, sort_keys=False)
    if config_path.exists() and config_path.read_text() != content:
        raise ValueError('Refusing to overwrite a different frozen experiment configuration')
    config_path.write_text(content)
    result = OUT / tag
    model_path = ROOT / 'models' / 'stage2' / tag
    weights_exist = (model_path / 'baseline.joblib').exists() and (model_path / 'bert' / 'model.safetensors').exists()
    if mode == 'all' and weights_exist and (result / 'metrics.csv').exists() and (result / 'status.json').exists() and json.loads((result / 'status.json').read_text()).get('status') == 'complete':
        print(f'Reusing completed {tag}', flush=True)
        return
    if mode == 'train' and weights_exist and (result / 'run_config.json').exists():
        print(f'Reusing trained {tag}', flush=True)
        return
    saved_status = json.loads((result / 'status.json').read_text()) if (result / 'status.json').exists() else {}
    if mode == 'evaluate' and (result / 'metrics.csv').exists() and saved_status.get('status') == 'complete' and saved_status.get('mode') in ['all', 'evaluate']:
        print(f'Reusing evaluated {tag}', flush=True)
        return
    result.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'run.py', '--mode', mode, '--config', str(config_path), '--run-name', f'stage2/{tag}']
    print('RUN', ' '.join(command), flush=True)
    start = time.time()
    with (result / f'{mode}.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    with (OUT / 'execution.jsonl').open('a') as log:
        log.write(json.dumps({'tag': tag, 'mode': mode, 'seconds': time.time()-start, 'returncode': completed.returncode, 'command': command})+'\n')
    if completed.returncode:
        raise RuntimeError(f'{tag} failed: see {result / (mode + ".log")}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'prepare', 'medium', 'formal', 'analyze'], default='all')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'hf_cache'))
    os.environ.setdefault('OUTDATED_IGNORE', '1')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'config.stage2.yaml').read_text())
    # Freeze the protocol before examining any second-stage test predictions.
    frozen = OUT / 'protocol.yaml'
    content = yaml.safe_dump(cfg, sort_keys=False)
    if frozen.exists() and frozen.read_text() != content:
        raise ValueError('Protocol changed; use a new experiment directory')
    frozen.write_text(content)
    if args.mode in ['all', 'prepare', 'medium', 'formal']:
        prepare(cfg)
        audit_prepared(cfg)
    if args.mode in ['all', 'medium']:
        run_one(cfg, 'medium_seed42', 3000, 'all', development=True)
    if args.mode in ['all', 'formal']:
        status = OUT / 'medium_seed42' / 'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'complete':
            raise RuntimeError('Complete medium experiment first')
        # Finish all pre-specified training before opening final-test predictions.
        for seed in cfg['seeds']:
            run_cfg = dict(cfg, seed=seed)
            run_one(run_cfg, f'formal_seed{seed}', 10000, 'train')
        for seed in cfg['seeds']:
            run_one(dict(cfg, seed=seed), f'formal_seed{seed}', 10000, 'evaluate')
    if args.mode in ['all', 'analyze']:
        from src.evaluate import analyze_stage2
        analyze_stage2(OUT, cfg)


if __name__ == '__main__':
    main()
