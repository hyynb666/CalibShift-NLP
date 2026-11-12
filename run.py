import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='CalibShift-NLP: official WILDS Amazon experiment')
    parser.add_argument('--mode', choices=['smoke', 'train', 'evaluate', 'all'], required=True)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--profile', choices=['smoke', 'full'], default='full', help='Saved experiment to evaluate')
    parser.add_argument('--models', nargs='+', choices=['baseline', 'bert'], default=['baseline', 'bert'])
    parser.add_argument('--run-name', help='Isolated result/model directory below their roots')
    args = parser.parse_args()
    os.chdir(ROOT)
    # Keep caches within the project; no system configuration changes.
    os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'hf_cache'))
    os.environ.setdefault('OUTDATED_IGNORE', '1')
    profile = 'smoke' if args.mode == 'smoke' else args.profile
    run_name = args.run_name or profile
    if Path(run_name).is_absolute() or '..' in Path(run_name).parts:
        raise ValueError('run-name must be a relative directory without ..')
    out = ROOT / 'results' / run_name
    out.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        import torch
        import joblib
        from src.data import load_data
        from src.train import train_baseline, train_bert, seed_all
        from src.evaluate import evaluate
        from src.plots import plot_results
        cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        cfg['data_dir'] = str((ROOT / cfg['data_dir']).resolve())
        if profile == 'smoke':
            cfg['bert']['max_length'] = 64
        device = 'cpu' if profile == 'smoke' else ('cuda' if torch.cuda.is_available() else 'cpu')
        seed_all(cfg['seed'])
        environment = {'python': sys.version, 'platform': platform.platform(), 'device': device,
            'cuda_available': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'wilds', 'numpy', 'pandas', 'scikit-learn']}}
        (out / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
        print(json.dumps(environment, indent=2), flush=True)
        model_dir = ROOT / 'models' / run_name
        signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if args.mode == 'evaluate':
            saved = json.loads((out / 'run_config.json').read_text())
            if saved['signature'] != signature:
                raise ValueError('Config changed since training; restore original config or retrain')
            if not set(args.models).issubset(saved['models']):
                raise ValueError('Requested model was not trained in this experiment')
            data = joblib.load(out / 'subsets.joblib')
        else:
            if profile == 'full':
                smoke_status = ROOT / 'results' / 'smoke' / 'status.json'
                status = json.loads(smoke_status.read_text()) if smoke_status.exists() else {}
                if status.get('status') != 'complete' or not set(args.models).issubset(status.get('models', [])):
                    raise RuntimeError('Run CPU smoke successfully for the requested models before full training')
            # A failed retraining must never leave an old configuration pointing
            # to a mixture of old weights and newly sampled data.
            (out / 'run_config.json').unlink(missing_ok=True)
            if cfg.get('prepared_data'):
                source = ROOT / cfg['prepared_data']
                prepared = json.loads((source / 'dataset_summary.json').read_text())
                if prepared['seed'] != cfg['seed'] or prepared['version'] != cfg['dataset_version']:
                    raise ValueError('Prepared data seed/version mismatch')
                data = joblib.load(source / 'subsets.joblib')
                data['train'] = {k: v[:cfg['limits']['train']] for k, v in data['train'].items()}
                import pandas as pd
                manifest = pd.read_csv(source / 'sample_manifest.csv', dtype={'original_user': str})
                manifest = manifest[(manifest.split != 'train') | manifest.index.isin(manifest[manifest.split == 'train'].index[:cfg['limits']['train']])]
                manifest.to_csv(out / 'sample_manifest.csv', index=False)
                prepared['splits']['train']['sampled'] = len(data['train']['labels'])
                import numpy as np
                prepared['splits']['train']['class_counts'] = np.bincount(data['train']['labels'], minlength=5).tolist()
                (out / 'dataset_summary.json').write_text(json.dumps(prepared, indent=2))
                joblib.dump(data, out / 'subsets.joblib')
            else:
                data = load_data(cfg, out, smoke=profile == 'smoke')
            # Invalidate old completion before attempting new training.
            (out / 'status.json').write_text(json.dumps({'status': 'running'}))
            for name in args.models:
                print(f'Training {name} on {len(data["train"]["labels"])} official train rows', flush=True)
                if name == 'baseline':
                    train_baseline(data['train'], cfg, model_dir)
                else:
                    train_bert(data['train'], cfg, model_dir, device, smoke=profile == 'smoke')
            (out / 'run_config.json').write_text(json.dumps({'config': cfg, 'signature': signature, 'models': args.models}, indent=2))
        if args.mode in ('evaluate', 'all', 'smoke'):
            metrics, curves = evaluate(data, cfg, model_dir, out, device, args.models)
            plot_results(metrics, curves, out)
            print(metrics.to_string(index=False), flush=True)
        (out / 'status.json').write_text(json.dumps({'status': 'complete', 'mode': args.mode, 'models': args.models}))
    except Exception as exc:
        (out / 'status.json').write_text(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=False), encoding='utf-8')
        (out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        if 'out of memory' in str(exc).lower():
            print('CUDA OOM: reduce batch_size to 2 or 1, increase gradient accumulation, keep checkpointing enabled; then rerun smoke/train.', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
