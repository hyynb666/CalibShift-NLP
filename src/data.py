"""Official WILDS splits; reproducible uniform row sampling without replacement."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_data(config, result_dir, smoke=False, dataset=None, selected_indices=None):
    from wilds import get_dataset
    ds = dataset if dataset is not None else get_dataset(dataset='amazon', version=config['dataset_version'],
                     root_dir=config['data_dir'], download=config['download'],
                     split_scheme='official')
    if ds.n_classes != 5 or list(ds.metadata_map['y']) != list(range(1, 6)):
        raise ValueError('Unexpected WILDS star mapping')
    limits = config['smoke_limits'] if smoke else config['limits']
    rng = np.random.default_rng(config['seed'])
    groups, records, summary = {}, [], {}
    for name in ('train', 'id_val', 'id_test', 'val', 'test'):
        subset = ds.get_subset(name)
        idx = (np.asarray(selected_indices[name]) if selected_indices is not None else
               rng.choice(subset.indices, min(limits[name], len(subset)), replace=False))
        if not np.isin(idx, subset.indices).all() or len(np.unique(idx)) != len(idx):
            raise ValueError('Invalid or duplicate official subset indices')
        groups[name] = idx
        summary[name] = {'available': len(subset), 'sampled': len(idx),
                         'class_counts': np.bincount(ds.y_array[idx].numpy(), minlength=5).tolist()}
    # Independent uniform random partition, with no test involvement or label balancing.
    idx = rng.permutation(groups.pop('id_val'))
    groups['temperature'], groups['conformal'] = np.array_split(idx, 2)
    if min(len(groups['temperature']), len(groups['conformal'])) == 0:
        raise ValueError('id_val must contain at least two samples')
    data = {}
    seen = set()
    for name, indices in groups.items():
        texts, labels = [], []
        for i in indices:
            i = int(i)
            if i in seen:
                raise AssertionError('Split overlap')
            seen.add(i)
            text, y, metadata = ds[i]
            label = int(y)
            if label not in range(5):
                raise ValueError('Label out of range')
            texts.append(text)
            labels.append(label)
            row = {'index': i, 'split': name, 'official_split': 'id_val' if name in ('temperature', 'conformal') else name,
                   'label': label, 'stars': label + 1}
            for field, value in zip(ds.metadata_fields, metadata.tolist()):
                row['metadata_' + field] = value
                row['original_' + field] = str(ds.metadata_map[field][value])
            records.append(row)
        data[name] = {'texts': texts, 'labels': np.asarray(labels, dtype=int)}
        summary[name] = {**summary.get(name, {}), 'sampled': len(labels),
                         'class_counts': np.bincount(labels, minlength=5).tolist()}
    path = Path(result_dir)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path / 'sample_manifest.csv', index=False)
    (path / 'dataset_summary.json').write_text(json.dumps({
        'dataset': 'WILDS Amazon', 'version': ds.version, 'seed': config['seed'],
        'split_scheme': 'official/user', 'sampling': 'uniform without replacement',
        'label_to_stars': dict(enumerate(range(1, 6))), 'splits': summary}, indent=2), encoding='utf-8')
    joblib.dump(data, path / 'subsets.joblib')
    return data
