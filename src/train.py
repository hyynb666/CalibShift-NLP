"""Two fixed models. No test-set model selection or calibration."""
import random
import time
import json
from pathlib import Path
import joblib
import numpy as np


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_baseline(data, config, path):
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    if set(data['labels']) != set(range(5)):
        raise ValueError('Training sample lacks a class; increase sample size')
    model = make_pipeline(TfidfVectorizer(max_features=config['baseline']['max_features'],
                                         ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(C=1.0, max_iter=config['baseline']['max_iter'],
                                             random_state=config['seed']))
    model.fit(data['texts'], data['labels'])
    Path(path).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(path) / 'baseline.joblib')


def bert_loader(data, tokenizer, config, shuffle=False):
    import torch
    from transformers import DataCollatorWithPadding
    class Reviews(torch.utils.data.Dataset):
        def __len__(self):
            return len(data['labels'])
        def __getitem__(self, i):
            item = tokenizer(data['texts'][i], truncation=True, max_length=config['bert']['max_length'])
            item['labels'] = int(data['labels'][i])
            return item
    generator = torch.Generator().manual_seed(config['seed'])
    return torch.utils.data.DataLoader(Reviews(), batch_size=config['bert']['batch_size'],
        shuffle=shuffle, num_workers=0, generator=generator,
        collate_fn=DataCollatorWithPadding(tokenizer))


def train_bert(data, config, path, device, smoke=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    seed_all(config['seed'])
    cfg = config['bert']
    tokenizer = AutoTokenizer.from_pretrained(cfg['name'], revision=cfg['revision'])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['name'], revision=cfg['revision'], num_labels=5,
        id2label={i: str(i + 1) + ' star' for i in range(5)},
        label2id={str(i + 1) + ' star': i for i in range(5)})
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    model.to(device).train()
    loader = bert_loader(data, tokenizer, config, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.01)
    amp = device == 'cuda' and cfg['mixed_precision']
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    accumulation = cfg['gradient_accumulation_steps']
    epochs = 1 if smoke else cfg['epochs']
    history = []
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Correct denominator for a final partial accumulation window.
            start = (step // accumulation) * accumulation
            window = min(accumulation, len(loader) - start)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            loss_sum += loss.item()
            scaler.scale(loss / window).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 250 == 0:
                print(f'Epoch {epoch+1}: batch {step+1}/{len(loader)}, loss={loss_sum/(step+1):.4f}', flush=True)
        print(f'DistilBERT epoch {epoch + 1}/{epochs}, mean batch loss={loss_sum / len(loader):.4f}', flush=True)
        history.append({'epoch': epoch+1, 'mean_batch_loss': loss_sum/len(loader),
                        'seconds': time.perf_counter()-started})
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(Path(path) / 'bert')
    tokenizer.save_pretrained(Path(path) / 'bert')
    (Path(path) / 'training_history.json').write_text(json.dumps({'epochs': history,
        'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None}, indent=2))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


def predict_logits(name, data, config, path, device):
    if name == 'baseline':
        return joblib.load(Path(path) / 'baseline.joblib').decision_function(data['texts'])
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(Path(path) / 'bert')
    model = AutoModelForSequenceClassification.from_pretrained(Path(path) / 'bert').to(device).eval()
    output = []
    with torch.inference_mode():
        for batch in bert_loader(data, tokenizer, config):
            batch.pop('labels')
            output.append(model(**{k: v.to(device) for k, v in batch.items()}).logits.cpu().numpy())
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(output)
