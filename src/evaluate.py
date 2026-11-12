import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .calibration import probabilities, fit_temperature, lac_quantile, prediction_sets
from .train import predict_logits


def reliability(probs, labels, bins=15):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    return [{'bin': i, 'count': int((indices == i).sum()),
             'confidence': float(confidence[indices == i].mean()),
             'accuracy': float(correct[indices == i].mean())}
            for i in range(bins) if (indices == i).any()]


def metrics(probs, labels, sets, bins=15):
    rows = reliability(probs, labels, bins)
    return {'n': len(labels), 'accuracy': float(np.mean(probs.argmax(axis=1) == labels)),
            'macro_f1': float(f1_score(labels, probs.argmax(axis=1), labels=range(5), average='macro', zero_division=0)),
            'nll': float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
            'ece': sum(r['count'] * abs(r['confidence'] - r['accuracy']) for r in rows) / len(labels),
            'brier': float(((probs - np.eye(5)[labels]) ** 2).sum(axis=1).mean()),
            'coverage': float(sets[np.arange(len(labels)), labels].mean()),
            'average_set_size': float(sets.sum(axis=1).mean())}


def evaluate(data, config, model_dir, result_dir, device, models):
    rows, curves, fitted = [], [], {}
    path = Path(result_dir)
    splits = config.get('evaluation_splits', ['id_test', 'val', 'test'])
    for name in models:
        z = {split: predict_logits(name, data[split], config, model_dir, device)
             for split in ['temperature', 'conformal'] + splits}
        t = fit_temperature(z['temperature'], data['temperature']['labels'])
        fitted[name] = {'temperature': t}
        for method, temp in [('raw', 1.0), ('temperature', t)]:
            threshold = lac_quantile(probabilities(z['conformal'], temp), data['conformal']['labels'], config['alpha'])
            fitted[name][method + '_qhat'] = threshold if np.isfinite(threshold) else 'infinity'
            for split in splits:
                p, y = probabilities(z[split], temp), data[split]['labels']
                sets = prediction_sets(p, threshold)
                rows.append({'model': name, 'method': method, 'split': split,
                             'target_coverage': 1 - config['alpha'],
                             **metrics(p, y, sets, config['ece_bins'])})
                curves.extend({'model': name, 'method': method, 'split': split, **r}
                              for r in reliability(p, y, config['ece_bins']))
                np.savez_compressed(path / f'predictions_{name}_{method}_{split}.npz',
                                    probabilities=p, labels=y, sets=sets)
    pd.DataFrame(rows).to_csv(path / 'metrics.csv', index=False)
    pd.DataFrame(curves).to_csv(path / 'reliability.csv', index=False)
    (path / 'calibration.json').write_text(json.dumps(fitted, indent=2), encoding='utf-8')
    return pd.DataFrame(rows), pd.DataFrame(curves)


def cluster_bootstrap(probs, labels, sets, reviewers, bins=15, replicates=2000, seed=814):
    """Percentile cluster bootstrap, conditional on fixed weights and calibrators.

    Reviewers are resampled with replacement; every observed review of a drawn
    reviewer receives the same multiplicity. Overall targets remain review-weighted.
    Sufficient statistics avoid constructing thousands of repeated text datasets.
    """
    y = np.asarray(labels, int)
    p = np.asarray(probs, float)
    pred, conf = p.argmax(1), p.max(1)
    _, group = np.unique(np.asarray(reviewers, str), return_inverse=True)
    g = group.max() + 1
    if g < 2:
        raise ValueError('Cluster bootstrap requires at least two reviewers')
    correct = pred == y
    covered = sets[np.arange(len(y)), y].astype(float)
    cm = np.eye(25)[5*y+pred]
    bin_id = np.minimum((conf*bins).astype(int), bins-1)
    onehot = np.eye(bins)[bin_id]
    features = np.column_stack([np.ones(len(y)), correct,
        -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)),
        ((p-np.eye(5)[y])**2).sum(1), covered, sets.sum(1), cm,
        onehot, onehot*conf[:, None], onehot*correct[:, None],
        np.eye(5)[y]*covered[:, None]])
    sums = np.zeros((g, features.shape[1]))
    np.add.at(sums, group, features)
    reviewer_coverages = sums[:, 4] / sums[:, 0]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(g, np.full(g, 1/g), size=replicates)
    # First row is the observed statistic, remaining rows are bootstrap draws.
    aggregate = np.vstack([sums.sum(0), counts @ sums])
    total = aggregate[:, 0]
    confusion = aggregate[:, 6:31].reshape(-1, 5, 5)
    tp = np.diagonal(confusion, axis1=1, axis2=2)
    support, predicted = confusion.sum(2), confusion.sum(1)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros_like(tp), where=(support+predicted)>0)
    bcount = aggregate[:, 31:31+bins]
    bconf = aggregate[:, 31+bins:31+2*bins]
    bcorrect = aggregate[:, 31+2*bins:31+3*bins]
    values = {'accuracy': aggregate[:, 1]/total, 'nll': aggregate[:, 2]/total,
        'brier': aggregate[:, 3]/total, 'coverage': aggregate[:, 4]/total,
        'average_set_size': aggregate[:, 5]/total, 'macro_f1': f1.mean(1),
        'ece': np.abs(bconf-bcorrect).sum(1)/total,
        'reviewer_mean_coverage': np.r_[reviewer_coverages.mean(), counts @ reviewer_coverages / g]}
    for k in range(5):
        values[f'precision_{k}'] = np.divide(tp[:, k], predicted[:, k], out=np.zeros(len(total)), where=predicted[:, k]>0)
        values[f'recall_{k}'] = np.divide(tp[:, k], support[:, k], out=np.full(len(total), np.nan), where=support[:, k]>0)
        values[f'class_coverage_{k}'] = np.divide(aggregate[:, 31+3*bins+k], support[:, k], out=np.full(len(total), np.nan), where=support[:, k]>0)
    intervals = []
    for metric, vector in values.items():
        valid = vector[1:][np.isfinite(vector[1:])]
        lower, upper = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
        intervals.append({'metric': metric, 'estimate': vector[0], 'lower': lower, 'upper': upper,
                          'bootstrap_valid': len(valid), 'reviewers': g, 'n': len(y)})
    return pd.DataFrame(intervals), {k: v[1:] for k, v in values.items()}


def class_set_bootstrap(labels, sets, reviewers, replicates=2000, seed=814):
    """Reviewer cluster bootstrap for class coverage and true-label set sizes.

    Uses the same sorted-reviewer/multinomial protocol as Stage 2. Absent true
    classes yield NaN, including macro/worst metrics, rather than being dropped.
    """
    y = np.asarray(labels, int)
    sets = np.asarray(sets, bool)
    if sets.shape != (len(y), 5) or len(reviewers) != len(y) or not len(y):
        raise ValueError('Misaligned labels, sets or reviewers')
    _, groups = np.unique(np.asarray(reviewers, str), return_inverse=True)
    g = groups.max()+1
    if g < 2:
        raise ValueError('At least two reviewers required')
    covered, size = sets[np.arange(len(y)), y], sets.sum(1)
    onehot = np.eye(5)[y]
    rows = np.column_stack([onehot, onehot*covered[:,None], onehot*size[:,None]])
    summed = np.zeros((g,15))
    np.add.at(summed, groups, rows)
    counts = np.random.default_rng(seed).multinomial(g, np.full(g,1/g), size=replicates)
    agg = np.vstack([summed.sum(0), counts@summed])
    n, cov, sizes = agg[:,:5], agg[:,5:10], agg[:,10:15]
    cc = np.divide(cov,n,out=np.full_like(cov,np.nan),where=n>0)
    cs = np.divide(sizes,n,out=np.full_like(sizes,np.nan),where=n>0)
    values = {'coverage':cov.sum(1)/n.sum(1), 'average_set_size':sizes.sum(1)/n.sum(1),
              'worst_class_coverage':cc.min(1), 'macro_class_coverage':cc.mean(1)}
    for k in range(5):
        values[f'class_coverage_{k}'] = cc[:,k]
        values[f'class_size_{k}'] = cs[:,k]
    result=[]
    for metric, v in values.items():
        valid=v[1:][np.isfinite(v[1:])]
        lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
        result.append(dict(metric=metric,estimate=v[0],lower=lo,upper=hi,bootstrap_valid=len(valid)))
    return pd.DataFrame(result), {k:v[1:] for k,v in values.items()}


def analyze_stage2(directory, config):
    """Analyze saved final predictions only; never fits or selects a model."""
    import joblib
    from sklearn.metrics import precision_recall_fscore_support
    from .plots import plot_stage2
    directory = Path(directory)
    all_metrics, intervals, classes, reviewers, differences, majority, paired = [], [], [], [], [], [], []
    draws_by_seed = {}
    for seed in config['seeds']:
        path = directory / f'formal_seed{seed}'
        if not (path / 'metrics.csv').exists():
            raise FileNotFoundError(f'Incomplete formal experiment: {path}')
        frame = pd.read_csv(path / 'metrics.csv')
        frame['seed'] = seed
        all_metrics.append(frame)
        manifest = pd.read_csv(path / 'sample_manifest.csv', dtype={'original_user': str})
        assert manifest['index'].is_unique
        data = joblib.load(path / 'subsets.joblib')
        majority_label = int(np.bincount(data['train']['labels'], minlength=5).argmax())
        seed_draws = {}
        base_seed = config['bootstrap_seed']
        for split, split_seed in [('id_test', base_seed), ('test', base_seed+1), ('val', base_seed+2)]:
            subset = manifest[manifest.split == split]
            y = subset.label.to_numpy()
            ids = subset.original_user.to_numpy()
            assert np.array_equal(y, data[split]['labels'])
            constant = np.zeros((len(y), 5))
            constant[:, majority_label] = 1
            # Majority reference has no claimed probabilistic calibration or CP.
            ref_ci, _ = cluster_bootstrap(constant, y, np.ones_like(constant, bool), ids,
                config['ece_bins'], config['bootstrap_replicates'], split_seed)
            for metric in ['accuracy', 'macro_f1']:
                r = ref_ci[ref_ci.metric == metric].iloc[0].to_dict()
                majority.append(dict(seed=seed, split=split, majority_label=majority_label, **r))
            for model in ['baseline', 'bert']:
                for method in ['raw', 'temperature']:
                    saved = np.load(path / f'predictions_{model}_{method}_{split}.npz')
                    p, actual, sets = [saved[k] for k in ['probabilities', 'labels', 'sets']]
                    assert np.array_equal(y, actual)
                    prefix = {'seed': seed, 'model': model, 'method': method, 'split': split}
                    ci, draws = cluster_bootstrap(p, y, sets, ids, config['ece_bins'],
                        config['bootstrap_replicates'], split_seed)
                    ci = ci.assign(**prefix)
                    intervals.append(ci)
                    seed_draws[model, method, split] = draws
                    prec, rec, f1, support = precision_recall_fscore_support(y, p.argmax(1), labels=range(5), zero_division=0)
                    covered = sets[np.arange(len(y)), y]
                    for k in range(5):
                        classes.append(dict(prefix, label=k, stars=k+1, support=int(support[k]),
                            precision=prec[k], recall=rec[k], f1=f1[k],
                            coverage=float(covered[y == k].mean()) if (y == k).any() else np.nan))
                    per_review = pd.DataFrame({'reviewer': ids, 'covered': covered, 'size': sets.sum(1), 'correct': p.argmax(1)==y})
                    per_user = per_review.groupby('reviewer', sort=True).agg(n=('covered', 'size'), coverage=('covered', 'mean'),
                        average_set_size=('size', 'mean'), accuracy=('correct', 'mean')).reset_index()
                    for k, v in prefix.items():
                        per_user[k] = v
                    reviewers.append(per_user)
        draws_by_seed[seed] = seed_draws
        for model in ['baseline', 'bert']:
            for method in ['raw', 'temperature']:
                delta = seed_draws[model, method, 'test']['coverage'] - seed_draws[model, method, 'id_test']['coverage']
                values = frame[(frame.model==model)&(frame.method==method)].set_index('split')
                differences.append({'seed': seed, 'model': model, 'method': method,
                    'difference_ood_minus_id': values.loc['test', 'coverage']-values.loc['id_test', 'coverage'],
                    'lower': np.quantile(delta, .025), 'upper': np.quantile(delta, .975)})
        for split in ['id_test', 'test']:
            for metric in ['accuracy', 'macro_f1']:
                delta = seed_draws['bert', 'raw', split][metric]-seed_draws['baseline', 'raw', split][metric]
                selected = frame[(frame.method=='raw')&(frame.split==split)].set_index('model')
                paired.append({'seed': seed, 'split': split, 'metric': metric,
                    'bert_minus_baseline': selected.loc['bert', metric]-selected.loc['baseline', metric],
                    'lower': np.quantile(delta, .025), 'upper': np.quantile(delta, .975)})
    multi, ci = pd.concat(all_metrics, ignore_index=True), pd.concat(intervals, ignore_index=True)
    multi.to_csv(directory / 'multiseed_metrics.csv', index=False)
    ci.to_csv(directory / 'cluster_bootstrap_ci.csv', index=False)
    pd.DataFrame(classes).to_csv(directory / 'per_class_metrics.csv', index=False)
    pd.concat(reviewers, ignore_index=True).to_csv(directory / 'reviewer_metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(directory / 'coverage_difference.csv', index=False)
    pd.DataFrame(majority).to_csv(directory / 'majority_baseline.csv', index=False)
    pd.DataFrame(paired).to_csv(directory / 'model_difference.csv', index=False)
    main = ci[ci.metric.isin(['accuracy', 'macro_f1', 'nll', 'ece', 'brier', 'coverage', 'average_set_size', 'reviewer_mean_coverage'])]
    main.to_csv(directory / 'id_ood_comparison.csv', index=False)
    # Average conditional bootstrap draws across fixed seeds using common cluster
    # resampling within each split; this preserves shared-test dependence.
    aggregate_rows, aggregate_differences = [], []
    for model in ['baseline', 'bert']:
        for method in ['raw', 'temperature']:
            for split in ['id_test', 'test']:
                for metric in ['accuracy', 'macro_f1', 'nll', 'ece', 'brier', 'coverage', 'average_set_size', 'reviewer_mean_coverage']:
                    boot = np.mean([draws_by_seed[s][model, method, split][metric] for s in config['seeds']], axis=0)
                    points = ci[(ci.model==model)&(ci.method==method)&(ci.split==split)&(ci.metric==metric)].estimate
                    aggregate_rows.append(dict(model=model, method=method, split=split, metric=metric,
                        mean=points.mean(), seed_sd=points.std(ddof=1), lower=np.quantile(boot, .025), upper=np.quantile(boot, .975)))
            boots = [draws_by_seed[s][model, method, 'test']['coverage']-draws_by_seed[s][model, method, 'id_test']['coverage'] for s in config['seeds']]
            boot = np.mean(boots, axis=0)
            points = [r['difference_ood_minus_id'] for r in differences if r['model']==model and r['method']==method]
            aggregate_differences.append(dict(model=model, method=method, difference_ood_minus_id=np.mean(points),
                seed_sd=np.std(points, ddof=1), lower=np.quantile(boot, .025), upper=np.quantile(boot, .975)))
    summary = pd.DataFrame(aggregate_rows)
    summary.to_csv(directory / 'seed_summary.csv', index=False)
    pd.DataFrame(aggregate_differences).to_csv(directory / 'mean_coverage_difference.csv', index=False)
    model_comparison = []
    for split in ['id_test', 'test']:
        for metric in ['accuracy', 'macro_f1']:
            boot = np.mean([draws_by_seed[s]['bert', 'raw', split][metric]-draws_by_seed[s]['baseline', 'raw', split][metric] for s in config['seeds']], axis=0)
            points = [r['bert_minus_baseline'] for r in paired if r['split']==split and r['metric']==metric]
            model_comparison.append(dict(split=split, metric=metric, bert_minus_baseline=np.mean(points),
                seed_sd=np.std(points, ddof=1), lower=np.quantile(boot, .025), upper=np.quantile(boot, .975)))
    comparison = pd.DataFrame(model_comparison)
    comparison.to_csv(directory / 'mean_model_difference.csv', index=False)
    plot_stage2(ci, pd.concat(reviewers, ignore_index=True), pd.DataFrame(differences), directory)
    write_stage2_report(directory, config, multi, summary, pd.DataFrame(differences),
                        pd.DataFrame(aggregate_differences), comparison, pd.concat(reviewers, ignore_index=True))
    print('Statistical analysis complete:', directory, flush=True)


def write_stage2_report(directory, config, multi, summary, differences, mean_difference, comparison, reviewers):
    def md(frame):
        def cell(v):
            return f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
        return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None))
    selected = summary[summary.method=='temperature'].pivot(index=['model', 'split'], columns='metric', values='mean').reset_index()
    selected = selected[['model', 'split', 'accuracy', 'macro_f1', 'nll', 'ece', 'brier', 'coverage', 'average_set_size']]
    seeds = []
    for seed in config['seeds']:
        for model in ['baseline', 'bert']:
            part = multi[(multi.seed==seed)&(multi.model==model)&(multi.method=='temperature')].set_index('split')
            diff = differences[(differences.seed==seed)&(differences.model==model)&(differences.method=='temperature')].iloc[0]
            seeds.append(dict(seed=seed, model=model, ID_acc=part.loc['id_test', 'accuracy'], OOD_acc=part.loc['test', 'accuracy'],
                ID_F1=part.loc['id_test', 'macro_f1'], OOD_F1=part.loc['test', 'macro_f1'],
                ID_coverage=part.loc['id_test', 'coverage'], OOD_coverage=part.loc['test', 'coverage'],
                OOD_minus_ID=diff.difference_ood_minus_id, CI_low=diff.lower, CI_high=diff.upper))
    stability = []
    for model in ['baseline', 'bert']:
        part = differences[(differences.model==model)&(differences.method=='temperature')]
        stability.append(f'- {model}：三个种子中 {(part.difference_ood_minus_id<0).sum()}/3 个 OOD Coverage 点估计低于 ID，{(part.upper<0).sum()}/3 个逐项95%区间完全低于0。是否稳定应同时看点估计、区间和种子变动，不能只看均值。')
    reviewer_counts = reviewers[(reviewers.seed==42)&(reviewers.model=='baseline')&(reviewers.method=='raw')].groupby('split').agg(reviewers=('reviewer', 'size'), median_reviews=('n', 'median'), min_reviews=('n', 'min'), max_reviews=('n', 'max')).reset_index()
    majority = pd.read_csv(directory / 'majority_baseline.csv')
    reference = majority[majority.seed==42][['split', 'majority_label', 'metric', 'estimate', 'lower', 'upper']]
    per_class = pd.read_csv(directory / 'per_class_metrics.csv')
    class_table = per_class[(per_class.method=='temperature')&(per_class.split=='test')].groupby(['model', 'stars']).agg(
        support=('support', 'first'), precision=('precision', 'mean'), recall=('recall', 'mean'), coverage=('coverage', 'mean')).reset_index()
    probability = summary[summary.metric.isin(['nll', 'ece', 'brier'])].pivot(index=['model', 'method', 'split'], columns='metric', values='mean').reset_index()
    conclusions = []
    for model in ['baseline', 'bert']:
        row = mean_difference[(mean_difference.model==model)&(mean_difference.method=='temperature')].iloc[0]
        status = '跨零，尚不足以确证稳定下降' if row.lower<=0<=row.upper else '未跨零，需结合探索性设计与条件推断限制解读'
        conclusions.append(f'- {model} 的平均 OOD minus ID Coverage 为 {row.difference_ood_minus_id*100:.2f} 个百分点，95% CI [{row.lower*100:.2f}, {row.upper*100:.2f}] 个百分点；{status}。')
    for metric in ['accuracy', 'macro_f1']:
        part = comparison[comparison.metric==metric]
        conclusion = '两个划分的区间均高于0' if (part.lower>0).all() else '存在跨零区间，不能仅凭均值声称稳定超过LR'
        conclusions.append(f'- BERT minus LR 的 {metric}：{conclusion}。')
    text = f'''# CalibShift-NLP 第二阶段实验报告

所有表格与图表来自本机真实实验。第一阶段结果保持不变；核对原归档中22个结果文件的哈希，见 phase1_preservation.json。

## 主要发现

{chr(10).join(conclusions)}

## 实验完成情况与划分

先完成3000条 seed42 中等规模实验，仅查看 OOD val，流程通过后按冻结参数训练三个10000条正式实验。种子为 {config['seeds']}，三个种子的 DistilBERT 均完成2个 epoch，LR按固定配置优化；temperature/conformal 各2000，ID test、OOD val、OOD test 各3000。使用现有 Windows RTX3070 CUDA 环境、原始两个模型、原始超参数；全程复用本地数据/预训练缓存。逐项命令、退出码与用时见 execution.jsonl，各 run 的环境/配置/训练日志均保留。

第一阶段确实查看过 test/id_test 各350行。第二阶段排除这些行，并在正式运行前固定新的评估行；但研究过程已经接触同一官方测试分布，因此视为探索性验证，不宣称完全未接触数据集的确认性检验。三个种子共享相同的3000条 ID 和3000条 OOD 测试行，不是9000独立样本。训练/校准抽样随种子改变；medium 的3000训练行是 seed42 正式训练子集的前3000行。

train 仅用于训练；id_val 随机等分为两个不重叠校准子集；val 用于开发描述；id_test/test 只作冻结后的最终评估。温度与分位数不访问最终测试标签。官方 reviewer 交集和逐行索引校验见 split_audit.json、各 sample_manifest.csv；ID 训练域 reviewer 重叠按官方设计保留，OOD reviewer 与训练/校准不重叠。

## 最终 ID/OOD 指标（三种子均值）

下表为 Temperature Scaling 概率及在这些概率上构造的 LAC。各类 Precision/Recall/F1/Coverage 见 per_class_metrics.csv；相应置信区间见 cluster_bootstrap_ci.csv。

{md(selected)}

## 三个种子与 Coverage Difference

差值为 OOD minus ID，负值代表 OOD 下降，CI 为每个已拟合模型条件下的 reviewer-cluster percentile 区间。

{md(pd.DataFrame(seeds))}

{chr(10).join(stability)}

三种子均值差值（同时给出原始概率和温度缩放）：

{md(mean_difference)}

均值区间使用相同 reviewer 重采样权重同步重算三个模型，以保留共享测试集的相关性；seed_sd 为三次运行的样本标准差。区间不包含重新训练和重拟合校准器的全部不确定性。

## 概率校准比较

LAC 是集合构造方法，不产生另一套概率；概率比较只针对 raw 和 temperature。两者的 argmax/Accuracy/Macro-F1 相同，集合阈值分别在相同 conformal 行上计算。

{md(probability)}

## DistilBERT 是否超过 Logistic Regression

以下为三个种子差值的平均，使用同一评估集的配对 reviewer 重采样，正值表示 BERT 较高。请分别判断 Accuracy 和 Macro-F1，不把某一个指标的改善当作所有指标均改善。

{md(comparison)}

多数类参考使用训练标签确定，下面 seed42 的参考对相同测试行进行评估（majority_label=4 对应五星）；其他种子结果在 majority_baseline.csv，不拟合额外模型，不给多数类参考虚构概率校准或 CP：

{md(reference)}

## Reviewer 层面与集合大小

{md(reviewer_counts)}

reviewer_metrics.csv 给出每位 reviewer 的 Coverage、观察评论数、Accuracy、Set Size。reviewer_coverage.png 为各 reviewer 跨种子平均覆盖率的等权 ECDF；它不是逐 reviewer 条件覆盖保证。单 reviewer 可能仅有少数观察评论，0或1的经验覆盖率应谨慎解读。综合表同时给出评论加权 Coverage 和 reviewer_mean_coverage。

OOD 各类别结果（温度缩放，三种子均值；support 为共享测试样本数，不乘以3）：

{md(class_table)}

总体覆盖率会掩盖低星级类别的明显不足。若观察到某类别零次覆盖，非参数 Bootstrap 可能给出退化的[0,0]区间，这不能证明该类别总体覆盖概率恰为0；需结合支持数与有限样本限制解读。

Average Set Size 介于0与5之间，与类别数量及 Coverage 一起解释；小集合只有在覆盖率充分时才更有效率。即使大小看起来接近1–2，也不能据此宣称达到90%目标。coverage_ci.png、set_size_ci.png、coverage_difference_ci.png 展示三种子逐项区间。

## 统计适用条件与限制

2000次 reviewer 聚类 Bootstrap 假设 reviewer 间近似独立、代表目标总体，保留已观察 reviewer 内的相关性。共享商品或时间依赖、样本选择、稀有类别、少量 reviewer 都可能使推断失真；这些区间是逐项未做多重校正的探索性区间。它们条件于已拟合模型和温度/分位数，不包含校准估计不确定性。3个 seed 的变动还混合了训练和校准抽样变化，不能仅归因于训练初始化。样本按评论均匀抽取而非按 reviewer 平衡，reviewer 等权结果仅描述观察到的 reviewer；该 Bootstrap 是近似超总体推断，不是对有限官方数据集的精确抽样设计区间。TF-IDF 使用完整文本而 BERT 截断128token，模型比较也包含预处理差异。

90%是可交换性等条件下的边际集合覆盖目标，不是逐样本正确率，不保证各类别或各 reviewer 的90%覆盖。官方ID标签不证明严格可交换；OOD分布偏移下不能沿用标准保证。

## 下一步研究问题

1. Coverage 差异来自评分先验变化、置信度偏移还是 reviewer 异质性？结合各类支持数与 Coverage 做描述性分解。
2. 类别不平衡下，Accuracy、Macro-F1 与小集合效率的取舍是否跨更多预先指定种子稳定？
3. 用 reviewer 为单位拆分校准集、重新拟合校准器后的覆盖率不确定性，与当前条件区间相差多少？
4. 商品与时间的跨 reviewer 相关是否要求多维聚类或其他重采样设计？

详细事先冻结方案见 ANALYSIS_PROTOCOL.md，所有原始多种子指标见 multiseed_metrics.csv，汇总均值/seed SD/条件CI见 seed_summary.csv。本阶段未使用最终测试结果选择超参数或统计方法。
'''
    (directory / 'EXPERIMENT_REPORT.md').write_text(text, encoding='utf-8')
