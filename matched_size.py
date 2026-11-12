"""Matched-size LAC comparison; frozen models, temperatures and evaluation rows."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from src.calibration import (probabilities, validate_probs, lac_quantile, prediction_sets,
                             class_conditional_quantiles, class_conditional_sets)
from src.evaluate import class_set_bootstrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'matched_size'
SEEDS = [42, 123, 2026]
MODELS = ['baseline', 'bert']
METHODS = ['marginal', 'class_conditional', 'matched_marginal']
METRICS = ['coverage', 'average_set_size', 'worst_class_coverage', 'macro_class_coverage'] + [f'class_coverage_{k}' for k in range(5)]


def match_marginal_size(calibration_probs, calibration_labels, validation_probs, target_total):
    """Exhaustive attainable-threshold search; NO validation/evaluation labels.

    Compare integer label-inclusion counts to avoid rounding-dependent ties.
    Tie order: absolute count error, smaller count, smaller threshold (first
    attainable rank for equal calibration scores). Alpha is the midpoint of
    the finite-sample rank's alpha interval, not a tuned continuous grid.
    """
    p = validate_probs(calibration_probs)
    val = validate_probs(validation_probs)
    y = np.asarray(calibration_labels)
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer) or ((y < 0) | (y > 4)).any():
        raise ValueError('Expected aligned integer calibration labels 0..4')
    if not isinstance(target_total, (int, np.integer)) or not 0 <= target_total <= val.size:
        raise ValueError('Target must be an attainable integer inclusion count range')
    scores = np.sort(1-p[np.arange(len(y)), y])
    thresholds, indices = np.unique(scores, return_index=True)
    thresholds = np.r_[thresholds, np.inf]
    ranks = np.r_[indices+1, len(scores)+1]
    validation_scores = np.sort((1-val).ravel())
    totals = np.searchsorted(validation_scores, thresholds, side='right')
    order = np.lexsort((ranks, totals, np.abs(totals-target_total)))
    selected = int(order[0])
    rank, q = int(ranks[selected]), float(thresholds[selected])
    alpha = float(1-(rank-.5)/(len(scores)+1))
    assert lac_quantile(p, y, alpha) == q
    return dict(alpha=alpha, rank=rank, qhat=q, candidate_count=len(thresholds),
        alpha_interval_lower=max(0., 1-rank/(len(scores)+1)),
        alpha_interval_upper=1-(rank-1)/(len(scores)+1),
        target_total=int(target_total), matched_total=int(totals[selected]),
        target_size=target_total/len(val), matched_size=float(totals[selected]/len(val)),
        signed_gap=float((totals[selected]-target_total)/len(val)),
        exact_match=bool(totals[selected] == target_total), validation_n=len(val))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bounds(draw):
    valid = np.asarray(draw)[np.isfinite(draw)]
    return np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=SEEDS, models=MODELS, base_alpha=.1, probability='saved temperature scaled',
        selection_split='val', evaluation_split='test', validation_labels_used_for_selection=False,
        selection='all distinct calibration score thresholds plus infinity',
        objective='absolute integer validation inclusion-count discrepancy',
        tie_break='smaller inclusion count, then smaller threshold/first attainable rank',
        alpha='midpoint of rank interval [1-r/(n+1), 1-(r-1)/(n+1)) intersect (0,1)',
        bootstrap_replicates=2000, bootstrap_seed=814, paired_reviewers=True,
        shared_test_rows_across_seeds=True, training=False, new_sampling=False,
        inference='conditional on models, temperatures, calibration scores and validation-selected alpha',
        test_status='previously examined; exploratory, not independent confirmation')
    frozen = OUT/'protocol.json'
    encoded = json.dumps(protocol, indent=2)
    if frozen.exists() and frozen.read_text() != encoded:
        raise ValueError('Refusing to change the matched-size protocol silently')
    frozen.write_text(encoded)
    prior = [p for folder in ['results/smoke', 'results/full', 'results/stage2', 'results/stage3', 'results/final_audit', 'models/stage2']
             for p in (ROOT/folder).rglob('*') if p.is_file()]
    prior += [ROOT/p for p in ['README.md', 'FINAL_RESEARCH_REPORT.pdf', 'FINAL_RESEARCH_REPORT.md', 'FINAL_RESULTS_SUMMARY.md']]
    old_hashes = {str(p.relative_to(ROOT)): digest(p) for p in prior}
    original = OUT/'preserved_input_hashes.json'
    if original.exists() and json.loads(original.read_text()) != old_hashes:
        raise ValueError('Earlier experiment inputs changed')
    original.write_text(json.dumps(old_hashes, indent=2))
    matches, calibrated = [], {}
    # Phase 1: do not access validation labels or ANY test arrays until all six
    # matching choices have been fixed and written to disk.
    for seed in SEEDS:
        run = ROOT/'results/stage2'/f'formal_seed{seed}'
        ts = json.loads((run/'calibration.json').read_text())
        for model in MODELS:
            cache = np.load(ROOT/f'results/stage3/predictions_calibration_{model}_seed{seed}.npz')
            pcal, ycal = probabilities(cache['logits'], ts[model]['temperature']), cache['labels']
            pval = np.load(run/f'predictions_{model}_temperature_val.npz')['probabilities']
            q = lac_quantile(pcal, ycal, .1)
            cc = class_conditional_quantiles(pcal, ycal, .1)
            target = int(class_conditional_sets(pval, cc).sum())
            match = match_marginal_size(pcal, ycal, pval, target)
            matches.append(dict(seed=seed, model=model, **match))
            calibrated[seed, model] = (pcal, ycal, q, cc, match)
    match_file = OUT/'matching_choices.csv'
    match_frame = pd.DataFrame(matches)
    if match_file.exists():
        pd.testing.assert_frame_equal(pd.read_csv(match_file), match_frame, check_dtype=False, rtol=1e-12)
    match_frame.to_csv(match_file, index=False)
    print('All six alpha choices locked using validation probabilities and size only.', flush=True)
    rows, summaries, differences, curves = [], [], [], []
    draws, points = {}, {}
    fixed_rows = {}
    for seed in SEEDS:
        run = ROOT/'results/stage2'/f'formal_seed{seed}'
        manifest = pd.read_csv(run/'sample_manifest.csv', dtype={'original_user': str})
        assert manifest['index'].is_unique
        for model in MODELS:
            pcal, ycal, q, cc, match = calibrated[seed, model]
            ordered = np.sort(1-pcal[np.arange(len(ycal)), ycal])
            curve_q = np.r_[ordered, np.inf]
            curve_alpha = 1-(np.arange(1, len(ordered)+2)-.5)/(len(ordered)+1)
            for split in ['val', 'test']:
                saved = np.load(run/f'predictions_{model}_temperature_{split}.npz')
                p, y = saved['probabilities'], saved['labels']
                frame = manifest[manifest.split == split]
                assert np.array_equal(y, frame.label.to_numpy())
                if split in fixed_rows:
                    assert np.array_equal(frame['index'].to_numpy(), fixed_rows[split])
                fixed_rows[split] = frame['index'].to_numpy()
                ids = frame.original_user.to_numpy()
                nc, ne = np.bincount(ycal, minlength=5), np.bincount(y, minlength=5)
                options = dict(marginal=prediction_sets(p, q), class_conditional=class_conditional_sets(p, cc),
                               matched_marginal=prediction_sets(p, match['qhat']))
                assert np.array_equal(options['marginal'], saved['sets'])
                stage3 = np.load(ROOT/f'results/stage3/predictions_{model}_temperature_class_conditional_seed{seed}_{split}.npz')
                assert np.array_equal(options['class_conditional'], stage3['sets'])
                for method, sets in options.items():
                    ci, boot = class_set_bootstrap(y, sets, ids, replicates=2000, seed=816 if split == 'val' else 815)
                    ci = ci.set_index('metric')
                    key = (seed, model, split, method)
                    draws[key] = {metric: boot[metric] for metric in METRICS}
                    points[key] = {metric: float(ci.loc[metric, 'estimate']) for metric in METRICS}
                    row = dict(seed=seed, model=model, split=split, method=method,
                        alpha=match['alpha'] if method == 'matched_marginal' else .1,
                        n=len(y), reviewers=len(np.unique(ids)), calibration_n=len(ycal),
                        **{f'n_rating_{k+1}': int(ne[k]) for k in range(5)},
                        **{f'calibration_n_rating_{k+1}': int(nc[k]) for k in range(5)})
                    for metric in METRICS:
                        row[metric] = points[key][metric]
                        row[metric+'_lower'], row[metric+'_upper'] = ci.loc[metric, ['lower', 'upper']]
                    rows.append(row)
                # Descriptive curves are computed ONLY AFTER all choices locked.
                # Every finite rank is represented, including duplicate-score plateaus.
                true_scores = 1-p[np.arange(len(y)), y]
                total = np.searchsorted(np.sort((1-p).ravel()), curve_q, side='right')
                covered = np.searchsorted(np.sort(true_scores), curve_q, side='right')
                one = np.searchsorted(np.sort(true_scores[y == 0]), curve_q, side='right')
                for i in range(len(curve_q)):
                    curves.append(dict(seed=seed, model=model, split=split, rank=i+1, alpha=curve_alpha[i],
                        qhat=curve_q[i], average_set_size=total[i]/len(y), coverage=covered[i]/len(y),
                        one_star_coverage=one[i]/ne[0]))
    for model in MODELS:
        for split in ['val', 'test']:
            for method in METHODS:
                entry = dict(model=model, split=split, method=method)
                for metric in METRICS:
                    values = [points[s, model, split, method][metric] for s in SEEDS]
                    mean_draw = np.mean([draws[s, model, split, method][metric] for s in SEEDS], axis=0)
                    entry[metric] = np.mean(values)
                    entry[metric+'_seed_sd'] = np.std(values, ddof=1)
                    entry[metric+'_lower'], entry[metric+'_upper'] = bounds(mean_draw)
                summaries.append(entry)
            for a, b in [('class_conditional', 'matched_marginal'), ('matched_marginal', 'marginal'), ('class_conditional', 'marginal')]:
                for metric in METRICS:
                    for seed in SEEDS+[None]:
                        selected = [seed] if seed is not None else SEEDS
                        delta = np.mean([draws[s, model, split, a][metric]-draws[s, model, split, b][metric] for s in selected], axis=0)
                        lo, hi = bounds(delta)
                        differences.append(dict(seed='mean' if seed is None else str(seed), model=model, split=split,
                            contrast=a+' minus '+b, metric=metric,
                            estimate=np.mean([points[s, model, split, a][metric]-points[s, model, split, b][metric] for s in selected]),
                            lower=lo, upper=hi, bootstrap_valid=int(np.isfinite(delta).sum())))
    pd.DataFrame(rows).to_csv(OUT/'matched_size_metrics.csv', index=False)
    summary, difference, curve = pd.DataFrame(summaries), pd.DataFrame(differences), pd.DataFrame(curves)
    summary.to_csv(OUT/'matched_size_summary.csv', index=False)
    difference.to_csv(OUT/'paired_differences.csv', index=False)
    curve.to_csv(OUT/'coverage_efficiency_curve.csv', index=False)
    plot_results(summary, curve)
    write_report(match_frame, summary, difference)
    assert all(digest(ROOT/path) == value for path, value in old_hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(status='complete', earlier_files_preserved=len(old_hashes),
        choices_locked_before_evaluation=True, matching_used_validation_labels=False, evaluation_rows_unchanged=True,
        all_six_choices_globally_optimal_on_attainable_thresholds=True, training=False,
        bootstrap_replicates=2000, shared_seed_test_dependence_preserved=True), indent=2))
    print(summary[summary.split == 'test'][['model', 'method']+METRICS].to_string(index=False), flush=True)


def plot_results(summary, curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Marginal (.10)', 'Class-conditional (.10)', 'Matched marginal']
    colors = ['#666666', '#d95f02', '#1b70a6']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    means = curve.groupby(['model', 'split', 'rank'], as_index=False)[['average_set_size', 'coverage', 'one_star_coverage']].mean()
    for j, model in enumerate(MODELS):
        for i, metric in enumerate(['coverage', 'one_star_coverage']):
            ax = axes[i, j]
            for split, style in [('val', '--'), ('test', '-')]:
                part = means[(means.model == model)&(means.split == split)]
                ax.plot(part.average_set_size, part[metric], style, color='#4b6280' if split == 'test' else '#b1b1b1', label='Marginal curve: '+split)
            for method, label, color in zip(METHODS, labels, colors):
                r = summary[(summary.model == model)&(summary.split == 'test')&(summary.method == method)].iloc[0]
                ax.scatter(r.average_set_size, r['coverage' if i == 0 else 'class_coverage_0'], c=color, s=45, label=label, zorder=3)
            ax.set(xlim=(0, 5), ylim=(-.02, 1.04), xlabel='Average set size', ylabel='Overall coverage' if i == 0 else 'One-star coverage', title=model)
            ax.legend(fontsize=7, loc='lower right')
    fig.savefig(OUT/'coverage_efficiency_curve.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        for method, label, color, offset in zip(METHODS, labels, colors, [-.15, 0, .15]):
            row = summary[(summary.model == model)&(summary.split == 'test')&(summary.method == method)].iloc[0]
            metrics = ['class_coverage_0', 'worst_class_coverage']
            point = np.array([row[m] for m in metrics])
            low, high = np.array([row[m+'_lower'] for m in metrics]), np.array([row[m+'_upper'] for m in metrics])
            ax.errorbar(np.arange(2)+offset, point, yerr=np.maximum(0, [point-low, high-point]), fmt='o', color=color,
                capsize=4, label=f'{label}; size={row.average_set_size:.3f}')
        ax.set(xticks=[0, 1], xticklabels=['One-star', 'Worst-class'], ylim=(-.02, 1.04), ylabel='Coverage (conditional 95% cluster CI)', title=model+' | OOD test')
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.11))
    fig.savefig(OUT/'matched_size_comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        for method, label, color, offset in zip(METHODS, labels, colors, [-.08, 0, .08]):
            row = summary[(summary.model == model)&(summary.split == 'test')&(summary.method == method)].iloc[0]
            keys = [f'class_coverage_{k}' for k in range(5)]
            point = np.array([row[k] for k in keys])
            low, high = np.array([row[k+'_lower'] for k in keys]), np.array([row[k+'_upper'] for k in keys])
            ax.errorbar(np.arange(1, 6)+offset, point, yerr=np.maximum(0, [point-low, high-point]), fmt='o-', capsize=3, color=color, label=label)
        ax.set(xticks=range(1, 6), ylim=(-.02, 1.04), xlabel='True star rating', ylabel='Coverage (conditional 95% cluster CI)', title=model+' | OOD test')
        ax.legend(fontsize=8)
    fig.savefig(OUT/'classwise_coverage.png', dpi=180); plt.close(fig)


def write_report(matches, summary, differences):
    def md(frame):
        def cell(v): return f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
        return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(cell, row))+' |' for row in frame.itertuples(index=False, name=None))
    display = summary[['model', 'split', 'method']+METRICS]
    primary = differences[(differences.seed == 'mean')&(differences.split == 'test')&(differences.contrast == 'class_conditional minus matched_marginal')]
    sample = pd.read_csv(ROOT/'results/stage3/sample_counts.csv')
    counts = sample[(sample.seed == 42)&sample.split.isin(['val', 'test'])][['split', 'stars', 'n', 'reviewers']]
    findings = []
    for model in MODELS:
        part = primary[primary.model == model].set_index('metric')
        selected = summary[(summary.model == model)&(summary.split == 'test')].set_index('method')
        row = part.loc['class_coverage_0']
        conclusion = 'CI entirely positive' if row.lower > 0 else ('CI entirely negative' if row.upper < 0 else 'CI includes zero; no clear residual advantage')
        findings.append(f'- {model}: matched marginal one-star coverage={selected.loc["matched_marginal", "class_coverage_0"]:.4f}; class-conditional={selected.loc["class_conditional", "class_coverage_0"]:.4f}. Difference={row.estimate:.4f}, 95% CI [{row.lower:.4f}, {row.upper:.4f}]; {conclusion}.')
        worst = part.loc['worst_class_coverage']
        overall = part.loc['coverage']
        findings.append(f'- {model}: worst-class coverage changes from {selected.loc["matched_marginal", "worst_class_coverage"]:.4f} (matched marginal) to {selected.loc["class_conditional", "worst_class_coverage"]:.4f} (class-conditional); paired difference {worst.estimate:.4f}, CI [{worst.lower:.4f}, {worst.upper:.4f}].')
        findings.append(f'- {model}: the minority gain costs overall coverage: matched marginal={selected.loc["matched_marginal", "coverage"]:.4f}, class-conditional={selected.loc["class_conditional", "coverage"]:.4f}, difference={overall.estimate:.4f}, CI [{overall.lower:.4f}, {overall.upper:.4f}].')
        for k in [3,4]:
            loss = part.loc[f'class_coverage_{k}']
            findings.append(f'- {model}, rating {k+1}: class-conditional minus matched marginal coverage={loss.estimate:.4f}, CI [{loss.lower:.4f}, {loss.upper:.4f}].')
        findings.append(f'- {model}: actual mean test size is {selected.loc["matched_marginal", "average_set_size"]:.6f} versus {selected.loc["class_conditional", "average_set_size"]:.6f}; signed matched-minus-class-conditional gap={selected.loc["matched_marginal", "average_set_size"]-selected.loc["class_conditional", "average_set_size"]:.6f}.')
    text=f'''# Matched Set Size and Coverage-Efficiency Trade-off Analysis

## Scope and fixed protocol

This is one supplementary statistical contrast, with no model retraining, new evaluation sampling, README/PDF changes, commit or push. It reuses saved temperature-scaled probabilities for LR/DistilBERT, seeds 42/123/2026, existing conformal data and true reviewer IDs. Methods are A: marginal LAC alpha=.10; B: class-conditional LAC alpha=.10; C: matched-size marginal LAC with alpha chosen from validation set sizes only. All six choices were written to matching_choices.csv before any evaluation labels or test arrays were accessed by this script.

All attainable marginal thresholds are the distinct true-label calibration scores plus infinity. At each threshold, validation inclusion counts are computed from probabilities alone. Minimize the absolute integer count discrepancy from method B; ties select the smaller inclusion count, then the smaller threshold/first attainable rank. A representative alpha is the midpoint of the rank's interval, ensuring the original finite-sample quantile reproduces that threshold. This exhaustive discrete search does not fabricate an interpolated threshold or exact match.

Validation labels, class coverage and final test results are not inputs to matching. Calibration labels are used only for the required conformal scores. Descriptive validation/test coverage curves are generated after all selections are locked and do not feed back into alpha selection. Curves average fitted-seed statistics at the same finite-sample rank/alpha; they are not an ensemble classifier. C has a different alpha and nominal target from B; validation selection and OOD shift prevent an automatic conformal guarantee.

## Matching on OOD validation

{md(matches[['seed','model','alpha','rank','candidate_count','target_size','matched_size','signed_gap','exact_match']])}

The validation gap is matched marginal minus class-conditional size. These choices remain fixed on test, where a new gap may appear. Mean test sizes and their paired difference are reported below. Very close means do not imply identical per-review sizes or label allocations.

## True sample sizes

{md(counts)}

The fixed validation/test samples contain 3000 comments each. Seeds reuse exactly the same comments; they do not yield 9000 independent observations. The one-star validation/test supports above are not multiplied by three. Per-seed class calibration counts are in matched_size_metrics.csv.

## Real results: three fitted-seed means

{md(display)}

Coverage columns are proportions; class_coverage_0..4 correspond to ratings 1..5. Worst-class is computed per fitted seed then averaged, not the minimum of already averaged class coverages. Macro-class coverage gives the five ratings equal weight. Per-seed estimates and all conditional intervals appear in matched_size_metrics.csv; seed-averaged intervals in matched_size_summary.csv.

## Primary matched contrast on OOD test

{chr(10).join(findings)}

Thus a residual minority/worst-class advantage remains in this nearly size-matched comparison, but class-conditional LAC does not dominate matched marginal LAC on overall or frequent-rating coverage. The conclusion is a trade-off in label allocation, not general efficiency superiority. All six validation gaps are nonzero; exact matching was impossible on their attainable threshold grids. The small test gaps are observed outcomes, not forced matches or formal equivalence tests.

Class-conditional minus matched marginal, using paired reviewer-cluster resampling:

{md(primary[['model','metric','estimate','lower','upper','bootstrap_valid']])}

The one-star and worst-class rows answer residual minority reliability at the observed size budget. Class_coverage_1..4 and overall coverage identify gains/losses elsewhere. Read these alongside the average_set_size row: validation matching does not force equal test sizes. The other pre-specified contrasts and all per-seed differences are retained in paired_differences.csv.

## Inference and limitations

The same 2000-replicate reviewer-cluster percentile protocol is used as in Stage 3. Every selected reviewer brings all observed comments with a shared multiplicity. Methods and seeds share bootstrap weights within each split, preserving pairing and dependence from shared test comments; validation and test use separate streams. The mean interval averages the three fixed models under common cluster draws, rather than treating them as independent datasets.

Intervals condition on model weights, temperature parameters, conformal scores/thresholds AND the validation-selected alpha/size target. They omit training, calibration-refitting and validation-selection uncertainty. Reviewer-cluster independence/representativeness is approximate; shared products, variable cluster sizes and sparse ratings remain limitations. Zero events can give degenerate [0,0] bootstrap bounds, not population certainty. Worst-class minima are nonsmooth; their intervals are descriptive and can be nonregular near ties. All intervals are pointwise without multiplicity correction.

OOD test was previously examined. This fixed supplementary analysis is exploratory, not independent confirmation. Any remaining difference describes label allocation at nearly matched *average* size, not a new theoretical guarantee, exact per-sample budget equivalence, or universal superiority. There is no assumption that class-conditional must win, and no alpha is selected using the desired conclusion.

## Figures and reproducibility

- [Coverage-efficiency curve](coverage_efficiency_curve.png): all attainable marginal ranks; validation/test curves with the three fixed test-method points.
- [Matched-size comparison](matched_size_comparison.png): one-star/worst-class test coverage, conditional CIs and actual test sizes.
- [Classwise coverage](classwise_coverage.png): test coverage for every rating.
- [Per-seed metrics](matched_size_metrics.csv), [seed-averaged metrics](matched_size_summary.csv), [matching choices](matching_choices.csv) and [paired differences](paired_differences.csv).

Run `python matched_size.py` from the existing project with local caches. This script does not import a training entry point or initiate inference/downloads; missing caches fail explicitly. protocol.json and preserved_input_hashes.json document the selection rules and unchanged earlier inputs. No new experiment stage is proposed.
'''
    (OUT/'MATCHED_SIZE_REPORT.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
