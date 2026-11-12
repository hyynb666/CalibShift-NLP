from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(metrics, curves, directory):
    path = Path(directory)
    if not {'id_test', 'test'}.issubset(set(metrics.split)):
        return  # Development-only medium run; final-test plots are intentionally deferred.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, split, title in zip(axes, ['id_test', 'test'], ['ID test', 'OOD test']):
        ax.plot([0, 1], [0, 1], '--', color='gray')
        for (model, method), group in curves[curves.split == split].groupby(['model', 'method']):
            ax.plot(group.confidence, group.accuracy, 'o-', label=f'{model}: {method}')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean confidence', ylabel='Empirical accuracy', title=title)
        ax.legend(fontsize=7)
    fig.savefig(path / 'calibration.png', dpi=180)
    plt.close(fig)
    for metric, label, filename in [('coverage', 'Marginal empirical coverage', 'coverage.png'),
                                     ('average_set_size', 'Mean prediction set size', 'set_size.png')]:
        selected = metrics[metrics.split.isin(['id_test', 'test'])].copy()
        selected['setting'] = selected.model + ' / ' + selected.method
        table = selected.pivot(index='setting', columns='split', values=metric)
        ax = table.plot.bar(figsize=(9, 5), rot=15)
        ax.set(ylabel=label, xlabel='', title='LAC: ID versus OOD (test)')
        if metric == 'coverage':
            ax.axhline(float(selected.target_coverage.iloc[0]), color='black', linestyle='--', label='Target')
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, 5.2)
        ax.legend()
        ax.figure.tight_layout()
        ax.figure.savefig(path / filename, dpi=180)
        plt.close(ax.figure)


def plot_stage2(intervals, reviewers, differences, directory):
    import numpy as np
    path = Path(directory)
    final = intervals[intervals.split.isin(['id_test', 'test'])]
    for metric, ylabel, filename in [('coverage', 'Empirical LAC coverage (95% reviewer-cluster CI)', 'coverage_ci.png'),
                                     ('average_set_size', 'Mean prediction set size (95% cluster CI)', 'set_size_ci.png')]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, model in zip(axes, ['baseline', 'bert']):
            part = final[(final.model==model)&(final.metric==metric)]
            for j, (method, split) in enumerate([('raw', 'id_test'), ('raw', 'test'), ('temperature', 'id_test'), ('temperature', 'test')]):
                group = part[(part.method==method)&(part.split==split)].sort_values('seed')
                x = np.arange(len(group)) + (j-1.5)*.16
                ax.errorbar(x, group.estimate, yerr=np.maximum(0, np.array([group.estimate-group.lower, group.upper-group.estimate])), fmt='o', capsize=3,
                    label=f'{method}: {"ID" if split=="id_test" else "OOD"}')
            ax.set(xticks=range(3), xticklabels=['42', '123', '2026'], xlabel='Seed', ylabel=ylabel, title=model)
            if metric=='coverage':
                ax.axhline(.9, color='gray', linestyle='--')
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(0, 5)
            ax.legend(fontsize=8)
        fig.savefig(path / filename, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    # Each observed reviewer contributes equally to this descriptive ECDF.
    # Avoid counting shared reviewers three times by averaging over fixed seeds.
    selected = reviewers[(reviewers.method=='temperature')&reviewers.split.isin(['id_test', 'test'])]
    for ax, model in zip(axes, ['baseline', 'bert']):
        for split in ['id_test', 'test']:
            part = selected[(selected.model==model)&(selected.split==split)].groupby('reviewer').coverage.mean().sort_values().to_numpy()
            ax.step(np.r_[-.02, part, 1.02], np.r_[0, np.arange(1, len(part)+1)/len(part), 1],
                    where='post', label=f'{split} ({len(part)} reviewers)')
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel='Reviewer coverage averaged across seeds', ylabel='Fraction of reviewers', title=model)
        ax.legend(fontsize=8)
    fig.savefig(path / 'reviewer_coverage.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for j, (model, method) in enumerate([('baseline', 'raw'), ('baseline', 'temperature'), ('bert', 'raw'), ('bert', 'temperature')]):
        part = differences[(differences.model==model)&(differences.method==method)].sort_values('seed')
        ax.errorbar(np.arange(3)+(j-1.5)*.15, part.difference_ood_minus_id,
            yerr=np.maximum(0, np.array([part.difference_ood_minus_id-part.lower, part.upper-part.difference_ood_minus_id])),
            fmt='o', capsize=3, label=f'{model}: {method}')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set(xticks=range(3), xticklabels=['42', '123', '2026'], xlabel='Seed', ylabel='OOD minus ID coverage (95% cluster CI)')
    ax.legend(fontsize=8)
    fig.savefig(path / 'coverage_difference_ci.png', dpi=180)
    plt.close(fig)


def plot_stage3(summary, directory):
    import numpy as np
    path=Path(directory)
    selected=summary[summary.probability=='temperature']
    for prefix,ylabel,filename,limits in [
        ('class_coverage_', 'Coverage by true rating (95% cluster CI)', 'class_coverage.png', (0,1.05)),
        ('class_size_', 'Mean set size by true rating (95% cluster CI)', 'prediction_set_size.png', (0,5.1))]:
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for i,model in enumerate(['baseline','bert']):
            for j,split in enumerate(['val','id_test','test']):
                ax=axes[i,j]
                for method,offset in [('marginal',-.08),('class_conditional',.08)]:
                    part=selected[(selected.model==model)&(selected.split==split)&(selected.method==method)&selected.metric.str.startswith(prefix)].sort_values('metric')
                    ax.errorbar(np.arange(1,6)+offset,part.estimate,
                        yerr=np.maximum(0,np.array([part.estimate-part.lower,part.upper-part.estimate])),
                        fmt='o-',capsize=3,label=method)
                if prefix=='class_coverage_':
                    ax.axhline(.9,color='gray',linestyle='--',linewidth=1)
                ax.set(xticks=range(1,6),xlabel='True star rating',ylabel=ylabel if j==0 else '',
                       title=f'{model}: {split}'+(' (primary)' if split=='val' else ''),ylim=limits)
                ax.legend(fontsize=8)
        fig.savefig(path/filename,dpi=180)
        plt.close(fig)
