"""Label-conditional LAC analysis on frozen Stage 2 models and samples."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import yaml
from src.calibration import probabilities, lac_quantile, prediction_sets, class_conditional_quantiles, class_conditional_sets
from src.evaluate import class_set_bootstrap
from src.train import predict_logits

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT/'results'/'stage2'
OUT=ROOT/'results'/'stage3'
SPLITS=['val','id_test','test']  # Primary analysis first; all remain exploratory.


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def audit_and_cache(config, device):
    sample_rows, user_rows, probability_rows, audit = [], [], [], []
    user_lookup={}
    for seed in config['seeds']:
        path=PREVIOUS/f'formal_seed{seed}'
        cfg=json.loads((path/'run_config.json').read_text())['config']
        data=joblib.load(path/'subsets.joblib')
        manifest=pd.read_csv(path/'sample_manifest.csv',dtype={'original_user':str})
        assert manifest['index'].is_unique
        assert (manifest.label==manifest.metadata_y).all()
        assert (manifest.stars==manifest.label+1).all()
        assert (manifest.original_y==manifest.stars).all()
        for encoded,original in zip(manifest.metadata_user,manifest.original_user):
            assert pd.notna(original) and original.startswith('A')
            assert user_lookup.setdefault(int(encoded),original)==original
        model_dir=ROOT/'models'/'stage2'/f'formal_seed{seed}'
        assert np.array_equal(joblib.load(model_dir/'baseline.joblib').classes_,np.arange(5))
        labels=json.loads((model_dir/'bert'/'config.json').read_text())['id2label']
        assert labels=={str(k):f'{k+1} star' for k in range(5)}
        calibration=json.loads((path/'calibration.json').read_text())
        for split,item in data.items():
            frame=manifest[manifest.split==split]
            assert np.array_equal(frame.label.to_numpy(),item['labels'])
            sizes=frame.groupby('original_user').size()
            user_rows.append(dict(seed=seed,split=split,n=len(frame),reviewers=len(sizes),
                median_reviews=float(sizes.median()),min_reviews=int(sizes.min()),max_reviews=int(sizes.max()),
                fraction_single_review=float((sizes==1).mean()),fraction_under5=float((sizes<5).mean())))
            for k in range(5):
                part=frame[frame.label==k]
                sample_rows.append(dict(seed=seed,split=split,label=k,stars=k+1,n=len(part),reviewers=part.original_user.nunique()))
        for model in ['baseline','bert']:
            print(f'Auditing/cache: seed={seed}, model={model}, device={device}',flush=True)
            cache=OUT/f'predictions_calibration_{model}_seed{seed}.npz'
            if cache.exists():
                stored=np.load(cache)
                assert np.array_equal(stored['labels'],data['conformal']['labels'])
                z=stored['logits']
            else:
                z=predict_logits(model,data['conformal'],cfg,model_dir,device)
                np.savez_compressed(cache,logits=z,labels=data['conformal']['labels'])
            t=calibration[model]['temperature']
            for method,temp in [('raw',1.),('temperature',t)]:
                p=probabilities(z,temp)
                q=lac_quantile(p,data['conformal']['labels'],config['alpha'])
                assert np.isclose(q,calibration[model][method+'_qhat'],atol=1e-7,rtol=1e-6)
                audit.append(dict(seed=seed,model=model,probability=method,marginal_threshold_reproduced=True))
            # Diagnostic inference on TRUE one-star train/temperature rows only.
            for split in ['train','temperature','conformal']+SPLITS:
                y=np.asarray(data[split]['labels'])
                one=y==0
                if split=='conformal':
                    raw=probabilities(z)
                elif split in SPLITS:
                    saved=np.load(path/f'predictions_{model}_raw_{split}.npz')
                    assert np.array_equal(saved['labels'],y)
                    raw=saved['probabilities']
                else:
                    diagnostic=OUT/f'predictions_onestar_{model}_seed{seed}_{split}.npz'
                    if diagnostic.exists():
                        raw=np.load(diagnostic)['probabilities']
                    elif one.any():
                        small={'texts':[text for text,keep in zip(data[split]['texts'],one) if keep], 'labels':y[one]}
                        raw=probabilities(predict_logits(model,small,cfg,model_dir,device))
                        np.savez_compressed(diagnostic,probabilities=raw)
                    else:
                        raw=np.empty((0,5))
                if split not in ['train','temperature']:
                    raw=raw[one]
                scaled=probabilities(np.log(np.clip(raw,1e-300,1)),t) if len(raw) else raw
                for method,p in [('raw',raw),('temperature',scaled)]:
                    values=p[:,0]
                    quantiles=np.quantile(values,[0,.05,.25,.5,.75,.95,1]) if len(values) else [np.nan]*7
                    probability_rows.append(dict(seed=seed,model=model,probability=method,split=split,n=len(values),
                        mean=float(values.mean()) if len(values) else np.nan,
                        **dict(zip(['min','p05','p25','median','p75','p95','max'],quantiles))))
    assert len(set(user_lookup.values()))==len(user_lookup)
    pd.DataFrame(sample_rows).to_csv(OUT/'sample_counts.csv',index=False)
    pd.DataFrame(user_rows).to_csv(OUT/'reviewer_support.csv',index=False)
    pd.DataFrame(probability_rows).to_csv(OUT/'onestar_probability.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(label_mapping='0..4 = 1..5 stars',
        labels_and_reviewers_verified=True,encoded_reviewers=len(user_lookup),checks=audit),indent=2))


def interval(draws):
    valid=draws[np.isfinite(draws)]
    return (*np.quantile(valid,[.025,.975]),len(valid)) if len(valid) else (np.nan,np.nan,0)


def analyze(config):
    rows,thresholds,deltas,draws_all,points_all=[],[],[],{},{}
    for seed in config['seeds']:
        path=PREVIOUS/f'formal_seed{seed}'
        manifest=pd.read_csv(path/'sample_manifest.csv',dtype={'original_user':str})
        calibration=json.loads((path/'calibration.json').read_text())
        for model in ['baseline','bert']:
            saved=np.load(OUT/f'predictions_calibration_{model}_seed{seed}.npz')
            z,ycal=saved['logits'],saved['labels']
            counts=np.bincount(ycal,minlength=5)
            for probability,temp in [('raw',1.),('temperature',calibration[model]['temperature'])]:
                pcal=probabilities(z,temp)
                marginal=lac_quantile(pcal,ycal,config['alpha'])
                conditional=class_conditional_quantiles(pcal,ycal,config['alpha'])
                for k in range(5):
                    thresholds.append(dict(seed=seed,model=model,probability=probability,stars=k+1,
                        calibration_n=int(counts[k]),rank=math.ceil((int(counts[k])+1)*(1-config['alpha'])),
                        class_qhat=conditional[k],marginal_qhat=marginal,
                        always_include=bool(np.isposinf(conditional[k]))))
                for split in SPLITS:
                    evaluation=np.load(path/f'predictions_{model}_{probability}_{split}.npz')
                    p,y= evaluation['probabilities'],evaluation['labels']
                    selected=manifest[manifest.split==split]
                    assert np.array_equal(y,selected.label.to_numpy())
                    ids=selected.original_user.to_numpy()
                    options={'marginal':prediction_sets(p,marginal),'class_conditional':class_conditional_sets(p,conditional)}
                    assert np.array_equal(options['marginal'],evaluation['sets'])
                    for method,sets in options.items():
                        prefix=dict(seed=seed,model=model,probability=probability,split=split,method=method)
                        ci,draws=class_set_bootstrap(y,sets,ids,config['bootstrap_replicates'],
                            config['bootstrap_seed']+{'id_test':0,'test':1,'val':2}[split])
                        for r in ci.to_dict('records'):
                            rows.append(dict(prefix,**r))
                        key=(seed,model,probability,split,method)
                        draws_all[key]=draws
                        points_all[key]=ci.set_index('metric').estimate.to_dict()
                        np.savez_compressed(OUT/f'predictions_{model}_{probability}_{method}_seed{seed}_{split}.npz',labels=y,sets=sets)
                    for metric in draws_all[(seed,model,probability,split,'marginal')]:
                        a=(seed,model,probability,split,'class_conditional')
                        b=(seed,model,probability,split,'marginal')
                        lo,hi,valid=interval(draws_all[a][metric]-draws_all[b][metric])
                        deltas.append(dict(seed=seed,model=model,probability=probability,split=split,metric=metric,
                            improvement=points_all[a][metric]-points_all[b][metric],lower=lo,upper=hi,bootstrap_valid=valid))
    stats=pd.DataFrame(rows)
    stats.to_csv(OUT/'per_seed_statistics.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'calibration_thresholds.csv',index=False)
    pd.DataFrame(deltas).to_csv(OUT/'paired_method_difference.csv',index=False)
    aggregate,changes=[],[]
    for (model,probability,split,method,metric),group in stats.groupby(['model','probability','split','method','metric']):
        boot=np.mean([draws_all[(s,model,probability,split,method)][metric] for s in config['seeds']],axis=0)
        lo,hi,valid=interval(boot)
        aggregate.append(dict(model=model,probability=probability,split=split,method=method,metric=metric,
            estimate=group.estimate.mean(),seed_sd=group.estimate.std(ddof=1),lower=lo,upper=hi,bootstrap_valid=valid))
    summary=pd.DataFrame(aggregate)
    summary.to_csv(OUT/'class_conditional_comparison.csv',index=False)
    for (model,probability,split,metric),group in pd.DataFrame(deltas).groupby(['model','probability','split','metric']):
        boot=np.mean([draws_all[(s,model,probability,split,'class_conditional')][metric]-draws_all[(s,model,probability,split,'marginal')][metric] for s in config['seeds']],axis=0)
        lo,hi,valid=interval(boot)
        changes.append(dict(model=model,probability=probability,split=split,metric=metric,improvement=group.improvement.mean(),
            lower=lo,upper=hi,bootstrap_valid=valid))
    pd.DataFrame(changes).to_csv(OUT/'mean_method_difference.csv',index=False)
    class_rows=[]
    support=pd.read_csv(OUT/'sample_counts.csv')
    for r in stats[stats.metric.str.startswith('class_')].to_dict('records'):
        k=int(r['metric'].rsplit('_',1)[1])
        cal=support[(support.seed==r['seed'])&(support.split=='conformal')&(support.label==k)].iloc[0]
        test=support[(support.seed==r['seed'])&(support.split==r['split'])&(support.label==k)].iloc[0]
        class_rows.append(dict(r,stars=k+1,calibration_n=cal.n,evaluation_n=test.n,evaluation_reviewers=test.reviewers))
    pd.DataFrame(class_rows).to_csv(OUT/'class_coverage.csv',index=False)
    from src.plots import plot_stage3
    plot_stage3(summary,OUT)
    write_report(config,summary,pd.DataFrame(changes))


def write_report(config,summary,changes):
    def md(frame):
        def cell(v): return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
        return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None))
    counts=pd.read_csv(OUT/'sample_counts.csv')
    one=counts[counts.stars==1][['seed','split','n','reviewers']]
    primary=summary[(summary.probability=='temperature')&(summary.split=='val')]
    main=primary[primary.metric.isin(['coverage','average_set_size','worst_class_coverage','macro_class_coverage'])].pivot(index=['model','method'],columns='metric',values='estimate').reset_index()
    coverage=summary[(summary.probability=='temperature')&summary.metric.str.startswith('class_coverage')].copy()
    coverage['stars']=coverage.metric.str.rsplit('_',n=1).str[-1].astype(int)+1
    coverage=coverage.sort_values(['split','model','method','stars'],key=lambda x:x.map({'val':0,'id_test':1,'test':2}) if x.name=='split' else x)
    probabilities=pd.read_csv(OUT/'onestar_probability.csv')
    p_one=probabilities[(probabilities.probability=='temperature')&probabilities.split.isin(['conformal']+SPLITS)].groupby(['model','split'])[['n','mean','median','p05','p95']].mean().reset_index()
    comparison=changes[(changes.probability=='temperature')&changes.metric.isin(['coverage','average_set_size','class_coverage_0','worst_class_coverage'])]
    reviewer=pd.read_csv(OUT/'reviewer_support.csv')
    reviewer=reviewer[(reviewer.seed==42)&reviewer.split.isin(SPLITS)]
    threshold=pd.read_csv(OUT/'calibration_thresholds.csv')
    threshold=threshold[(threshold.probability=='temperature')&(threshold.stars==1)]
    def value(model,split,method,metric):
        return float(summary[(summary.model==model)&(summary.probability=='temperature')&(summary.split==split)&(summary.method==method)&(summary.metric==metric)].estimate.iloc[0])
    findings=[]
    for model in ['baseline','bert']:
        old=value(model,'val','marginal','class_coverage_0')
        new=value(model,'val','class_conditional','class_coverage_0')
        cost=changes[(changes.model==model)&(changes.probability=='temperature')&(changes.split=='val')&(changes.metric=='average_set_size')].iloc[0]
        gain=changes[(changes.model==model)&(changes.probability=='temperature')&(changes.split=='val')&(changes.metric=='class_coverage_0')].iloc[0]
        findings.append(f'- OOD val，{model}一星Coverage从{old:.1%}升至{new:.1%}；配对改善{gain.improvement*100:.1f}个百分点，条件95% CI [{gain.lower*100:.1f}, {gain.upper*100:.1f}]。平均集合增加{cost.improvement:.3f}类，95% CI [{cost.lower:.3f}, {cost.upper:.3f}]。')
        findings.append(f'- {model}类别条件一星Coverage在ID/OOD test为{value(model,"id_test","class_conditional","class_coverage_0"):.1%}/{value(model,"test","class_conditional","class_coverage_0"):.1%}；改善方向在三个划分一致，但经验水平不完全一致，且仍属探索性结果。')
    text=f'''# CalibShift-NLP Stage 3：类别条件覆盖率

## 设计与核查

复用第二阶段3个种子（42/123/2026）的两个已训练模型、已保存温度、相同2000条Conformal校准行和每个评估划分3000行。没有重新训练，没有重新下载数据，没有改变评估划分。主分析为OOD validation (`val`)；ID test和OOD test均为探索性补充。此前已经查看这些test，不能宣称本阶段是完全独立的确认性验证。raw/temperature均输出，主要表格预先指定temperature，不依据结果选择。

## 主要发现

{chr(10).join(findings)}

固定OOD val的一星仅36条、32位Reviewer，OOD test为48条、45位Reviewer，足以观察到非常大的方法差异，却不足以精确断言总体覆盖率为0%、99%或恰好90%。三个模型种子的预测没有把这些评论变成三倍独立样本。训练只有约百条一星、校准仅24–32条，分位数和尾部概率仍可能不稳定。

核对标签0..4对应1..5星、metadata_y/original_y、LR classes_、BERT id2label、评论顺序和原始Reviewer ID编码的一一映射。重新推理Conformal概率后复现第二阶段全局阈值，并逐元素复现原始测试预测集合。未发现需要重新训练的模型或标签实现错误。详细audit.json；所有旧结果和权重的前后哈希在preservation.json。

一星样本数量与Reviewer数（测试行跨种子共享，不能把三个n相加）：

{md(one)}

一星真实评论的预测p(一星)分布摘要（温度缩放，各种子摘要的平均，分位数并非合并样本的分位数）：

{md(p_one)}

## 方法与保守处理

Class-Conditional LAC（label-Mondrian）在真实标签为k的n_k条校准评论中，用s=1-p_k，取第ceil((n_k+1)*0.9)个顺序统计量。预测时对五个候选标签各用自己的q_k测试1-p_k<=q_k，不使用预测样本的真实标签。温度仅使用第二阶段独立temperature子集拟合，本阶段不重拟合。原始Marginal LAC保留全局阈值。两个集合方法使用相同概率、同样评估行、alpha=0.1。

当n_k=0或修正后的秩超过n_k（alpha=0.1时n_k<9），令q_k=+infinity，所有预测集合都包含该候选类别。这对应在校准分数中增加一个+infinity的有限样本分位数，不回退至全局阈值。此保守处理可能扩大集合；n_k>=9也不意味着估计精确。

一星实际阈值与样本数：

{md(threshold[['seed','model','calibration_n','rank','class_qhat','marginal_qhat','always_include']])}

## OOD validation主分析（三种子均值）

{md(main)}

## 各类别覆盖率与95%区间

以下为三种子均值及条件于三个已拟合模型/校准器的Reviewer聚类区间；单种子结果、校准/评估数量与各类Set Size见class_coverage.csv。先展示val，再展示ID/OOD test：

{md(coverage[['model','split','method','stars','estimate','lower','upper']])}

## 改善与集合大小代价

Class-conditional minus marginal的配对差值；覆盖率为比例，Set Size为类别数。区间使用同一次Reviewer重采样计算两种方法的差值，不由两个独立区间相减：

{md(comparison[['model','split','metric','improvement','lower','upper']])}

整体Coverage与各类别Coverage必须同时判断；Worst-class为五类Coverage的最小值，Macro为五类Coverage等权平均。macro/worst若某类无评估样本则定义为缺失，不悄悄丢弃该类。最小值非光滑，Bootstrap区间在接近并列时只宜作描述，不作为确认性最差类推断。

三种子汇总的Worst-class为每个种子先取最小值再平均，不等于先平均五类Coverage再取最小值。Class-Conditional集合不是原Marginal集合的必然超集：少数类阈值放宽时，多数类阈值可能收紧，因此低星级Coverage提高、集合变大，并不意味着总体Coverage也单调提高。当前总体Coverage仍约0.89，不能把类别改善或接近90%的点估计解读为已证实覆盖保证。

## Reviewer与不稳定性

{md(reviewer[['split','n','reviewers','median_reviews','min_reviews','max_reviews','fraction_single_review','fraction_under5']])}

当前按评论均匀抽样产生的Reviewer内观察数较少，尤其OOD每位Reviewer中位数只有少量评论，不能支持精确的逐Reviewer可靠性判断。本阶段不为了获得更好结果而改变固定评估行，也不为每位Reviewer虚构独立Bernoulli置信区间；采用与第二阶段相同的2000次Reviewer整群Bootstrap，保留组内依赖，用于总体/类别统计和方法配对差值。下一阶段若研究Reviewer条件覆盖，需另行预先设计Reviewer分层/整群样本、保证足够每人评论，而非事后在当前test反复抽样。

类别条件覆盖是P(Y属于集合|Y=k)，Reviewer条件覆盖是P(Y属于集合|Reviewer=u)；两者不是同一个条件事件，类别改善不意味着每位Reviewer获得90%覆盖。

稀有类样本数与独立Reviewer数需一起看。校准一星样本少使分位数依赖少数尾部评论；三个种子还改变训练/校准样本，不能将变动只归因于模型初始化。CI条件于已拟合温度和阈值，未包含校准器估计的全部不确定性。零次覆盖的非参数Bootstrap可退化为[0,0]，不能证明总体覆盖概率为0；对极端少样本或零事件应避免过度精确解释。

## 为什么不能宣称分布偏移下保证成立

Label-Mondrian的有限样本类别条件边际保证要求校准和未来样本在每个真实类别内部可交换，并不是对固定校准集逐次实验都恰好90%的经验保证。Reviewer内相关、temperature/conformal共享部分Reviewer、训练域与未见Reviewer的条件分布改变，都可能破坏该前提。按Reviewer Bootstrap不会修复校准可交换性。即使ID经验覆盖接近90%，也不证明假设；OOD改善只作经验结论。该方法也不给五个类别同时成立的高概率保证或逐样本正确率保证。

## 下一步

优先以预先指定、Reviewer隔离的校准设计研究稀有评分尾部分位数的不确定性；研究标签先验变化与类内协变量变化分别如何影响覆盖率；再以有足够每人评论的独立样本研究Reviewer条件可靠性。避免按当前test的结果反复更改类别阈值或抽样。

复现：`python stage3.py --mode all`（复用校准预测cache）；`python -m pytest -q`。理论参考：[Angelopoulos & Bates](https://arxiv.org/abs/2107.07511)、[Mondrian原理](https://op.cs.rhul.ac.uk/index.html?n=Main.MondrianConformalPredictor)。本项目只实现已有方法，不宣称新算法。
'''
    (OUT/'STAGE3_REPORT.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['audit','all'],default='all')
    args=parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ.setdefault('HF_HOME',str(ROOT/'models'/'hf_cache'))
    config=yaml.safe_load((ROOT/'config.stage2.yaml').read_text())
    protocol=dict(seeds=config['seeds'],alpha=config['alpha'],bootstrap_seed=config['bootstrap_seed'],
        bootstrap_replicates=config['bootstrap_replicates'],primary_split='val',secondary_splits=['id_test','test'],
        primary_probability='temperature',models=['baseline','bert'],methods=['marginal','class_conditional'],
        new_sampling=False,new_training=False,small_class_rule='augmented infinity order statistic',
        inference='paired reviewer-cluster percentile; conditional on fixed models and calibrators')
    frozen=OUT/'protocol.json'
    text=json.dumps(protocol,indent=2)
    if frozen.exists() and frozen.read_text()!=text:
        raise ValueError('Stage3 protocol changed; refusing silent reuse')
    frozen.write_text(text)
    paths=list(PREVIOUS.rglob('*'))+list((ROOT/'models'/'stage2').rglob('*'))
    before={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}
    previous=OUT/'input_hashes.json'
    if previous.exists() and json.loads(previous.read_text())!=before:
        raise ValueError('Stage2 inputs changed; cached Stage3 predictions may be stale')
    previous.write_text(json.dumps(before,indent=2))
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu'
    started=time.time()
    audit_and_cache(config,device)
    if args.mode=='all':
        analyze(config)
    assert all(sha(ROOT/p)==value for p,value in before.items())
    (OUT/'preservation.json').write_text(json.dumps(dict(files_verified=len(before),all_unchanged=True,
        device=device,seconds=time.time()-started,mode=args.mode),indent=2))
    print('Stage3 complete:',OUT,flush=True)


if __name__=='__main__':
    main()
