# Matched Set Size and Coverage-Efficiency Trade-off Analysis

## Scope and fixed protocol

This is one supplementary statistical contrast, with no model retraining, new evaluation sampling, README/PDF changes, commit or push. It reuses saved temperature-scaled probabilities for LR/DistilBERT, seeds 42/123/2026, existing conformal data and true reviewer IDs. Methods are A: marginal LAC alpha=.10; B: class-conditional LAC alpha=.10; C: matched-size marginal LAC with alpha chosen from validation set sizes only. All six choices were written to matching_choices.csv before any evaluation labels or test arrays were accessed by this script.

All attainable marginal thresholds are the distinct true-label calibration scores plus infinity. At each threshold, validation inclusion counts are computed from probabilities alone. Minimize the absolute integer count discrepancy from method B; ties select the smaller inclusion count, then the smaller threshold/first attainable rank. A representative alpha is the midpoint of the rank's interval, ensuring the original finite-sample quantile reproduces that threshold. This exhaustive discrete search does not fabricate an interpolated threshold or exact match.

Validation labels, class coverage and final test results are not inputs to matching. Calibration labels are used only for the required conformal scores. Descriptive validation/test coverage curves are generated after all selections are locked and do not feed back into alpha selection. Curves average fitted-seed statistics at the same finite-sample rank/alpha; they are not an ensemble classifier. C has a different alpha and nominal target from B; validation selection and OOD shift prevent an automatic conformal guarantee.

## Matching on OOD validation

| seed | model | alpha | rank | candidate_count | target_size | matched_size | signed_gap | exact_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | baseline | 0.0442 | 1913 | 2001 | 2.3673 | 2.3620 | -0.0053 | False |
| 42 | bert | 0.0387 | 1924 | 2001 | 2.2147 | 2.2143 | -0.0003 | False |
| 123 | baseline | 0.0417 | 1918 | 2001 | 2.2673 | 2.2660 | -0.0013 | False |
| 123 | bert | 0.0337 | 1934 | 2001 | 2.3213 | 2.3193 | -0.0020 | False |
| 2026 | baseline | 0.0492 | 1903 | 2001 | 2.2293 | 2.2257 | -0.0037 | False |
| 2026 | bert | 0.0212 | 1959 | 2001 | 2.4940 | 2.4907 | -0.0033 | False |

The validation gap is matched marginal minus class-conditional size. These choices remain fixed on test, where a new gap may appear. Mean test sizes and their paired difference are reported below. Very close means do not imply identical per-review sizes or label allocations.

## True sample sizes

| split | stars | n | reviewers |
| --- | --- | --- | --- |
| val | 1 | 36 | 32 |
| val | 2 | 93 | 85 |
| val | 3 | 280 | 241 |
| val | 4 | 804 | 557 |
| val | 5 | 1787 | 929 |
| test | 1 | 48 | 45 |
| test | 2 | 95 | 84 |
| test | 3 | 301 | 253 |
| test | 4 | 848 | 588 |
| test | 5 | 1708 | 895 |

The fixed validation/test samples contain 3000 comments each. Seeds reuse exactly the same comments; they do not yield 9000 independent observations. The one-star validation/test supports above are not multiplied by three. Per-seed class calibration counts are in matched_size_metrics.csv.

## Real results: three fitted-seed means

| model | split | method | coverage | average_set_size | worst_class_coverage | macro_class_coverage | class_coverage_0 | class_coverage_1 | class_coverage_2 | class_coverage_3 | class_coverage_4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | val | marginal | 0.8999 | 1.8807 | 0.0000 | 0.5212 | 0.0000 | 0.0824 | 0.6071 | 0.9179 | 0.9983 |
| baseline | val | class_conditional | 0.8939 | 2.2880 | 0.8420 | 0.8929 | 0.8981 | 0.8781 | 0.8988 | 0.8972 | 0.8924 |
| baseline | val | matched_marginal | 0.9520 | 2.2846 | 0.0093 | 0.6530 | 0.0093 | 0.4158 | 0.8571 | 0.9830 | 0.9998 |
| baseline | test | marginal | 0.8901 | 1.9066 | 0.0000 | 0.5250 | 0.0000 | 0.0877 | 0.6357 | 0.9037 | 0.9979 |
| baseline | test | class_conditional | 0.8901 | 2.3529 | 0.8726 | 0.9068 | 0.9444 | 0.8842 | 0.9369 | 0.8860 | 0.8827 |
| baseline | test | matched_marginal | 0.9506 | 2.3388 | 0.0000 | 0.6628 | 0.0000 | 0.4316 | 0.9025 | 0.9800 | 1.0000 |
| bert | val | marginal | 0.8996 | 1.7839 | 0.4143 | 0.7169 | 0.4167 | 0.6344 | 0.6964 | 0.8677 | 0.9692 |
| bert | val | class_conditional | 0.8914 | 2.3433 | 0.8731 | 0.9164 | 0.9815 | 0.9140 | 0.8952 | 0.9129 | 0.8782 |
| bert | val | matched_marginal | 0.9661 | 2.3414 | 0.6513 | 0.8578 | 0.6574 | 0.8029 | 0.8524 | 0.9867 | 0.9894 |
| bert | test | marginal | 0.8878 | 1.8014 | 0.5001 | 0.7239 | 0.5069 | 0.6000 | 0.7010 | 0.8404 | 0.9709 |
| bert | test | class_conditional | 0.8853 | 2.3680 | 0.8743 | 0.9142 | 0.9931 | 0.9123 | 0.8970 | 0.8943 | 0.8743 |
| bert | test | matched_marginal | 0.9636 | 2.3644 | 0.6904 | 0.8628 | 0.7083 | 0.7719 | 0.8638 | 0.9784 | 0.9916 |

Coverage columns are proportions; class_coverage_0..4 correspond to ratings 1..5. Worst-class is computed per fitted seed then averaged, not the minimum of already averaged class coverages. Macro-class coverage gives the five ratings equal weight. Per-seed estimates and all conditional intervals appear in matched_size_metrics.csv; seed-averaged intervals in matched_size_summary.csv.

## Primary matched contrast on OOD test

- baseline: matched marginal one-star coverage=0.0000; class-conditional=0.9444. Difference=0.9444, 95% CI [0.8889, 0.9858]; CI entirely positive.
- baseline: worst-class coverage changes from 0.0000 (matched marginal) to 0.8726 (class-conditional); paired difference 0.8726, CI [0.8193, 0.8815].
- baseline: the minority gain costs overall coverage: matched marginal=0.9506, class-conditional=0.8901, difference=-0.0604, CI [-0.0724, -0.0482].
- baseline, rating 4: class-conditional minus matched marginal coverage=-0.0939, CI [-0.1098, -0.0778].
- baseline, rating 5: class-conditional minus matched marginal coverage=-0.1173, CI [-0.1325, -0.1032].
- baseline: actual mean test size is 2.338778 versus 2.352889; signed matched-minus-class-conditional gap=-0.014111.
- bert: matched marginal one-star coverage=0.7083; class-conditional=0.9931. Difference=0.2847, 95% CI [0.1933, 0.3778]; CI entirely positive.
- bert: worst-class coverage changes from 0.6904 (matched marginal) to 0.8743 (class-conditional); paired difference 0.1839, CI [0.1156, 0.2649].
- bert: the minority gain costs overall coverage: matched marginal=0.9636, class-conditional=0.8853, difference=-0.0782, CI [-0.0883, -0.0690].
- bert, rating 4: class-conditional minus matched marginal coverage=-0.0841, CI [-0.1010, -0.0684].
- bert, rating 5: class-conditional minus matched marginal coverage=-0.1173, CI [-0.1310, -0.1038].
- bert: actual mean test size is 2.364444 versus 2.368000; signed matched-minus-class-conditional gap=-0.003556.

Thus a residual minority/worst-class advantage remains in this nearly size-matched comparison, but class-conditional LAC does not dominate matched marginal LAC on overall or frequent-rating coverage. The conclusion is a trade-off in label allocation, not general efficiency superiority. All six validation gaps are nonzero; exact matching was impossible on their attainable threshold grids. The small test gaps are observed outcomes, not forced matches or formal equivalence tests.

Class-conditional minus matched marginal, using paired reviewer-cluster resampling:

| model | metric | estimate | lower | upper | bootstrap_valid |
| --- | --- | --- | --- | --- | --- |
| baseline | coverage | -0.0604 | -0.0724 | -0.0482 | 2000 |
| baseline | average_set_size | 0.0141 | -0.0142 | 0.0417 | 2000 |
| baseline | worst_class_coverage | 0.8726 | 0.8193 | 0.8815 | 2000 |
| baseline | macro_class_coverage | 0.2440 | 0.2224 | 0.2649 | 2000 |
| baseline | class_coverage_0 | 0.9444 | 0.8889 | 0.9858 | 2000 |
| baseline | class_coverage_1 | 0.4526 | 0.3640 | 0.5426 | 2000 |
| baseline | class_coverage_2 | 0.0343 | 0.0205 | 0.0493 | 2000 |
| baseline | class_coverage_3 | -0.0939 | -0.1098 | -0.0778 | 2000 |
| baseline | class_coverage_4 | -0.1173 | -0.1325 | -0.1032 | 2000 |
| bert | coverage | -0.0782 | -0.0883 | -0.0690 | 2000 |
| bert | average_set_size | 0.0036 | -0.0217 | 0.0268 | 2000 |
| bert | worst_class_coverage | 0.1839 | 0.1156 | 0.2649 | 2000 |
| bert | macro_class_coverage | 0.0514 | 0.0299 | 0.0725 | 2000 |
| bert | class_coverage_0 | 0.2847 | 0.1933 | 0.3778 | 2000 |
| bert | class_coverage_1 | 0.1404 | 0.0957 | 0.1897 | 2000 |
| bert | class_coverage_2 | 0.0332 | 0.0221 | 0.0455 | 2000 |
| bert | class_coverage_3 | -0.0841 | -0.1010 | -0.0684 | 2000 |
| bert | class_coverage_4 | -0.1173 | -0.1310 | -0.1038 | 2000 |

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
