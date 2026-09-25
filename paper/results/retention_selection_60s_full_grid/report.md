# Common-Source Retention--Selection Budget Sweep

## Protocol

Every policy supplies its real reconstructed bank and selected indices. Candidate image features for every policy come from the same `baseline` rollout. Target features come from exact-index dataset ground truth. One of every `19` context slots is sampled.

This isolates candidate-set retention and selection from policy-specific generated histories. The DINO hindsight-best candidate is a diagnostic proxy, not a deployable oracle.

Matched manifest rows used by every plotted policy: `3,8,13,18,23,28,33,38,43,48,53,58,63,68,73` (`n=15`).
Rows excluded because at least one requested trace/cache was missing: `none`.

## Whole Rollout

| policy | budget | trajectories | candidates | retention gap | 95% CI | selection gap | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Unbounded | -- | 15 | 909.0 | 0.0000 | [0.0000, 0.0000] | 0.1996 | [0.1583, 0.2410] |
| FIFO | 16 | 15 | 12.0 | 0.1771 | [0.1372, 0.2181] | 0.0365 | [0.0323, 0.0410] |
| FIFO | 32 | 15 | 28.0 | 0.1562 | [0.1171, 0.1971] | 0.0546 | [0.0481, 0.0618] |
| FIFO | 64 | 15 | 60.0 | 0.1355 | [0.0957, 0.1775] | 0.0749 | [0.0649, 0.0863] |
| FIFO | 128 | 15 | 121.8 | 0.1160 | [0.0782, 0.1572] | 0.0928 | [0.0796, 0.1071] |
| RI | 16 | 15 | 14.9 | 0.0487 | [0.0367, 0.0618] | 0.1518 | [0.1090, 0.1973] |
| RI | 32 | 15 | 30.9 | 0.0401 | [0.0314, 0.0487] | 0.1721 | [0.1297, 0.2146] |
| RI | 64 | 15 | 62.7 | 0.0286 | [0.0203, 0.0377] | 0.1735 | [0.1333, 0.2141] |
| RI | 128 | 15 | 124.2 | 0.0185 | [0.0132, 0.0241] | 0.1844 | [0.1475, 0.2241] |
| K-center | 16 | 15 | 15.0 | 0.0429 | [0.0312, 0.0549] | 0.1539 | [0.1175, 0.1941] |
| K-center | 32 | 15 | 31.0 | 0.0361 | [0.0271, 0.0458] | 0.1657 | [0.1272, 0.2071] |
| K-center | 64 | 15 | 62.9 | 0.0219 | [0.0157, 0.0287] | 0.1776 | [0.1387, 0.2177] |
| K-center | 128 | 15 | 124.5 | 0.0123 | [0.0078, 0.0178] | 0.1864 | [0.1472, 0.2271] |
| GeoCov | 16 | 15 | 14.9 | 0.0513 | [0.0380, 0.0654] | 0.1269 | [0.0971, 0.1563] |
| GeoCov | 32 | 15 | 30.8 | 0.0339 | [0.0244, 0.0442] | 0.1749 | [0.1319, 0.2176] |
| GeoCov | 64 | 15 | 62.5 | 0.0219 | [0.0154, 0.0288] | 0.1962 | [0.1603, 0.2323] |
| GeoCov | 128 | 15 | 123.7 | 0.0123 | [0.0081, 0.0169] | 0.1945 | [0.1534, 0.2382] |
| mce b16 lambda1 pilot | 16 | 15 | 14.8 | 0.0507 | [0.0364, 0.0669] | 0.1523 | [0.1147, 0.1952] |
| mce b32 lambda1 pilot | 32 | 15 | 30.5 | 0.0376 | [0.0280, 0.0478] | 0.1585 | [0.1208, 0.1998] |
| mce b64 lambda1 pilot | 64 | 15 | 62.2 | 0.0272 | [0.0191, 0.0353] | 0.1716 | [0.1313, 0.2145] |
| mce b128 lambda1 pilot | 128 | 15 | 123.6 | 0.0244 | [0.0135, 0.0402] | 0.1745 | [0.1351, 0.2157] |

## Late Rollout (section >= 18)

| policy | budget | trajectories | candidates | retention gap | 95% CI | selection gap | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Unbounded | -- | 15 | 1555.0 | 0.0000 | [0.0000, 0.0000] | 0.2483 | [0.1974, 0.2985] |
| FIFO | 16 | 15 | 12.0 | 0.2164 | [0.1631, 0.2706] | 0.0396 | [0.0330, 0.0467] |
| FIFO | 32 | 15 | 28.0 | 0.1942 | [0.1444, 0.2445] | 0.0609 | [0.0471, 0.0774] |
| FIFO | 64 | 15 | 60.0 | 0.1715 | [0.1230, 0.2209] | 0.0840 | [0.0638, 0.1107] |
| FIFO | 128 | 15 | 124.0 | 0.1483 | [0.1032, 0.1955] | 0.1053 | [0.0828, 0.1335] |
| RI | 16 | 15 | 15.0 | 0.0492 | [0.0326, 0.0675] | 0.1906 | [0.1343, 0.2495] |
| RI | 32 | 15 | 31.0 | 0.0416 | [0.0261, 0.0572] | 0.2260 | [0.1737, 0.2805] |
| RI | 64 | 15 | 62.9 | 0.0317 | [0.0189, 0.0469] | 0.2184 | [0.1588, 0.2786] |
| RI | 128 | 15 | 126.9 | 0.0225 | [0.0153, 0.0303] | 0.2196 | [0.1706, 0.2688] |
| K-center | 16 | 15 | 14.9 | 0.0566 | [0.0386, 0.0748] | 0.1841 | [0.1381, 0.2317] |
| K-center | 32 | 15 | 31.0 | 0.0523 | [0.0386, 0.0665] | 0.1988 | [0.1481, 0.2527] |
| K-center | 64 | 15 | 63.0 | 0.0323 | [0.0227, 0.0422] | 0.2171 | [0.1691, 0.2643] |
| K-center | 128 | 15 | 127.0 | 0.0168 | [0.0099, 0.0254] | 0.2292 | [0.1789, 0.2789] |
| GeoCov | 16 | 15 | 15.0 | 0.0584 | [0.0389, 0.0791] | 0.1498 | [0.0979, 0.2069] |
| GeoCov | 32 | 15 | 31.0 | 0.0408 | [0.0272, 0.0560] | 0.2044 | [0.1406, 0.2738] |
| GeoCov | 64 | 15 | 62.9 | 0.0277 | [0.0186, 0.0372] | 0.2275 | [0.1790, 0.2817] |
| GeoCov | 128 | 15 | 126.9 | 0.0157 | [0.0090, 0.0234] | 0.2366 | [0.1834, 0.2939] |
| mce b16 lambda1 pilot | 16 | 15 | 14.9 | 0.0598 | [0.0427, 0.0759] | 0.1961 | [0.1442, 0.2486] |
| mce b32 lambda1 pilot | 32 | 15 | 30.5 | 0.0485 | [0.0345, 0.0623] | 0.1909 | [0.1440, 0.2382] |
| mce b64 lambda1 pilot | 64 | 15 | 62.4 | 0.0320 | [0.0210, 0.0433] | 0.2151 | [0.1644, 0.2656] |
| mce b128 lambda1 pilot | 128 | 15 | 126.2 | 0.0379 | [0.0205, 0.0612] | 0.2067 | [0.1586, 0.2539] |

## Increasing-Budget Steps

Negative changes are improvements. `moves_left` means the larger budget reduced retention loss; `moves_down` means it also reduced selection loss.

| family | budget step | retention change | selection change | moves left | moves down |
| --- | --- | ---: | ---: | --- | --- |
| FIFO | B16 -> B32 | -0.0209 | +0.0181 | True | False |
| FIFO | B32 -> B64 | -0.0207 | +0.0203 | True | False |
| FIFO | B64 -> B128 | -0.0195 | +0.0179 | True | False |
| GeoCov | B16 -> B32 | -0.0174 | +0.0480 | True | False |
| GeoCov | B32 -> B64 | -0.0120 | +0.0213 | True | False |
| GeoCov | B64 -> B128 | -0.0096 | -0.0017 | True | True |
| K-center | B16 -> B32 | -0.0068 | +0.0118 | True | False |
| K-center | B32 -> B64 | -0.0142 | +0.0119 | True | False |
| K-center | B64 -> B128 | -0.0095 | +0.0087 | True | False |
| RI | B16 -> B32 | -0.0086 | +0.0203 | True | False |
| RI | B32 -> B64 | -0.0116 | +0.0014 | True | False |
| RI | B64 -> B128 | -0.0101 | +0.0109 | True | False |

## Files

- `figures/retention_selection_budget_all.png`
- `figures/retention_selection_budget_late.png`
- `tables/run_summary_all.csv`
- `tables/run_summary_late.csv`
- `tables/trajectory_summary_all.csv`
- `tables/query_decomposition_common_source.csv`
- `tables/increasing_budget_steps.csv`
