# EDA summary

- Train rows (raw): 82,332 | Test rows (raw): 175,341
- Exact duplicate rows in train: 28,380 (removed in `clean`)
- +-inf values in numeric train columns: 0 (-> NaN -> median-imputed)
- Attack share, train: 36.6% | test: 68.1%
  (attack-heavy on purpose; real networks are overwhelmingly benign, so
  precision here is optimistic vs. production prevalence)

## Top-10 features by |Spearman| with label
|                  |   abs_corr |
|:-----------------|-----------:|
| sttl             |      0.432 |
| ct_state_ttl     |      0.388 |
| dload            |      0.277 |
| ct_srv_dst       |      0.253 |
| ct_dst_sport_ltm |      0.249 |
| ct_dst_src_ltm   |      0.228 |
| ct_dst_ltm       |      0.225 |
| ct_srv_src       |      0.194 |
| dmean            |      0.191 |
| ct_src_ltm       |      0.182 |

## Top-10 train-vs-test shift (KS statistic)
|        |   ks_stat |
|:-------|----------:|
| ackdat |     0.304 |
| dbytes |     0.3   |
| dmean  |     0.3   |
| dpkts  |     0.299 |
| dinpkt |     0.299 |
| dload  |     0.299 |
| dttl   |     0.299 |
| sinpkt |     0.299 |
| swin   |     0.298 |
| sloss  |     0.297 |

Large KS values show the official split is already distribution-shifted,
a natural motivation for the drift monitoring built on Day 3.