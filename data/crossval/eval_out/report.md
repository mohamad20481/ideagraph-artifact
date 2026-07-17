# Evaluation report

- entries: 100 (generated=50, published=50)
- judges: ['kimi/moonshot-v1-32k', 'kimi/kimi-k2.5', 'kimi/kimi-k2.6']
- non-inferiority margin delta=0.3, alpha=0.05

## Aggregate scores (unit scale 0-1)

| group | Nov-A | Feas-A | NFT |
|---|---|---|---|
| generated | 0.568 | 0.597 | 0.573 |
| published | 0.548 | 0.732 | 0.616 |

## Non-inferiority (generated vs published, delta=0.3)

| dimension | gen mean | pub mean | diff | p (one-sided) | NI? |
|---|---|---|---|---|---|
| novelty | 3.27 | 3.19 | +0.08 | 0.0002 | YES |
| feasibility | 3.39 | 3.93 | -0.54 | 0.9822 | no |
| clarity | 3.81 | 3.56 | +0.25 | 0.0003 | YES |
| significance | 3.54 | 3.22 | +0.32 | 0.0000 | YES |
| excitement | 3.32 | 3.13 | +0.19 | 0.0000 | YES |
| overall | 3.42 | 3.23 | +0.19 | 0.0000 | YES |

## Div-Pair (generated pool): 0.700
