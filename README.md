# GARCH Quant modelling prototype

[Research hub](https://github.com/DrFuliyang/research) · [Replication guide](https://github.com/DrFuliyang/research/blob/main/REPRODUCIBILITY.md)

A Python prototype combining rolling EGARCH volatility estimates, regime features, LSTM retraining, and portfolio-sizing calculations.

## Contents

[`garch_quant_skill.py`](garch_quant_skill.py) defines `GARCHQuantSkill`. Its methods cover data acquisition, stationarity diagnostics, regime features, rolling volatility estimation, neural-network training, backtesting, and next-period calculations.

## Interface

```python
from garch_quant_skill import GARCHQuantSkill

model = GARCHQuantSkill(
    ticker="BTC-USD",
    retrain_every=30,
    seq_length=20,
    fast_mode=True,
    telegram_token=None,
    telegram_chat_id=None,
)
```

The script provides `run()`, `predict_next()`, and `get_current_garch()`. Data access must be configured before running the pipeline. Optional notification credentials belong in the local runtime, not in committed files.

## Dependencies

The source imports pandas, NumPy, statsmodels, arch, PyTorch, SciPy, and requests. Review the data acquisition method for its provider requirements.

## Evaluation work

Before using this prototype for a paper or a performance claim, freeze the sample and environment, audit temporal alignment, and reproduce the outputs. The current volatility preprocessing applies backward filling to the initial missing values; this step needs attention in a chronological evaluation because it can propagate later estimates into earlier observations.

The September 2026 maintenance pass reviewed source structure and documentation. It did not execute the empirical pipeline. Earlier README versions remain available in Git history.

## Citation and reuse

Use the [research hub](https://github.com/DrFuliyang/research) to find paper-specific releases. This prototype has no paper-specific replication designation in the research directory.

## License

MIT, as stated in the existing project README.
