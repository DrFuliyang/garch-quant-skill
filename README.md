# GARCH QUANT Framework v6.0 Skill

> Rolling EGARCH(1,1) + LSTM Walk-Forward + Kelly Sizing — 完整可重用 Class

## 架構

```
GARCHQuantSkill
├── _get_data()              Twelve Data API（脫敏後需填入個人 API Key）
├── _adf_test()              ADF 平穩檢定
├── _detect_regime()         Markov 政權 (0=牛市, 1=熊市, 2=震盪)
├── _add_rolling_garch()     EGARCH(1,1) 滾動波動率 + ffill().bfill()
├── _create_features()        ret/vol/mom + volume_z + regime_dummies
├── _train_lstm()            RegimeLSTM (2-layer, hidden=64, dropout=0.2)
├── _walk_forward_retraining() Walk-Forward + KS 分布監控 + Telegram 警報
├── _run_backtest()          equity curve, Sharpe, max drawdown
├── predict_next()           即時下一期預測 + Kelly 倉位
└── get_current_garch()      回傳目前 EGARCH(1,1) 波動率
```

## 關鍵修復記錄（踩坑日誌）

| 問題 | 修復 |
|------|------|
| `garch_vol` 前 180 筆 NaN → `dropna` 全刪 | `ffill().bfill()` |
| `garch_vol_z` rolling std=0 → 除零 NaN | `.replace(0, 1)` |
| `volume_z` std=0 → 全部 NaN | Twelve Data 日線無 volume，設 Volume=1.0 |
| `feature_cols` 包含 `garch_vol_z` → 觸發 dropna | 排除 `garch_vol_z` |
| `torch.tensor` 不接受 `numpy.object_` | `.astype(np.float32)` |
| `regime_dummies` 產生 object 型別 | 全程確保 float32 |
| Twelve Data API rate limit | 備援 yfinance，失敗時列印提示 |

## 安裝依賴

```bash
pip install pandas numpy yfinance statsmodels arch torch scipy requests
```

## 使用方式

```python
from garch_quant_skill import GARCHQuantSkill

skill = GARCHQuantSkill(
    ticker="BTC-USD",       # 可換 ETH-USD, SOL-USD
    retrain_every=30,      # 每 N 天重新訓練
    seq_length=20,         # LSTM 序列長度
    fast_mode=True,        # True=驗證（30 epochs），False=生產（60 epochs）
    telegram_token=None,   # 你的 Bot Token（可選）
    telegram_chat_id=None # 你的 Chat ID（可選）
)

skill.run()             # 完整流程
skill.predict_next()     # 即時預測
skill.get_current_garch()  # 目前波動率
```

## 參數說明

- **fast_mode=True**: 30 epochs + 90天 retrain，適合快速驗證
- **fast_mode=False**: 60 epochs + 30天 retrain，適合正式生產
- **telegram_token/chat_id**: 設好後 KS 偏移警報和 re-train 通知會自動推播

## 回測結果（BTC-USD, fast_mode=True）

| 指標 | 數值 |
|------|------|
| 保留資料 | 4224 筆 |
| 總報酬 | +74.71% |
| 年化報酬 | +4.48% |
| Sharpe Ratio | 0.554 |
| 最大回撤 | -18.33% |
| 方向準確率 | 67.7% |
| 平均 Kelly 倉位 | ~7.6% |

## 資料來源

`garch_quant_skill.py` 內的 API Key 為範例值，請更換為個人 [Twelve Data](https://twelvedata.com) API Key（免費額度足夠）。

## License

MIT
