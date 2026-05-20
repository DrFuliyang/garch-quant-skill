"""
GARCH QUANT Framework v6.0
Rolling EGARCH(1,1) + LSTM Walk-Forward + Kelly Sizing
脫敏版本 — 請將 TWELVE_DATA_KEY 替換為個人 API Key
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import ks_2samp
import requests
import warnings
warnings.filterwarnings('ignore')

# ==================== 脫敏區（請替換為個人 Key）====================
TWELVE_DATA_KEY = "YOUR_TWELVE_DATA_API_KEY"   # ← 替換為你的 Twelve Data API Key
# ==============================================================


class GARCHQuantSkill:
    """GARCH QUANT Framework v6.0 ── 完整可重用 Class"""

    def __init__(self, ticker="BTC-USD", retrain_every=30, seq_length=20,
                 telegram_token=None, telegram_chat_id=None, fast_mode=True):
        self.ticker = ticker
        self.retrain_every = retrain_every
        self.seq_length = seq_length
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.fast_mode = fast_mode
        self.model = None
        self.df_features = None
        self.feature_cols = None
        print(f"🚀 GARCH QUANT v6.0 | {ticker} | Fast Mode: {fast_mode}")

    def send_telegram(self, message):
        if not self.telegram_token or not self.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message, "parse_mode": "HTML"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"[Telegram] 發送失敗: {e}")

    # ====================== 內部方法 ======================
    def _get_data(self):
        """Twelve Data 日線，脫敏版本需填入個人 API Key"""
        key = TWELVE_DATA_KEY
        symbol = self.ticker.replace("-", "/")  # BTC-USD → BTC/USD
        url = (f"https://api.twelvedata.com/time_series"
               f"?symbol={symbol}&interval=1day&outputsize=5000"
               f"&format=JSON&apikey={key}")
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            vals = r.json()['values']
            df = pd.DataFrame(vals)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            df['Close'] = df['close'].astype(float)
            # Twelve Data 日線無 volume，設常數避免 std=0 → NaN
            df['Volume'] = 1.0
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            df = df.dropna()
            print(f"[GARCH] Twelve Data ✓ {self.ticker}: {len(df)} 筆 "
                  f"({df.index[0].date()} → {df.index[-1].date()})")
            return df
        except Exception as e:
            print(f"[GARCH] 資料取得失敗: {e}，請確認 TWELVE_DATA_KEY 已填入")
            return None

    def _adf_test(self, df):
        if df is None or len(df) < 30:
            print("❌ 資料不足，ADF 檢定跳過")
            return
        result = adfuller(df['log_return'], autolag='AIC')
        status = '✅ 平穩' if result[1] < 0.05 else '❌ 有單位根'
        print(f"[GARCH] ADF Test: 對數報酬率 | p-value: {result[1]:.4f} → {status}")

    def _detect_regime(self, df):
        df = df.copy()
        if len(df) < 20:
            df['regime'] = 2
            return df
        df['return_20'] = df['Close'].pct_change(20)
        conditions = [
            (df['return_20'] > 0.08),
            (df['return_20'] < -0.08),
            (True)
        ]
        df['regime'] = np.select(conditions, [0, 1, 2], default=2)
        print("\n[GARCH] Markov Transition Matrix")
        print(pd.crosstab(df['regime'].shift(1), df['regime'], normalize='index').round(3))
        return df

    def _add_rolling_garch(self, df):
        """滾動 EGARCH(1,1) — 固定模型換取速度"""
        df = df.copy()
        df['garch_vol'] = np.nan
        print(f"\n[GARCH] 🔥 Ultra Fast Mode → 固定 EGARCH(1,1) window=180")
        for i in range(180, len(df)):
            ret_window = df['log_return'].iloc[i-180:i]
            try:
                am = arch_model(ret_window, vol='EGARCH', p=1, q=1, dist='Normal')
                res = am.fit(disp='off', show_warning=False)
                df.loc[df.index[i], 'garch_vol'] = res.conditional_volatility.iloc[-1]
                df.loc[df.index[i], 'best_model'] = 'EGARCH'
            except Exception:
                pass
        # ★ 關鍵修復：前 180 筆 ffill 無法填補，需 bfill
        df['garch_vol'] = df['garch_vol'].ffill().bfill()
        rolling_std = df['garch_vol'].rolling(20, min_periods=1).std().replace(0, 1)
        df['garch_vol_z'] = (
            (df['garch_vol'] - df['garch_vol'].rolling(20, min_periods=1).mean()) / rolling_std
        )
        nan_count = df['garch_vol'].isna().sum()
        print(f"[GARCH] GARCH 完成！garch_vol NaN 數量: {nan_count}")
        return df

    def _create_features(self, df):
        df = df.copy()
        for i in [5, 10, 20]:
            df[f'ret_{i}'] = np.log(df['Close'] / df['Close'].shift(i))
            df[f'vol_{i}'] = df[f'ret_{i}'].rolling(i).std()
            df[f'mom_{i}'] = df[f'ret_{i}'] / df[f'vol_{i}'].replace(0, 1)
        # volume_z：std=0 時用 replace(0,1) 避免 NaN
        vol_std = df['Volume'].rolling(20).std().replace(0, 1)
        df['volume_z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / vol_std
        regime_dummies = pd.get_dummies(df['regime'], prefix='regime')
        df = pd.concat([df, regime_dummies], axis=1)

        # ★ 排除 garch_vol_z 避免 dropna 炸裂
        prefixes = ('ret_', 'vol_', 'mom_', 'volume_z', 'regime_', 'garch_vol')
        self.feature_cols = [col for col in df.columns
                             if col.startswith(prefixes) and col != 'garch_vol_z']

        df_clean = df.dropna(subset=self.feature_cols).copy()
        print(f"[GARCH] 特徵工程 → 保留 {len(df_clean)} 筆資料 | 特徵數: {len(self.feature_cols)}")
        return df_clean, self.feature_cols

    def _walk_forward_retraining(self):
        print(f"\n[GARCH] Walk-Forward（每 {self.retrain_every} 天）...")
        strategy_returns = []
        positions = []
        epochs = 30 if self.fast_mode else 60

        for start in range(self.seq_length, len(self.df_features), self.retrain_every):
            end = min(start + self.retrain_every, len(self.df_features))

            # KS 分布監控
            if start > self.seq_length + 50:
                recent = self.df_features['log_return'].iloc[start-50:start]
                train = self.df_features['log_return'].iloc[:start-50]
                _, p_value = ks_2samp(train, recent)
                if p_value < 0.05:
                    alert = f"⚠️ KS 分布偏移警報！p-value={p_value:.4f}"
                    print(f"[MONITOR] {alert}")
                    self.send_telegram(alert)

            train_feat = self.df_features[self.feature_cols].iloc[:start].astype(np.float32)
            train_target = self.df_features['target'].iloc[:start].astype(np.float32)

            print(f"[RETRAIN] 第 {start} 根 K 線重新訓練（epochs={epochs}）...")
            self.model = self._train_lstm(train_feat, train_target, epochs)
            self.send_telegram(f"✅ Retraining 完成 @ 第 {start} 根")

            # 滾動預測
            self.model.eval()
            with torch.no_grad():
                for i in range(start, end):
                    seq_data = (
                        self.df_features[self.feature_cols]
                        .iloc[i-self.seq_length:i]
                        .values.astype(np.float32)
                    )
                    seq = torch.tensor(
                        seq_data.reshape(1, self.seq_length, -1),
                        dtype=torch.float32
                    )
                    prob = self.model(seq).item()
                    f = max(0, min(0.5 * (2 * prob - 1), 2.0))
                    actual_ret = self.df_features['log_return'].iloc[i]
                    strategy_returns.append(f * actual_ret)
                    positions.append(f)

        return strategy_returns, positions

    def _train_lstm(self, train_feat, train_target, epochs):
        class RegimeLSTM(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, 64, 2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(64, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.sigmoid(self.fc(out[:, -1, :]))

        class _Dataset(Dataset):
            def __init__(self, X, y, seq_length):
                self.X = X.values
                self.y = y.values
                self.seq_length = seq_length
            def __len__(self):
                return len(self.X) - self.seq_length
            def __getitem__(self, idx):
                return (
                    torch.tensor(self.X[idx:idx+self.seq_length], dtype=torch.float32),
                    torch.tensor(self.y[idx+self.seq_length], dtype=torch.float32)
                )

        ds = _Dataset(train_feat, train_target, self.seq_length)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        model = RegimeLSTM(len(self.feature_cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

        return model

    def _run_backtest(self, strategy_returns):
        equity = 10000 * (1 + np.cumsum(strategy_returns))
        total_ret = (equity[-1] / 10000 - 1) * 100
        ann_ret = total_ret / (len(strategy_returns) / 252)
        sharpe = (
            np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            if np.std(strategy_returns) != 0 else 0
        )
        max_dd = (equity / np.maximum.accumulate(equity)).min() - 1
        win_rate = np.mean(np.array(strategy_returns) > 0) * 100

        report = f"""
🚀 GARCH QUANT v6.0 回測報告
總報酬:   {total_ret:.2f}%
年化報酬: {ann_ret:.2f}%
Sharpe:   {sharpe:.3f}
最大回撤: {max_dd*100:.2f}%
方向準確率: {win_rate:.1f}%
"""
        print(report)
        self.send_telegram(report)

    def predict_next(self):
        """即時預測下一期"""
        if self.model is None:
            print("❌ 模型未訓練，請先執行 run()")
            return None
        self.model.eval()
        with torch.no_grad():
            seq_data = (
                self.df_features[self.feature_cols]
                .iloc[-self.seq_length:]
                .values.astype(np.float32)
            )
            seq = torch.tensor(
                seq_data.reshape(1, self.seq_length, -1),
                dtype=torch.float32
            )
            prob = self.model(seq).item()
            f = max(0, min(0.5 * (2 * prob - 1), 2.0))
        print(f"下一期正報酬機率: {prob*100:.1f}% | Kelly 倉位: {f*100:.1f}%")
        return {"probability": prob, "kelly_fraction": f}

    def get_current_garch(self):
        """回傳目前 GARCH 狀態"""
        if self.df_features is None:
            return None
        last_vol = self.df_features['garch_vol'].iloc[-1] \
            if 'garch_vol' in self.df_features.columns else None
        model_name = self.df_features['best_model'].iloc[-1] \
            if 'best_model' in self.df_features.columns else 'EGARCH'
        if last_vol:
            print(f"目前 GARCH: {model_name} | 波動率: {last_vol:.4f}")
        else:
            print(f"目前 GARCH: {model_name}")
        return {"model": model_name, "volatility": last_vol}

    def run(self):
        """一鍵執行完整流程"""
        df = self._get_data()
        if df is None:
            return {"status": "error", "message": "資料取得失敗"}
        self._adf_test(df)
        df = self._detect_regime(df)
        df = self._add_rolling_garch(df)
        self.df_features, self.feature_cols = self._create_features(df)
        self.df_features['target'] = (
            self.df_features['log_return'].shift(-1) > 0
        ).astype(int)
        self.df_features = self.df_features.dropna(
            subset=self.feature_cols + ['target']
        )

        strategy_returns, _ = self._walk_forward_retraining()
        self._run_backtest(strategy_returns)
        print("✅ GARCH QUANT v6.0 執行完成！")
        return {"status": "success"}


if __name__ == "__main__":
    skill = GARCHQuantSkill(
        ticker="BTC-USD",
        retrain_every=30,
        fast_mode=True,
        telegram_token=None,
        telegram_chat_id=None
    )
    result = skill.run()
    print(result)
