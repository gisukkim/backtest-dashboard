"""5개 코인으로 빠른 검증 (원본 vs 수정 버전 비교)"""
import sys
sys.path.insert(0, '.')

from optimize_short_fixed import (
    backtest_short, prepare_coin_data,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_PER_4H
)
from optuna.samplers import TPESampler
import optuna
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

TEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT']
TRIALS = 30

PARAM_RANGES = {
    'ma_range': (30, 350),
    'stoch_k_range': (20, 150),
    'stoch_smooth_range': (5, 80),
    'stoch_d_range': (3, 50),
    'leverage_range': (1, 5),
}

print("=" * 60)
print("수정 버전 검증 테스트 (5개 코인)")
print("=" * 60)
print(f"수수료: {FEE_RATE*100:.2f}%, 슬리피지: {SLIPPAGE_PCT*100:.2f}%")
print(f"펀딩비: {FUNDING_PER_4H*100:.4f}% / 4H봉")
print(f"방식: 포지션 내 비복리, 거래 간 복리")
print("=" * 60)

results = []
for symbol in TEST_SYMBOLS:
    print(f"\n--- {symbol} ---")
    data = prepare_coin_data(symbol, silent=False)
    if not data:
        print(f"  데이터 부족, 스킵")
        continue

    def objective(trial):
        ma = trial.suggest_int('ma', *PARAM_RANGES['ma_range'])
        sk = trial.suggest_int('stoch_k', *PARAM_RANGES['stoch_k_range'])
        ss = trial.suggest_int('stoch_smooth', *PARAM_RANGES['stoch_smooth_range'])
        sd = trial.suggest_int('stoch_d', *PARAM_RANGES['stoch_d_range'])
        lev = trial.suggest_int('leverage', *PARAM_RANGES['leverage_range'])
        r = backtest_short(data, ma, sk, ss, sd, lev)
        if r['cagr'] < -50 or r['mdd'] < -90:
            return -9999
        return r['cagr'] + r['mdd'] * 0.5 + r['sharpe'] * 15

    sampler = TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=False)

    best = study.best_params
    r = backtest_short(data, best['ma'], best['stoch_k'], best['stoch_smooth'],
                       best['stoch_d'], best['leverage'])

    print(f"  CAGR: {r['cagr']:.1f}%, MDD: {r['mdd']:.1f}%, Sharpe: {r['sharpe']:.2f}, Trades: {r['trades']}")
    print(f"  Params: MA={best['ma']}, K={best['stoch_k']}, Smooth={best['stoch_smooth']}, D={best['stoch_d']}, Lev={best['leverage']}x")
    results.append({'symbol': symbol, **r})

print("\n" + "=" * 60)
print("요약")
print("=" * 60)
cagrs = [r['cagr'] for r in results if r['cagr'] > -900]
if cagrs:
    print(f"평균 CAGR: {np.mean(cagrs):.1f}%")
    print(f"최대 CAGR: {max(cagrs):.1f}%")
    print(f"최소 CAGR: {min(cagrs):.1f}%")
