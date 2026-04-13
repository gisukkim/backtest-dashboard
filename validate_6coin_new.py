"""
================================================================================
6코인 NEW 파라미터 검증 (3가지 개선방향)
================================================================================
1. Walk-Forward 검증: 2020~2024 학습기간 → 2025~ 테스트기간
2. 파라미터 강건성 테스트: 최적값 ±10%, ±20% 변동 시 성과 변화
3. 리스크 관리: MDD 기반 전략 중단 시뮬레이션
================================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
import time
import itertools

from backtest_bots import (
    prepare_coin_data, prepare_signals,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H,
    calculate_portfolio_performance,
)

SAVE_DIR = os.path.expanduser("~/Downloads")
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5
CAPITAL = 10000

# 6코인 NEW 파라미터 (최적화 결과)
NEW_CONFIGS = {
    'BTCUSDT': {'priority': 'short', 'short_ma': 254, 'short_sk': 27, 'short_sks': 23, 'short_sd': 19, 'short_lev': 1,
                'long_ma': 350, 'long_sk': 36, 'long_sks': 32, 'long_sd': 10, 'long_lev': 5},
    'ETHUSDT': {'priority': 'long', 'long_ma': 322, 'long_sk': 54, 'long_sks': 10, 'long_sd': 36, 'long_lev': 5,
                'short_ma': 220, 'short_sk': 31, 'short_sks': 44, 'short_sd': 26, 'short_lev': 2},
    'XRPUSDT': {'priority': 'short', 'short_ma': 269, 'short_sk': 121, 'short_sks': 35, 'short_sd': 47, 'short_lev': 1,
                'long_ma': 107, 'long_sk': 14, 'long_sks': 13, 'long_sd': 23, 'long_lev': 5},
    'SOLUSDT': {'priority': 'long', 'long_ma': 73, 'long_sk': 33, 'long_sks': 16, 'long_sd': 38, 'long_lev': 4,
                'short_ma': 314, 'short_sk': 37, 'short_sks': 34, 'short_sd': 44, 'short_lev': 1},
    'DOGEUSDT': {'priority': 'short', 'short_ma': 250, 'short_sk': 36, 'short_sks': 15, 'short_sd': 40, 'short_lev': 1,
                 'long_ma': 31, 'long_sk': 48, 'long_sks': 50, 'long_sd': 17, 'long_lev': 2},
    'ADAUSDT': {'priority': 'short', 'short_ma': 80, 'short_sk': 31, 'short_sks': 77, 'short_sd': 46, 'short_lev': 1,
                'long_ma': 296, 'long_sk': 19, 'long_sks': 53, 'long_sd': 15, 'long_lev': 3},
}


# ==========================================
# 포트폴리오 백테스트 엔진 (기간 제한 + 모드)
# ==========================================
def run_portfolio(configs_dict, coin_data_cache, initial_capital=CAPITAL,
                  start_date=None, end_date=None):
    """
    start_date/end_date: 백테스트 기간 제한 (pd.Timestamp)
    """
    coin_signals = {}
    for symbol, config in configs_dict.items():
        data = coin_data_cache.get(symbol)
        if data is None:
            continue
        df_4h = data['df_4h']
        df_daily = data['df_daily']
        df_long = prepare_signals(df_4h, df_4h, df_daily,
                                  config['long_ma'], config['long_sk'],
                                  config['long_sks'], config['long_sd'], use_1h=False)
        df_short = prepare_signals(df_4h, df_4h, df_daily,
                                   config['short_ma'], config['short_sk'],
                                   config['short_sks'], config['short_sd'], use_1h=False)
        df_bt = df_long[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                          'prev_slow_k', 'prev_slow_d']].copy()
        df_bt.rename(columns={'ma': 'ma_long', 'prev_slow_k': 'long_k', 'prev_slow_d': 'long_d'}, inplace=True)
        df_bt['ma_short'] = df_short['ma'].values
        df_bt['short_k'] = df_short['prev_slow_k'].values
        df_bt['short_d'] = df_short['prev_slow_d'].values
        df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d', 'ma_short', 'short_k', 'short_d'])
        df_bt = df_bt.set_index('timestamp').sort_index()

        # 기간 필터
        if start_date:
            df_bt = df_bt[df_bt.index >= start_date]
        if end_date:
            df_bt = df_bt[df_bt.index <= end_date]

        if len(df_bt) >= 50:
            coin_signals[symbol] = {
                'df': df_bt,
                'long_lev': config['long_lev'], 'short_lev': config['short_lev'],
                'priority': config.get('priority', 'long'),
            }

    if not coin_signals:
        return None

    all_ts = set()
    for s_data in coin_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    timeline = sorted(all_ts)

    cash = initial_capital
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0
    long_trades_cnt = 0
    short_trades_cnt = 0

    for ts in timeline:
        active_coins = [s for s, d in coin_signals.items() if ts in d['df'].index]
        n_active = len(active_coins)

        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        for symbol in active_coins:
            s_data = coin_signals[symbol]
            df = s_data['df']
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            op = curr['open']
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
            prev = df.iloc[curr_idx - 1]
            long_lev = s_data['long_lev']
            short_lev = s_data['short_lev']
            priority = s_data['priority']
            long_signal = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
            short_signal = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
            pos = positions.get(symbol)

            def settle_pos(p):
                nonlocal cash, total_trades
                pr = op / p['entry_price'] - 1
                if p['side'] == 'short':
                    pr = -pr
                pnl = p['margin'] * pr * p['lev']
                settled = p['margin'] + pnl - p['cum_funding']
                exit_not = abs(p['margin'] * p['lev'] * (op / p['entry_price']))
                settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                cash += max(settled, 0)
                total_trades += 1

            def open_pos(side, lev):
                nonlocal cash, total_trades, long_trades_cnt, short_trades_cnt
                total_eq = cash + sum(p['margin'] for p in positions.values())
                alloc = min(total_eq / n_active, cash * 0.995)
                if alloc > 1:
                    margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                    if margin > 0:
                        cash -= alloc
                        positions[symbol] = {'side': side, 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': lev}
                        total_trades += 1
                        if side == 'long':
                            long_trades_cnt += 1
                        else:
                            short_trades_cnt += 1

            if priority == 'long':
                if long_signal:
                    if pos and pos['side'] == 'short':
                        settle_pos(pos); del positions[symbol]; pos = None
                    if pos is None:
                        open_pos('long', long_lev)
                elif short_signal and (pos is None or pos['side'] != 'long'):
                    if pos is None:
                        open_pos('short', short_lev)
                else:
                    if pos:
                        settle_pos(pos); del positions[symbol]
            else:
                if short_signal:
                    if pos and pos['side'] == 'long':
                        settle_pos(pos); del positions[symbol]; pos = None
                    if pos is None:
                        open_pos('short', short_lev)
                elif long_signal and (pos is None or pos['side'] != 'short'):
                    if pos is None:
                        open_pos('long', long_lev)
                else:
                    if pos:
                        settle_pos(pos); del positions[symbol]

            pos = positions.get(symbol)
            if pos and ts in df.index:
                pos['cum_funding'] += pos['margin'] * pos['lev'] * FUNDING_PER_4H
                if pos['side'] == 'long' and pos['lev'] > 0:
                    if curr['low'] <= pos['entry_price'] * (1 - 1 / pos['lev']):
                        del positions[symbol]; total_trades += 1
                elif pos['side'] == 'short' and pos['lev'] > 0:
                    if curr['high'] >= pos['entry_price'] * (1 + 1 / pos['lev']):
                        del positions[symbol]; total_trades += 1

        unrealized = 0
        for sym, pos in positions.items():
            df = coin_signals[sym]['df']
            if ts in df.index:
                cp = df.loc[ts, 'close']
                pr = cp / pos['entry_price'] - 1
                if pos['side'] == 'short':
                    pr = -pr
                val = pos['margin'] + pos['margin'] * pr * pos['lev'] - pos['cum_funding']
                unrealized += max(val, 0)
            else:
                unrealized += pos['margin']
        equity_curve.append(cash + unrealized)
        eq_timestamps.append(ts)

    for sym, pos in list(positions.items()):
        df = coin_signals[sym]['df']
        lp = df.iloc[-1]['close']
        pr = lp / pos['entry_price'] - 1
        if pos['side'] == 'short':
            pr = -pr
        pnl = pos['margin'] * pr * pos['lev']
        settled = pos['margin'] + pnl - pos['cum_funding']
        exit_not = abs(pos['margin'] * pos['lev'] * (lp / pos['entry_price']))
        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(settled, 0)

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf is None:
        return None
    perf['trades'] = total_trades
    perf['long_trades'] = long_trades_cnt
    perf['short_trades'] = short_trades_cnt
    perf['equity_curve'] = equity_curve
    perf['timestamps'] = eq_timestamps
    return perf


def perturb_configs(base_configs, factor):
    """파라미터를 factor 비율로 변동 (MA, Stoch K/Ks/D)"""
    new = {}
    for sym, cfg in base_configs.items():
        c = dict(cfg)
        for key in ['long_ma', 'short_ma']:
            c[key] = max(5, int(cfg[key] * factor))
        for key in ['long_sk', 'long_sks', 'long_sd', 'short_sk', 'short_sks', 'short_sd']:
            c[key] = max(2, int(cfg[key] * factor))
        new[sym] = c
    return new


# ============================================================
# 1. Walk-Forward 검증
# ============================================================
def test_walk_forward(coin_data_cache):
    print("\n" + "=" * 70)
    print("  [1] Walk-Forward 검증")
    print("=" * 70)

    # 구간 정의
    periods = [
        ("In-Sample 전체", "2020-01-01", "2024-12-31"),
        ("Out-of-Sample 2025~", "2025-01-01", "2026-12-31"),
        ("IS: 2020-2022", "2020-01-01", "2022-12-31"),
        ("OOS: 2023-2024", "2023-01-01", "2024-12-31"),
        ("IS: 2020-2023", "2020-01-01", "2023-12-31"),
        ("OOS: 2024", "2024-01-01", "2024-12-31"),
        ("IS: 2020-2024H1", "2020-01-01", "2024-06-30"),
        ("OOS: 2024H2~", "2024-07-01", "2026-12-31"),
    ]

    results = []
    for label, s, e in periods:
        sd = pd.Timestamp(s)
        ed = pd.Timestamp(e)
        perf = run_portfolio(NEW_CONFIGS, coin_data_cache, start_date=sd, end_date=ed)
        if perf:
            results.append({
                'period': label,
                'start': s[:7], 'end': e[:7],
                'cagr': perf['cagr'],
                'mdd': perf['mdd'],
                'sharpe': perf['sharpe'],
                'trades': perf['trades'],
                'final': perf['final_equity'],
            })
            oos = "OOS" in label
            marker = " ◀ OUT-OF-SAMPLE" if oos else ""
            print(f"  {label:<22} CAGR={perf['cagr']:>8.1f}%  MDD={perf['mdd']:>7.1f}%  "
                  f"Sharpe={perf['sharpe']:.3f}  거래={perf['trades']}{marker}")

    # 연도별 구간 검증
    print(f"\n  --- 연도별 성과 ---")
    yearly_results = []
    for year in range(2020, 2027):
        sd = pd.Timestamp(f"{year}-01-01")
        ed = pd.Timestamp(f"{year}-12-31")
        perf = run_portfolio(NEW_CONFIGS, coin_data_cache, start_date=sd, end_date=ed)
        if perf:
            yearly_results.append({
                'year': year, 'cagr': perf['cagr'], 'mdd': perf['mdd'],
                'sharpe': perf['sharpe'], 'return': perf['total_return'],
                'equity_curve': perf['equity_curve'], 'timestamps': perf['timestamps'],
            })
            is_oos = year >= 2025
            marker = " ◀ OOS" if is_oos else ""
            print(f"  {year}: Return={perf['total_return']:>8.1f}%  MDD={perf['mdd']:>7.1f}%  "
                  f"Sharpe={perf['sharpe']:.3f}{marker}")

    return results, yearly_results


# ============================================================
# 2. 파라미터 강건성 테스트
# ============================================================
def test_robustness(coin_data_cache):
    print("\n" + "=" * 70)
    print("  [2] 파라미터 강건성 테스트")
    print("=" * 70)

    factors = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    robustness_results = []

    for factor in factors:
        cfgs = perturb_configs(NEW_CONFIGS, factor)
        perf = run_portfolio(cfgs, coin_data_cache)
        if perf:
            pct_change = (factor - 1.0) * 100
            robustness_results.append({
                'factor': factor,
                'pct': pct_change,
                'cagr': perf['cagr'],
                'mdd': perf['mdd'],
                'sharpe': perf['sharpe'],
                'trades': perf['trades'],
            })
            opt = " ◀ OPTIMAL" if factor == 1.0 else ""
            print(f"  {pct_change:>+5.0f}%: CAGR={perf['cagr']:>8.1f}%  MDD={perf['mdd']:>7.1f}%  "
                  f"Sharpe={perf['sharpe']:.3f}  거래={perf['trades']}{opt}")

    # 개별 파라미터 민감도 (MA만 / Stoch만 변동)
    print(f"\n  --- MA만 변동 (Stoch 고정) ---")
    ma_results = []
    for factor in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        cfgs = {}
        for sym, cfg in NEW_CONFIGS.items():
            c = dict(cfg)
            c['long_ma'] = max(5, int(cfg['long_ma'] * factor))
            c['short_ma'] = max(5, int(cfg['short_ma'] * factor))
            cfgs[sym] = c
        perf = run_portfolio(cfgs, coin_data_cache)
        if perf:
            ma_results.append({'factor': factor, 'pct': (factor-1)*100, 'cagr': perf['cagr'],
                               'mdd': perf['mdd'], 'sharpe': perf['sharpe']})
            print(f"  MA {(factor-1)*100:>+5.0f}%: CAGR={perf['cagr']:>8.1f}%  MDD={perf['mdd']:>7.1f}%  Sharpe={perf['sharpe']:.3f}")

    print(f"\n  --- Stoch만 변동 (MA 고정) ---")
    stoch_results = []
    for factor in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        cfgs = {}
        for sym, cfg in NEW_CONFIGS.items():
            c = dict(cfg)
            for key in ['long_sk', 'long_sks', 'long_sd', 'short_sk', 'short_sks', 'short_sd']:
                c[key] = max(2, int(cfg[key] * factor))
            cfgs[sym] = c
        perf = run_portfolio(cfgs, coin_data_cache)
        if perf:
            stoch_results.append({'factor': factor, 'pct': (factor-1)*100, 'cagr': perf['cagr'],
                                  'mdd': perf['mdd'], 'sharpe': perf['sharpe']})
            print(f"  Stoch {(factor-1)*100:>+5.0f}%: CAGR={perf['cagr']:>8.1f}%  MDD={perf['mdd']:>7.1f}%  Sharpe={perf['sharpe']:.3f}")

    return robustness_results, ma_results, stoch_results


# ============================================================
# 3. 리스크 관리: MDD 손절 시뮬레이션
# ============================================================
def test_risk_management(coin_data_cache):
    print("\n" + "=" * 70)
    print("  [3] 리스크 관리: MDD 기반 전략 중단 시뮬레이션")
    print("=" * 70)

    # 기본 전체 백테스트
    base_perf = run_portfolio(NEW_CONFIGS, coin_data_cache)
    if not base_perf:
        return None

    eq = np.array(base_perf['equity_curve'])
    ts = base_perf['timestamps']

    # MDD 손절 시뮬레이션: 특정 MDD 도달시 전략 중단 후 현금 보유
    thresholds = [-20, -25, -30, -35, -40, -45, -50, -55, -60]
    print(f"\n  기본 전략: CAGR={base_perf['cagr']:.1f}%, MDD={base_perf['mdd']:.1f}%, 최종=${eq[-1]:,.0f}")
    print(f"\n  {'MDD 손절':>10} {'최종 자산':>15} {'최종 수익률':>12} {'중단 시점':>12} {'중단 횟수':>8}")
    print(f"  {'-'*60}")

    mdd_results = []
    for threshold in thresholds:
        # 시뮬: MDD 도달 → 전략 중단, 60일(360봉) 후 재개
        COOLDOWN = 360  # 4H봉 기준 약 60일
        simulated_eq = [eq[0]]
        peak = eq[0]
        stopped = False
        stop_until = 0
        stop_count = 0

        for i in range(1, len(eq)):
            if stopped and i < stop_until:
                simulated_eq.append(simulated_eq[-1])  # 현금 보유
                continue
            elif stopped and i >= stop_until:
                stopped = False
                # 재개: 현재 보유 현금으로 전략 재개, 비율 적용
                ratio = simulated_eq[-1] / eq[i-1] if eq[i-1] > 0 else 1
                val = eq[i] * ratio
                simulated_eq.append(val)
                peak = val
            else:
                ratio = simulated_eq[-1] / eq[i-1] if eq[i-1] > 0 else 1
                val = eq[i] * ratio
                simulated_eq.append(val)
                if val > peak:
                    peak = val
                dd = (val - peak) / peak * 100 if peak > 0 else 0
                if dd <= threshold:
                    stopped = True
                    stop_until = i + COOLDOWN
                    stop_count += 1

        final = simulated_eq[-1]
        ret = (final / simulated_eq[0] - 1) * 100

        # 첫 중단 시점 찾기
        first_stop = "없음"
        peak_check = eq[0]
        for i in range(1, len(eq)):
            if eq[i] > peak_check:
                peak_check = eq[i]
            dd = (eq[i] - peak_check) / peak_check * 100
            if dd <= threshold:
                first_stop = str(ts[i].date())
                break

        mdd_results.append({
            'threshold': threshold, 'final': final, 'return': ret,
            'first_stop': first_stop, 'stop_count': stop_count,
            'eq_curve': simulated_eq,
        })
        print(f"  {threshold:>8}%  ${final:>13,.0f}  {ret:>10.0f}%  {first_stop:>12}  {stop_count:>6}")

    return base_perf, mdd_results


# ============================================================
# 차트 생성
# ============================================================
def plot_all_charts(wf_results, yearly_results, robust_results, ma_results, stoch_results,
                    base_perf, mdd_results, save_dir):

    # ── 1. Walk-Forward: 연도별 에쿼티 ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1-1. IS vs OOS 비교 바차트
    ax = axes[0][0]
    wf_df = pd.DataFrame(wf_results)
    if not wf_df.empty:
        colors_wf = ['#4CAF50' if 'OOS' not in r['period'] else '#FF5722' for _, r in wf_df.iterrows()]
        bars = ax.barh(range(len(wf_df)), wf_df['cagr'], color=colors_wf, alpha=0.8)
        ax.set_yticks(range(len(wf_df)))
        ax.set_yticklabels(wf_df['period'], fontsize=9)
        for bar, val in zip(bars, wf_df['cagr']):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}%', va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel('CAGR (%)')
        ax.set_title('[1] Walk-Forward: In-Sample vs Out-of-Sample CAGR', fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.legend(['In-Sample', 'Out-of-Sample'], loc='lower right')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#4CAF50', label='In-Sample'),
                           Patch(color='#FF5722', label='Out-of-Sample')], loc='lower right')

    # 1-2. 연도별 수익률 + Sharpe
    ax = axes[0][1]
    if yearly_results:
        yr_df = pd.DataFrame(yearly_results)
        x = np.arange(len(yr_df))
        w = 0.35
        ax.bar(x - w/2, yr_df['return'], w, label='Return %', color='#2196F3', alpha=0.8)
        ax2_twin = ax.twinx()
        ax2_twin.plot(x, yr_df['sharpe'], 'o-', color='#FF5722', linewidth=2, markersize=8, label='Sharpe')
        ax2_twin.axhline(y=1.0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax2_twin.set_ylabel('Sharpe', color='#FF5722')
        ax.set_xticks(x)
        ax.set_xticklabels(yr_df['year'])
        ax.set_ylabel('Return (%)')
        ax.set_title('[1] Walk-Forward: Annual Return + Sharpe', fontweight='bold')
        ax.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        # OOS 구분 배경색
        for i, yr in enumerate(yr_df['year']):
            if yr >= 2025:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='red')
        ax.grid(True, alpha=0.3, axis='y')

    # 2. 파라미터 강건성
    ax = axes[1][0]
    if robust_results:
        rb_df = pd.DataFrame(robust_results)
        ax.plot(rb_df['pct'], rb_df['cagr'], 'o-', color='#2196F3', linewidth=2, markersize=8, label='All Params')
    if ma_results:
        ma_df = pd.DataFrame(ma_results)
        ax.plot(ma_df['pct'], ma_df['cagr'], 's--', color='#4CAF50', linewidth=1.5, markersize=6, label='MA only')
    if stoch_results:
        st_df = pd.DataFrame(stoch_results)
        ax.plot(st_df['pct'], st_df['cagr'], '^--', color='#FF5722', linewidth=1.5, markersize=6, label='Stoch only')
    ax.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Optimal')
    ax.set_xlabel('Parameter Change (%)')
    ax.set_ylabel('CAGR (%)')
    ax.set_title('[2] Parameter Robustness: CAGR vs Perturbation', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. MDD 손절 시뮬레이션
    ax = axes[1][1]
    if mdd_results:
        thresholds = [r['threshold'] for r in mdd_results]
        returns = [r['return'] for r in mdd_results]
        base_ret = (base_perf['equity_curve'][-1] / base_perf['equity_curve'][0] - 1) * 100
        ax.bar(range(len(thresholds)), returns, color=['#4CAF50' if r > base_ret else '#FF5722' for r in returns],
               alpha=0.8)
        ax.axhline(y=base_ret, color='blue', linewidth=2, linestyle='--', label=f'No Stop (Return={base_ret:.0f}%)')
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t}%' for t in thresholds], fontsize=9)
        ax.set_xlabel('MDD Stop Threshold')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('[3] Risk Management: MDD Stop-Loss Effect', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (th, ret) in enumerate(zip(thresholds, returns)):
            ax.text(i, ret + max(abs(ret) * 0.02, base_ret * 0.03), f'{ret:.0f}%',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, '6coin_new_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  차트 저장: {path}")

    # ── 추가: 강건성 Sharpe + MDD 차트 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sharpe 강건성
    ax = axes[0]
    if robust_results:
        rb_df = pd.DataFrame(robust_results)
        ax.plot(rb_df['pct'], rb_df['sharpe'], 'o-', color='#2196F3', linewidth=2, markersize=8, label='All')
    if ma_results:
        ma_df = pd.DataFrame(ma_results)
        ax.plot(ma_df['pct'], ma_df['sharpe'], 's--', color='#4CAF50', linewidth=1.5, markersize=6, label='MA')
    if stoch_results:
        st_df = pd.DataFrame(stoch_results)
        ax.plot(st_df['pct'], st_df['sharpe'], '^--', color='#FF5722', linewidth=1.5, markersize=6, label='Stoch')
    ax.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Parameter Change (%)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Robustness: Sharpe', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MDD 강건성
    ax = axes[1]
    if robust_results:
        rb_df = pd.DataFrame(robust_results)
        ax.plot(rb_df['pct'], rb_df['mdd'], 'o-', color='#2196F3', linewidth=2, markersize=8, label='All')
    if ma_results:
        ma_df = pd.DataFrame(ma_results)
        ax.plot(ma_df['pct'], ma_df['mdd'], 's--', color='#4CAF50', linewidth=1.5, markersize=6, label='MA')
    if stoch_results:
        st_df = pd.DataFrame(stoch_results)
        ax.plot(st_df['pct'], st_df['mdd'], '^--', color='#FF5722', linewidth=1.5, markersize=6, label='Stoch')
    ax.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Parameter Change (%)')
    ax.set_ylabel('MDD (%)')
    ax.set_title('Robustness: MDD', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MDD 손절 에쿼티 커브 (선택적)
    ax = axes[2]
    eq = np.array(base_perf['equity_curve'])
    ts_arr = base_perf['timestamps']
    ax.semilogy(ts_arr, eq, color='blue', linewidth=1.5, label='No Stop', alpha=0.9)
    for r in mdd_results:
        if r['threshold'] in [-30, -40, -50]:
            ax.semilogy(ts_arr, r['eq_curve'], linewidth=1, alpha=0.7,
                        label=f"Stop@{r['threshold']}%")
    ax.set_ylabel('Equity (log)')
    ax.set_title('MDD Stop: Equity Curves', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    plt.tight_layout()
    path2 = os.path.join(save_dir, '6coin_new_validation_detail.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  차트 저장: {path2}")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 70)
    print("  6코인 NEW 파라미터 검증 (3가지 개선방향)")
    print("=" * 70)
    start_time = time.time()

    # 데이터 로드
    print("\n  --- 데이터 로드 ---")
    coin_data_cache = {}
    for symbol in sorted(NEW_CONFIGS.keys()):
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data_cache[symbol] = data
            print(f"  {symbol}: {data['days']}일")

    # 1. Walk-Forward
    wf_results, yearly_results = test_walk_forward(coin_data_cache)

    # 2. 강건성
    robust_results, ma_results, stoch_results = test_robustness(coin_data_cache)

    # 3. 리스크 관리
    base_perf, mdd_results = test_risk_management(coin_data_cache)

    # 차트
    print("\n  --- 차트 생성 ---")
    plot_all_charts(wf_results, yearly_results, robust_results, ma_results, stoch_results,
                    base_perf, mdd_results, SAVE_DIR)

    # ==========================================
    # 종합 진단
    # ==========================================
    print("\n" + "=" * 70)
    print("  종합 진단")
    print("=" * 70)

    # WF 진단
    is_perfs = [r for r in wf_results if 'OOS' not in r['period']]
    oos_perfs = [r for r in wf_results if 'OOS' in r['period']]
    avg_is_cagr = np.mean([r['cagr'] for r in is_perfs]) if is_perfs else 0
    avg_oos_cagr = np.mean([r['cagr'] for r in oos_perfs]) if oos_perfs else 0
    wf_ratio = avg_oos_cagr / avg_is_cagr if avg_is_cagr > 0 else 0

    print(f"\n  [1] Walk-Forward")
    print(f"      IS 평균 CAGR: {avg_is_cagr:.1f}%")
    print(f"      OOS 평균 CAGR: {avg_oos_cagr:.1f}%")
    print(f"      WF Ratio (OOS/IS): {wf_ratio:.2f}")
    if wf_ratio > 0.5:
        print(f"      → 양호 (OOS가 IS의 {wf_ratio*100:.0f}%)")
    elif wf_ratio > 0.2:
        print(f"      → 주의 (OOS가 IS 대비 상당히 하락)")
    else:
        print(f"      → 위험 (심각한 오버피팅 의심)")

    # 강건성 진단
    if robust_results:
        opt_cagr = [r['cagr'] for r in robust_results if r['factor'] == 1.0][0]
        nearby = [r['cagr'] for r in robust_results if 0.85 <= r['factor'] <= 1.15]
        cagr_std = np.std(nearby)
        cagr_min = min(nearby)
        cagr_cv = cagr_std / abs(np.mean(nearby)) * 100 if np.mean(nearby) != 0 else 999
        print(f"\n  [2] 파라미터 강건성 (±15% 범위)")
        print(f"      최적 CAGR: {opt_cagr:.1f}%")
        print(f"      ±15% 범위 최소 CAGR: {cagr_min:.1f}%")
        print(f"      ±15% 범위 CAGR 표준편차: {cagr_std:.1f}%")
        print(f"      변동계수(CV): {cagr_cv:.1f}%")
        if cagr_cv < 20:
            print(f"      → 강건 (파라미터 변동에 안정적)")
        elif cagr_cv < 50:
            print(f"      → 보통 (일부 민감도 있음)")
        else:
            print(f"      → 취약 (파라미터 의존도 높음, 오버피팅 위험)")

    # 2025 성과
    yr_2025 = [r for r in yearly_results if r['year'] == 2025]
    yr_2026 = [r for r in yearly_results if r['year'] == 2026]
    print(f"\n  [3] 최근 실전 구간")
    if yr_2025:
        print(f"      2025: Return={yr_2025[0]['return']:.1f}%, MDD={yr_2025[0]['mdd']:.1f}%, Sharpe={yr_2025[0]['sharpe']:.3f}")
    if yr_2026:
        print(f"      2026: Return={yr_2026[0]['return']:.1f}%, MDD={yr_2026[0]['mdd']:.1f}%, Sharpe={yr_2026[0]['sharpe']:.3f}")

    elapsed = time.time() - start_time
    print(f"\n  완료! ({elapsed:.0f}초)")


if __name__ == '__main__':
    main()
