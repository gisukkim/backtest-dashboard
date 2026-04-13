"""
================================================================================
6코인 포트폴리오 롱/숏/롱+숏 효과 분석 v2
================================================================================
compare_old_vs_new_params.py의 run_portfolio 엔진 동일 사용
- OLD 파라미터 (BITGET_CONFIGS, 롱우선 고정)
- NEW 파라미터 (최적화 CSV, 코인별 우선순위)
- 각각에 대해 SHORT only / LONG only / COMBINED 3종 백테스트
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
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime
import time

from backtest_bots import (
    prepare_coin_data, prepare_signals,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H,
    calculate_portfolio_performance,
)

SAVE_DIR = os.path.expanduser("~/Downloads")
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5
CAPITAL = 10000

# ==========================================
# 파라미터 정의
# ==========================================
OLD_CONFIGS = {
    'BTCUSDT': {'priority': 'long', 'long_ma': 219, 'long_sk': 27, 'long_sks': 33, 'long_sd': 4, 'long_lev': 5,
                'short_ma': 247, 'short_sk': 35, 'short_sks': 52, 'short_sd': 29, 'short_lev': 1},
    'ETHUSDT': {'priority': 'long', 'long_ma': 152, 'long_sk': 19, 'long_sks': 28, 'long_sd': 14, 'long_lev': 5,
                'short_ma': 248, 'short_sk': 53, 'short_sks': 40, 'short_sd': 11, 'short_lev': 2},
    'SOLUSDT': {'priority': 'long', 'long_ma': 284, 'long_sk': 37, 'long_sks': 15, 'long_sd': 36, 'long_lev': 3,
                'short_ma': 65, 'short_sk': 39, 'short_sks': 28, 'short_sd': 50, 'short_lev': 1},
    'ADAUSDT': {'priority': 'long', 'long_ma': 113, 'long_sk': 91, 'long_sks': 45, 'long_sd': 14, 'long_lev': 3,
                'short_ma': 216, 'short_sk': 107, 'short_sks': 23, 'short_sd': 19, 'short_lev': 1},
    'DOGEUSDT': {'priority': 'long', 'long_ma': 236, 'long_sk': 36, 'long_sks': 34, 'long_sd': 50, 'long_lev': 2,
                 'short_ma': 101, 'short_sk': 83, 'short_sks': 17, 'short_sd': 6, 'short_lev': 2},
    'XRPUSDT': {'priority': 'long', 'long_ma': 337, 'long_sk': 16, 'long_sks': 12, 'long_sd': 14, 'long_lev': 4,
                'short_ma': 37, 'short_sk': 116, 'short_sks': 60, 'short_sd': 20, 'short_lev': 1},
}

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
# 포트폴리오 백테스트 엔진 (compare_old_vs_new_params.py 동일)
# ==========================================
def run_portfolio(configs_dict, coin_data_cache, mode='combined', initial_capital=CAPITAL):
    """
    mode: 'combined', 'short', 'long'
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
        if len(df_bt) >= 100:
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

            # mode에 따라 신호 차단
            if mode == 'short':
                long_signal = False
            elif mode == 'long':
                short_signal = False

            pos = positions.get(symbol)

            def settle_pos(pos_to_close):
                nonlocal cash, total_trades
                pr = op / pos_to_close['entry_price'] - 1
                if pos_to_close['side'] == 'short':
                    pr = -pr
                pnl = pos_to_close['margin'] * pr * pos_to_close['lev']
                settled = pos_to_close['margin'] + pnl - pos_to_close['cum_funding']
                exit_not = abs(pos_to_close['margin'] * pos_to_close['lev'] * (op / pos_to_close['entry_price']))
                settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                cash += max(settled, 0)
                total_trades += 1

            def open_pos(side, lev):
                nonlocal cash, total_trades, long_trades_cnt, short_trades_cnt
                total_eq = cash + sum(p['margin'] for p in positions.values())
                alloc = min(total_eq / n_active, cash * 0.995)
                if alloc > 1:
                    cost = alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                    margin = alloc - cost
                    if margin > 0:
                        cash -= alloc
                        positions[symbol] = {'side': side, 'entry_price': op, 'margin': margin,
                                             'cum_funding': 0, 'lev': lev}
                        total_trades += 1
                        if side == 'long':
                            long_trades_cnt += 1
                        else:
                            short_trades_cnt += 1

            # priority 기반 로직
            if priority == 'long':
                if long_signal:
                    if pos and pos['side'] == 'short':
                        settle_pos(pos)
                        del positions[symbol]
                        pos = None
                    if pos is None:
                        open_pos('long', long_lev)
                elif short_signal and (pos is None or pos['side'] != 'long'):
                    if pos is None:
                        open_pos('short', short_lev)
                else:
                    if pos:
                        settle_pos(pos)
                        del positions[symbol]
            else:  # short priority
                if short_signal:
                    if pos and pos['side'] == 'long':
                        settle_pos(pos)
                        del positions[symbol]
                        pos = None
                    if pos is None:
                        open_pos('short', short_lev)
                elif long_signal and (pos is None or pos['side'] != 'short'):
                    if pos is None:
                        open_pos('long', long_lev)
                else:
                    if pos:
                        settle_pos(pos)
                        del positions[symbol]

            # 펀딩 + 강제청산
            pos = positions.get(symbol)
            if pos and ts in df.index:
                pos['cum_funding'] += pos['margin'] * pos['lev'] * FUNDING_PER_4H
                if pos['side'] == 'long' and pos['lev'] > 0:
                    if curr['low'] <= pos['entry_price'] * (1 - 1 / pos['lev']):
                        del positions[symbol]
                        total_trades += 1
                elif pos['side'] == 'short' and pos['lev'] > 0:
                    if curr['high'] >= pos['entry_price'] * (1 + 1 / pos['lev']):
                        del positions[symbol]
                        total_trades += 1

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

    # 마지막 정리
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
    positions.clear()

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf is None:
        return None
    perf['trades'] = total_trades
    perf['long_trades'] = long_trades_cnt
    perf['short_trades'] = short_trades_cnt
    perf['equity_curve'] = equity_curve
    perf['timestamps'] = eq_timestamps
    return perf


# ============================================================
# 차트
# ============================================================
def build_equity_df(perf):
    df = pd.DataFrame({'timestamp': perf['timestamps'], 'equity': perf['equity_curve']})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp').sort_index()


def plot_6way_charts(old_results, new_results, save_dir):
    """OLD vs NEW × SHORT/LONG/COMBINED = 6개 에쿼티 커브"""

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={'hspace': 0.05})

    styles = {
        'OLD_short':    {'color': '#E91E63', 'ls': '--', 'lw': 1.0, 'alpha': 0.6},
        'OLD_long':     {'color': '#4CAF50', 'ls': '--', 'lw': 1.0, 'alpha': 0.6},
        'OLD_combined': {'color': '#2196F3', 'ls': '--', 'lw': 1.5, 'alpha': 0.7},
        'NEW_short':    {'color': '#E91E63', 'ls': '-',  'lw': 1.2, 'alpha': 0.9},
        'NEW_long':     {'color': '#4CAF50', 'ls': '-',  'lw': 1.2, 'alpha': 0.9},
        'NEW_combined': {'color': '#2196F3', 'ls': '-',  'lw': 2.0, 'alpha': 0.95},
    }

    ax1 = axes[0]
    for tag, perf in [('OLD_short', old_results['short']),
                       ('OLD_long', old_results['long']),
                       ('OLD_combined', old_results['combined']),
                       ('NEW_short', new_results['short']),
                       ('NEW_long', new_results['long']),
                       ('NEW_combined', new_results['combined'])]:
        eq_df = build_equity_df(perf)
        s = styles[tag]
        prefix = tag.split('_')[0]
        mode = tag.split('_')[1].upper()
        label = f'{prefix} {mode} (CAGR={perf["cagr"]:.0f}%)'
        ax1.semilogy(eq_df.index, eq_df['equity'], color=s['color'], linestyle=s['ls'],
                     linewidth=s['lw'], alpha=s['alpha'], label=label)

    ax1.set_ylabel('Equity (log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.set_title('6 Coins - OLD vs NEW × SHORT / LONG / COMBINED', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # 드로다운 (COMBINED만)
    ax2 = axes[1]
    for tag, perf, col, ls in [('OLD', old_results['combined'], '#2196F3', '--'),
                                ('NEW', new_results['combined'], '#2196F3', '-')]:
        eq_df = build_equity_df(perf)
        peak = eq_df['equity'].cummax()
        dd = (eq_df['equity'] - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, color=col, alpha=0.1 if ls == '--' else 0.2)
        ax2.plot(dd.index, dd.values, color=col, linestyle=ls, linewidth=0.8, alpha=0.7,
                 label=f'{tag} COMBINED DD')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, '6coin_v2_all_equity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1] 에쿼티 6종: {path}")

    # ── 요약 대시보드 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    modes = ['SHORT only', 'LONG only', 'COMBINED']
    x = np.arange(3)
    w = 0.3

    # CAGR
    ax = axes[0]
    old_vals = [old_results['short']['cagr'], old_results['long']['cagr'], old_results['combined']['cagr']]
    new_vals = [new_results['short']['cagr'], new_results['long']['cagr'], new_results['combined']['cagr']]
    b1 = ax.bar(x - w/2, old_vals, w, label='OLD', color='#90CAF9', alpha=0.9)
    b2 = ax.bar(x + w/2, new_vals, w, label='NEW', color='#FF8A65', alpha=0.9)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 5:
                ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 10),
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=10)
    ax.set_title('CAGR (%)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # MDD
    ax = axes[1]
    old_vals = [abs(old_results['short']['mdd']), abs(old_results['long']['mdd']), abs(old_results['combined']['mdd'])]
    new_vals = [abs(new_results['short']['mdd']), abs(new_results['long']['mdd']), abs(new_results['combined']['mdd'])]
    b1 = ax.bar(x - w/2, old_vals, w, label='OLD', color='#90CAF9', alpha=0.9)
    b2 = ax.bar(x + w/2, new_vals, w, label='NEW', color='#FF8A65', alpha=0.9)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                    f'-{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=10)
    ax.set_title('Max Drawdown (%)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Sharpe
    ax = axes[2]
    old_vals = [old_results['short']['sharpe'], old_results['long']['sharpe'], old_results['combined']['sharpe']]
    new_vals = [new_results['short']['sharpe'], new_results['long']['sharpe'], new_results['combined']['sharpe']]
    b1 = ax.bar(x - w/2, old_vals, w, label='OLD', color='#90CAF9', alpha=0.9)
    b2 = ax.bar(x + w/2, new_vals, w, label='NEW', color='#FF8A65', alpha=0.9)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=10)
    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('6 Coins - OLD vs NEW Parameters × SHORT / LONG / COMBINED ($10K)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path2 = os.path.join(save_dir, '6coin_v2_summary.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 요약 대시보드: {path2}")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 70)
    print("  6코인 롱/숏/롱+숏 효과 분석 v2 (OLD vs NEW)")
    print("  compare_old_vs_new_params.py 동일 엔진")
    print("=" * 70)
    start_time = time.time()

    # 데이터 로드
    print("\n  --- 데이터 로드 ---")
    all_symbols = set(OLD_CONFIGS.keys()) | set(NEW_CONFIGS.keys())
    coin_data_cache = {}
    for symbol in sorted(all_symbols):
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data_cache[symbol] = data
            print(f"  {symbol}: OK ({data['days']}일)")

    # OLD 백테스트
    print("\n  --- OLD 파라미터 (롱우선 고정) ---")
    old_results = {}
    for mode_name in ['short', 'long', 'combined']:
        perf = run_portfolio(OLD_CONFIGS, coin_data_cache, mode=mode_name)
        old_results[mode_name] = perf
        if perf:
            print(f"  OLD {mode_name:>10}: CAGR={perf['cagr']:>8.1f}%, MDD={perf['mdd']:>7.1f}%, "
                  f"Sharpe={perf['sharpe']:.3f}, Trades={perf['trades']}")

    # NEW 백테스트
    print("\n  --- NEW 파라미터 (코인별 우선순위) ---")
    new_results = {}
    for mode_name in ['short', 'long', 'combined']:
        perf = run_portfolio(NEW_CONFIGS, coin_data_cache, mode=mode_name)
        new_results[mode_name] = perf
        if perf:
            print(f"  NEW {mode_name:>10}: CAGR={perf['cagr']:>8.1f}%, MDD={perf['mdd']:>7.1f}%, "
                  f"Sharpe={perf['sharpe']:.3f}, Trades={perf['trades']}")

    # 결과 비교표
    print("\n" + "=" * 70)
    print("  결과 비교 (초기 자본: $10,000)")
    print("=" * 70)
    print(f"\n  {'':>12} {'OLD SHORT':>12} {'OLD LONG':>12} {'OLD COMB':>12} │ {'NEW SHORT':>12} {'NEW LONG':>12} {'NEW COMB':>12}")
    print(f"  {'-'*87}")

    os_perf = old_results['short']
    ol_perf = old_results['long']
    oc_perf = old_results['combined']
    ns_perf = new_results['short']
    nl_perf = new_results['long']
    nc_perf = new_results['combined']

    print(f"  {'CAGR':>12} {os_perf['cagr']:>11.1f}% {ol_perf['cagr']:>11.1f}% {oc_perf['cagr']:>11.1f}% │ "
          f"{ns_perf['cagr']:>11.1f}% {nl_perf['cagr']:>11.1f}% {nc_perf['cagr']:>11.1f}%")
    print(f"  {'MDD':>12} {os_perf['mdd']:>11.1f}% {ol_perf['mdd']:>11.1f}% {oc_perf['mdd']:>11.1f}% │ "
          f"{ns_perf['mdd']:>11.1f}% {nl_perf['mdd']:>11.1f}% {nc_perf['mdd']:>11.1f}%")
    print(f"  {'Sharpe':>12} {os_perf['sharpe']:>12.3f} {ol_perf['sharpe']:>12.3f} {oc_perf['sharpe']:>12.3f} │ "
          f"{ns_perf['sharpe']:>12.3f} {nl_perf['sharpe']:>12.3f} {nc_perf['sharpe']:>12.3f}")
    print(f"  {'최종자산':>12} ${os_perf['final_equity']:>10,.0f} ${ol_perf['final_equity']:>10,.0f} ${oc_perf['final_equity']:>10,.0f} │ "
          f"${ns_perf['final_equity']:>10,.0f} ${nl_perf['final_equity']:>10,.0f} ${nc_perf['final_equity']:>10,.0f}")
    print(f"  {'거래수':>12} {os_perf['trades']:>12,} {ol_perf['trades']:>12,} {oc_perf['trades']:>12,} │ "
          f"{ns_perf['trades']:>12,} {nl_perf['trades']:>12,} {nc_perf['trades']:>12,}")

    # 시너지 분석
    print(f"\n  --- 시너지 효과 ---")
    old_synergy = oc_perf['cagr'] - max(os_perf['cagr'], ol_perf['cagr'])
    new_synergy = nc_perf['cagr'] - max(ns_perf['cagr'], nl_perf['cagr'])
    print(f"  OLD: COMBINED({oc_perf['cagr']:.0f}%) - max(S:{os_perf['cagr']:.0f}%, L:{ol_perf['cagr']:.0f}%) = {old_synergy:+.0f}%p 시너지")
    print(f"  NEW: COMBINED({nc_perf['cagr']:.0f}%) - max(S:{ns_perf['cagr']:.0f}%, L:{nl_perf['cagr']:.0f}%) = {new_synergy:+.0f}%p 시너지")

    # MDD 개선
    old_mdd_benefit = oc_perf['mdd'] - ol_perf['mdd']
    new_mdd_benefit = nc_perf['mdd'] - nl_perf['mdd']
    print(f"\n  --- MDD 변화 (COMBINED vs LONG only) ---")
    print(f"  OLD: {ol_perf['mdd']:.1f}% → {oc_perf['mdd']:.1f}% ({old_mdd_benefit:+.1f}%p)")
    print(f"  NEW: {nl_perf['mdd']:.1f}% → {nc_perf['mdd']:.1f}% ({new_mdd_benefit:+.1f}%p)")

    # 차트
    print("\n  --- 차트 생성 ---")
    plot_6way_charts(old_results, new_results, SAVE_DIR)

    elapsed = time.time() - start_time
    print(f"\n  완료! ({elapsed:.0f}초)")


if __name__ == '__main__':
    main()
