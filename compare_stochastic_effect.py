"""
================================================================================
스토캐스틱 효과 분석: MA only vs MA+Stochastic
================================================================================
- 6코인 NEW 파라미터: SHORT / LONG / COMBINED × MA only / MA+Stoch
- 288코인 HYBRID 파라미터: SHORT / LONG / COMBINED × MA only / MA+Stoch
- 동일 엔진, 동일 비용 모델
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

from backtest_bots import (
    prepare_coin_data, prepare_signals,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H,
    calculate_portfolio_performance,
)

SAVE_DIR = os.path.expanduser("~/Downloads")
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5


# ==========================================
# 6코인 NEW 파라미터
# ==========================================
SIX_COIN_CONFIGS = {
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


def load_288_configs():
    bot_path = os.path.join(SAVE_DIR, 'binance_bot-5.py')
    with open(bot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    ns = {}
    s = content.index('SHORT_TRADING_CONFIGS = [')
    e = content.index(']', s) + 1
    exec(content[s:e], {}, ns)
    short_configs = ns['SHORT_TRADING_CONFIGS']

    ns2 = {}
    s2 = content.index('LONG_TRADING_CONFIGS = [')
    e2 = content.index(']', s2) + 1
    exec(content[s2:e2], {}, ns2)
    long_configs = ns2['LONG_TRADING_CONFIGS']

    ns3 = {}
    s3 = content.index('COIN_PRIORITY = {')
    e3 = content.index('}', s3) + 1
    exec(content[s3:e3], {}, ns3)
    priority = ns3['COIN_PRIORITY']

    # 통합 config dict 생성
    configs = {}
    short_d = {c['symbol']: c for c in short_configs}
    long_d = {c['symbol']: c for c in long_configs}
    for sym in set(short_d.keys()) | set(long_d.keys()):
        sc = short_d.get(sym, {})
        lc = long_d.get(sym, {})
        if not sc or not lc:
            continue
        configs[sym] = {
            'priority': priority.get(sym, 'short'),
            'short_ma': sc.get('ma_period', 100),
            'short_sk': sc.get('stoch_k_period', 14),
            'short_sks': sc.get('stoch_k_smooth', 3),
            'short_sd': sc.get('stoch_d_period', 3),
            'short_lev': sc.get('leverage', 1),
            'long_ma': lc.get('long_ma', 100),
            'long_sk': lc.get('long_sk', 14),
            'long_sks': lc.get('long_sks', 3),
            'long_sd': lc.get('long_sd', 3),
            'long_lev': lc.get('long_lev', 1),
        }
    return configs


# ==========================================
# 포트폴리오 백테스트 (stoch_enabled 옵션 추가)
# ==========================================
def run_portfolio(configs_dict, coin_data_cache, mode='combined',
                  stoch_enabled=True, initial_capital=10000):
    """
    mode: 'combined', 'short', 'long'
    stoch_enabled: True = MA+Stoch, False = MA only
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

            # ★ 핵심 차이: stoch_enabled에 따라 조건 변경
            if stoch_enabled:
                long_signal = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
                short_signal = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
            else:
                long_signal = (op > prev['ma_long'])
                short_signal = (op < prev['ma_short'])

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
            else:
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


def plot_stoch_effect(results, title_prefix, save_prefix, save_dir):
    """
    results: dict with keys like 'ma_short', 'ma_long', 'ma_combined',
             'stoch_short', 'stoch_long', 'stoch_combined'
    """
    # ── 1. 에쿼티 6종 ──
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={'hspace': 0.05})

    styles = {
        'ma_short':       {'color': '#E91E63', 'ls': '--', 'lw': 1.0, 'alpha': 0.6},
        'ma_long':        {'color': '#4CAF50', 'ls': '--', 'lw': 1.0, 'alpha': 0.6},
        'ma_combined':    {'color': '#2196F3', 'ls': '--', 'lw': 1.5, 'alpha': 0.7},
        'stoch_short':    {'color': '#E91E63', 'ls': '-',  'lw': 1.2, 'alpha': 0.9},
        'stoch_long':     {'color': '#4CAF50', 'ls': '-',  'lw': 1.2, 'alpha': 0.9},
        'stoch_combined': {'color': '#2196F3', 'ls': '-',  'lw': 2.0, 'alpha': 0.95},
    }

    labels_map = {
        'ma_short': 'MA only SHORT',    'ma_long': 'MA only LONG',    'ma_combined': 'MA only COMBINED',
        'stoch_short': 'MA+Stoch SHORT', 'stoch_long': 'MA+Stoch LONG', 'stoch_combined': 'MA+Stoch COMBINED',
    }

    ax1 = axes[0]
    for key in ['ma_short', 'ma_long', 'ma_combined', 'stoch_short', 'stoch_long', 'stoch_combined']:
        perf = results[key]
        eq_df = build_equity_df(perf)
        s = styles[key]
        label = f'{labels_map[key]} (CAGR={perf["cagr"]:.0f}%)'
        ax1.semilogy(eq_df.index, eq_df['equity'], color=s['color'], linestyle=s['ls'],
                     linewidth=s['lw'], alpha=s['alpha'], label=label)

    ax1.set_ylabel('Equity (log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.set_title(f'{title_prefix} - MA only vs MA+Stochastic Effect', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2 = axes[1]
    for key, ls_style in [('ma_combined', '--'), ('stoch_combined', '-')]:
        eq_df = build_equity_df(results[key])
        peak = eq_df['equity'].cummax()
        dd = (eq_df['equity'] - peak) / peak * 100
        lbl = 'MA only' if 'ma_' in key else 'MA+Stoch'
        ax2.fill_between(dd.index, dd.values, 0, color='#2196F3', alpha=0.1 if ls_style == '--' else 0.2)
        ax2.plot(dd.index, dd.values, color='#2196F3', linestyle=ls_style, linewidth=0.8, alpha=0.7,
                 label=f'{lbl} COMBINED DD')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path1 = os.path.join(save_dir, f'{save_prefix}_equity.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1] 에쿼티: {path1}")

    # ── 2. 요약 대시보드 ──
    fig, axes_dash = plt.subplots(1, 4, figsize=(22, 6))
    modes = ['SHORT', 'LONG', 'COMBINED']
    x = np.arange(3)
    w = 0.3

    ma_perfs = [results['ma_short'], results['ma_long'], results['ma_combined']]
    st_perfs = [results['stoch_short'], results['stoch_long'], results['stoch_combined']]

    # CAGR
    ax = axes_dash[0]
    ma_vals = [p['cagr'] for p in ma_perfs]
    st_vals = [p['cagr'] for p in st_perfs]
    b1 = ax.bar(x - w/2, ma_vals, w, label='MA only', color='#BBDEFB', alpha=0.9, edgecolor='#1565C0', linewidth=0.5)
    b2 = ax.bar(x + w/2, st_vals, w, label='MA+Stoch', color='#FF8A65', alpha=0.9, edgecolor='#BF360C', linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 5:
                ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 10),
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_title('CAGR (%)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # MDD
    ax = axes_dash[1]
    ma_vals = [abs(p['mdd']) for p in ma_perfs]
    st_vals = [abs(p['mdd']) for p in st_perfs]
    b1 = ax.bar(x - w/2, ma_vals, w, label='MA only', color='#BBDEFB', alpha=0.9, edgecolor='#1565C0', linewidth=0.5)
    b2 = ax.bar(x + w/2, st_vals, w, label='MA+Stoch', color='#FF8A65', alpha=0.9, edgecolor='#BF360C', linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                    f'-{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_title('Max Drawdown (%)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # Sharpe
    ax = axes_dash[2]
    ma_vals = [p['sharpe'] for p in ma_perfs]
    st_vals = [p['sharpe'] for p in st_perfs]
    b1 = ax.bar(x - w/2, ma_vals, w, label='MA only', color='#BBDEFB', alpha=0.9, edgecolor='#1565C0', linewidth=0.5)
    b2 = ax.bar(x + w/2, st_vals, w, label='MA+Stoch', color='#FF8A65', alpha=0.9, edgecolor='#BF360C', linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.03,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # Trades
    ax = axes_dash[3]
    ma_vals = [p['trades'] for p in ma_perfs]
    st_vals = [p['trades'] for p in st_perfs]
    b1 = ax.bar(x - w/2, ma_vals, w, label='MA only', color='#BBDEFB', alpha=0.9, edgecolor='#1565C0', linewidth=0.5)
    b2 = ax.bar(x + w/2, st_vals, w, label='MA+Stoch', color='#FF8A65', alpha=0.9, edgecolor='#BF360C', linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + max(h*0.02, 20),
                    f'{h:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_title('Total Trades', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{title_prefix} - Stochastic Filter Effect', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path2 = os.path.join(save_dir, f'{save_prefix}_summary.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 요약 대시보드: {path2}")


def run_6way(configs, coin_data_cache, capital, label):
    """MA only × 3 + MA+Stoch × 3 = 6개 백테스트"""
    results = {}
    for stoch_name, stoch_flag in [('ma', False), ('stoch', True)]:
        for mode_name in ['short', 'long', 'combined']:
            key = f'{stoch_name}_{mode_name}'
            stoch_label = 'MA+Stoch' if stoch_flag else 'MA only'
            perf = run_portfolio(configs, coin_data_cache, mode=mode_name,
                                stoch_enabled=stoch_flag, initial_capital=capital)
            results[key] = perf
            if perf:
                print(f"  [{label}] {stoch_label:>9} {mode_name:>10}: "
                      f"CAGR={perf['cagr']:>8.1f}%, MDD={perf['mdd']:>7.1f}%, "
                      f"Sharpe={perf['sharpe']:.3f}, Trades={perf['trades']:,}")
    return results


def print_comparison_table(results, label):
    """비교표 출력"""
    print(f"\n  {'':>14} {'MA only':>40} │ {'MA + Stochastic':>40}")
    print(f"  {'':>14} {'SHORT':>12} {'LONG':>12} {'COMBINED':>12} │ {'SHORT':>12} {'LONG':>12} {'COMBINED':>12}")
    print(f"  {'-'*95}")

    ms = results['ma_short']
    ml = results['ma_long']
    mc = results['ma_combined']
    ss = results['stoch_short']
    sl = results['stoch_long']
    sc = results['stoch_combined']

    print(f"  {'CAGR':>14} {ms['cagr']:>11.1f}% {ml['cagr']:>11.1f}% {mc['cagr']:>11.1f}% │ "
          f"{ss['cagr']:>11.1f}% {sl['cagr']:>11.1f}% {sc['cagr']:>11.1f}%")
    print(f"  {'MDD':>14} {ms['mdd']:>11.1f}% {ml['mdd']:>11.1f}% {mc['mdd']:>11.1f}% │ "
          f"{ss['mdd']:>11.1f}% {sl['mdd']:>11.1f}% {sc['mdd']:>11.1f}%")
    print(f"  {'Sharpe':>14} {ms['sharpe']:>12.3f} {ml['sharpe']:>12.3f} {mc['sharpe']:>12.3f} │ "
          f"{ss['sharpe']:>12.3f} {sl['sharpe']:>12.3f} {sc['sharpe']:>12.3f}")
    print(f"  {'Trades':>14} {ms['trades']:>12,} {ml['trades']:>12,} {mc['trades']:>12,} │ "
          f"{ss['trades']:>12,} {sl['trades']:>12,} {sc['trades']:>12,}")
    print(f"  {'최종자산':>14} ${ms['final_equity']:>10,.0f} ${ml['final_equity']:>10,.0f} ${mc['final_equity']:>10,.0f} │ "
          f"${ss['final_equity']:>10,.0f} ${sl['final_equity']:>10,.0f} ${sc['final_equity']:>10,.0f}")

    # 스토캐스틱 효과 분석
    print(f"\n  --- 스토캐스틱 필터 효과 ({label}) ---")
    for mode_name in ['short', 'long', 'combined']:
        ma_p = results[f'ma_{mode_name}']
        st_p = results[f'stoch_{mode_name}']
        cagr_diff = st_p['cagr'] - ma_p['cagr']
        mdd_diff = st_p['mdd'] - ma_p['mdd']
        sharpe_diff = st_p['sharpe'] - ma_p['sharpe']
        trade_ratio = st_p['trades'] / ma_p['trades'] * 100 if ma_p['trades'] > 0 else 0
        print(f"  {mode_name:>10}: CAGR {cagr_diff:>+8.1f}%p, MDD {mdd_diff:>+7.1f}%p, "
              f"Sharpe {sharpe_diff:>+6.3f}, 거래수 {trade_ratio:.0f}% ({ma_p['trades']:,}→{st_p['trades']:,})")


def main():
    print("=" * 70)
    print("  스토캐스틱 효과 분석: MA only vs MA+Stochastic")
    print("=" * 70)
    start_time = time.time()

    # ── Part 1: 6코인 ──
    print("\n" + "=" * 70)
    print("  PART 1: 6코인 NEW 파라미터 ($10K)")
    print("=" * 70)
    coin_data_6 = {}
    for sym in sorted(SIX_COIN_CONFIGS.keys()):
        data = prepare_coin_data(sym, silent=True)
        if data:
            coin_data_6[sym] = data
    print(f"  로드: {len(coin_data_6)}코인")

    results_6 = run_6way(SIX_COIN_CONFIGS, coin_data_6, capital=10000, label="6coin")
    print_comparison_table(results_6, "6코인")
    plot_stoch_effect(results_6, '6 Coins NEW', 'stoch_effect_6coin', SAVE_DIR)

    # ── Part 2: 288코인 ──
    print("\n" + "=" * 70)
    print("  PART 2: HYBRID 288코인 ($100K)")
    print("=" * 70)
    configs_288 = load_288_configs()
    print(f"  설정: {len(configs_288)}코인")

    all_symbols = set(configs_288.keys())
    coin_data_288 = {}
    for idx, sym in enumerate(sorted(all_symbols), 1):
        if idx % 50 == 0:
            print(f"  진행: {idx}/{len(all_symbols)}...")
        data = prepare_coin_data(sym, silent=True)
        if data:
            coin_data_288[sym] = data
    print(f"  로드: {len(coin_data_288)}코인")

    results_288 = run_6way(configs_288, coin_data_288, capital=100000, label="288coin")
    print_comparison_table(results_288, "288코인")
    plot_stoch_effect(results_288, 'HYBRID 288 Coins', 'stoch_effect_288coin', SAVE_DIR)

    elapsed = time.time() - start_time
    print(f"\n  전체 완료! ({elapsed/60:.1f}분)")


if __name__ == '__main__':
    main()
