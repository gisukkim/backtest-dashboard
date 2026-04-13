"""
================================================================================
6코인 포트폴리오 롱/숏/롱+숏 효과 분석 백테스트
================================================================================
BTC, ETH, XRP, SOL, DOGE, ADA (BITGET_CONFIGS)
- SHORT only: 숏 포지션만 운영
- LONG only: 롱 포지션만 운영
- COMBINED: 롱우선 숏+롱 동시 운영 (기존 bitget 로직)
- 4H 시가 기반, 비용 반영
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
    prepare_coin_data, prepare_signals, calculate_stochastic,
    calculate_portfolio_performance, BITGET_CONFIGS,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H, BITGET_CAPITAL
)

SAVE_DIR = os.path.expanduser("~/Downloads")
CAPITAL = BITGET_CAPITAL  # $10,000


def prepare_6coin_signals(coin_data_cache, end_date):
    """6코인 롱+숏 신호 준비 (BITGET_CONFIGS 기반)"""
    coin_signals = {}
    for config in BITGET_CONFIGS:
        symbol = config['symbol']
        data = coin_data_cache.get(symbol)
        if data is None:
            continue
        df_long = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                                  config['long_ma'], config['long_sk'], config['long_sks'], config['long_sd'],
                                  use_1h=False)
        df_short = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                                   config['short_ma'], config['short_sk'], config['short_sks'], config['short_sd'],
                                   use_1h=False)
        df_bt = df_long[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                          'prev_slow_k', 'prev_slow_d']].copy()
        df_bt.rename(columns={'ma': 'ma_long', 'prev_slow_k': 'long_k', 'prev_slow_d': 'long_d'}, inplace=True)
        df_bt['ma_short'] = df_short['ma'].values
        df_bt['short_k'] = df_short['prev_slow_k'].values
        df_bt['short_d'] = df_short['prev_slow_d'].values
        df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d', 'ma_short', 'short_k', 'short_d'])
        if end_date is not None:
            df_bt = df_bt[df_bt['timestamp'] <= end_date]
        df_bt = df_bt.set_index('timestamp').sort_index()
        if len(df_bt) >= 100:
            coin_signals[symbol] = {
                'df': df_bt,
                'long_lev': config['long_lev'],
                'short_lev': config['short_lev']
            }
    return coin_signals


def run_short_only_6coin(coin_signals, label="SHORT"):
    """숏 포지션만 운영"""
    funding_per_bar = FUNDING_RATE_8H * 0.5
    all_ts = set()
    for s in coin_signals.values():
        all_ts.update(s['df'].index.tolist())
    timeline = sorted(all_ts)
    print(f"  [{label}] {len(coin_signals)}코인, {len(timeline)}봉 ({timeline[0].date()} ~ {timeline[-1].date()})")

    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0

    for ts in timeline:
        active = [s for s, d in coin_signals.items() if ts in d['df'].index]
        n_active = len(active)
        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        for symbol in active:
            s_data = coin_signals[symbol]
            df = s_data['df']
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
            curr = df.loc[ts]
            prev = df.iloc[curr_idx - 1]
            op = curr['open']
            short_lev = s_data['short_lev']

            short_signal = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
            pos = positions.get(symbol)

            if short_signal:
                if pos is None:
                    total_eq = cash + sum(p['margin'] for p in positions.values())
                    alloc = min(total_eq / n_active, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * short_lev
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'short', 'entry_price': op, 'margin': margin,
                                                 'cum_funding': 0, 'lev': short_lev}
                            total_trades += 1
            else:
                if pos:
                    pr = op / pos['entry_price'] - 1
                    pnl = pos['margin'] * (-pr) * pos['lev']
                    settled = pos['margin'] + pnl - pos['cum_funding']
                    exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                    settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                    cash += max(settled, 0)
                    del positions[symbol]
                    total_trades += 1

            # 펀딩 + 청산
            pos = positions.get(symbol)
            if pos and ts in df.index:
                pos['cum_funding'] += pos['margin'] * pos['lev'] * funding_per_bar
                if pos['lev'] > 0 and curr['high'] >= pos['entry_price'] * (1 + 1 / pos['lev']):
                    del positions[symbol]
                    total_trades += 1

        unrealized = 0
        for sym, pos in positions.items():
            df = coin_signals[sym]['df']
            if ts in df.index:
                close_p = df.loc[ts, 'close']
                val = pos['margin'] + pos['margin'] * (-(close_p / pos['entry_price'] - 1)) * pos['lev'] - pos['cum_funding']
                unrealized += max(val, 0)
            else:
                unrealized += pos['margin']
        equity_curve.append(cash + unrealized)
        eq_timestamps.append(ts)

    # 마지막 청산
    for sym, pos in list(positions.items()):
        df = coin_signals[sym]['df']
        last_p = df.iloc[-1]['close']
        pnl = pos['margin'] * (-(last_p / pos['entry_price'] - 1)) * pos['lev']
        settled = pos['margin'] + pnl - pos['cum_funding']
        exit_not = abs(pos['margin'] * pos['lev'] * (last_p / pos['entry_price']))
        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(settled, 0)

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf:
        perf['trades'] = total_trades
        perf['short_trades'] = total_trades
        perf['long_trades'] = 0
        perf['equity_curve'] = equity_curve
        perf['timestamps'] = eq_timestamps
    return perf


def run_long_only_6coin(coin_signals, label="LONG"):
    """롱 포지션만 운영"""
    funding_per_bar = FUNDING_RATE_8H * 0.5
    all_ts = set()
    for s in coin_signals.values():
        all_ts.update(s['df'].index.tolist())
    timeline = sorted(all_ts)
    print(f"  [{label}] {len(coin_signals)}코인, {len(timeline)}봉 ({timeline[0].date()} ~ {timeline[-1].date()})")

    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0

    for ts in timeline:
        active = [s for s, d in coin_signals.items() if ts in d['df'].index]
        n_active = len(active)
        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        for symbol in active:
            s_data = coin_signals[symbol]
            df = s_data['df']
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
            curr = df.loc[ts]
            prev = df.iloc[curr_idx - 1]
            op = curr['open']
            long_lev = s_data['long_lev']

            long_signal = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
            pos = positions.get(symbol)

            if long_signal:
                if pos is None:
                    total_eq = cash + sum(p['margin'] for p in positions.values())
                    alloc = min(total_eq / n_active, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * long_lev
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'long', 'entry_price': op, 'margin': margin,
                                                 'cum_funding': 0, 'lev': long_lev}
                            total_trades += 1
            else:
                if pos:
                    pr = op / pos['entry_price'] - 1
                    pnl = pos['margin'] * pr * pos['lev']
                    settled = pos['margin'] + pnl - pos['cum_funding']
                    exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                    settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                    cash += max(settled, 0)
                    del positions[symbol]
                    total_trades += 1

            # 펀딩 + 청산
            pos = positions.get(symbol)
            if pos and ts in df.index:
                pos['cum_funding'] += pos['margin'] * pos['lev'] * funding_per_bar
                if pos['lev'] > 0 and curr['low'] <= pos['entry_price'] * (1 - 1 / pos['lev']):
                    del positions[symbol]
                    total_trades += 1

        unrealized = 0
        for sym, pos in positions.items():
            df = coin_signals[sym]['df']
            if ts in df.index:
                close_p = df.loc[ts, 'close']
                val = pos['margin'] + pos['margin'] * (close_p / pos['entry_price'] - 1) * pos['lev'] - pos['cum_funding']
                unrealized += max(val, 0)
            else:
                unrealized += pos['margin']
        equity_curve.append(cash + unrealized)
        eq_timestamps.append(ts)

    # 마지막 청산
    for sym, pos in list(positions.items()):
        df = coin_signals[sym]['df']
        last_p = df.iloc[-1]['close']
        pnl = pos['margin'] * (last_p / pos['entry_price'] - 1) * pos['lev']
        settled = pos['margin'] + pnl - pos['cum_funding']
        exit_not = abs(pos['margin'] * pos['lev'] * (last_p / pos['entry_price']))
        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(settled, 0)

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf:
        perf['trades'] = total_trades
        perf['short_trades'] = 0
        perf['long_trades'] = total_trades
        perf['equity_curve'] = equity_curve
        perf['timestamps'] = eq_timestamps
    return perf


def run_combined_6coin(coin_signals, label="COMBINED"):
    """롱우선 숏+롱 동시 운영 (기존 bitget 로직 그대로)"""
    funding_per_bar = FUNDING_RATE_8H * 0.5
    all_ts = set()
    for s in coin_signals.values():
        all_ts.update(s['df'].index.tolist())
    timeline = sorted(all_ts)
    print(f"  [{label}] {len(coin_signals)}코인, {len(timeline)}봉")

    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0
    long_trades = 0
    short_trades = 0

    for ts in timeline:
        active = [s for s, d in coin_signals.items() if ts in d['df'].index]
        n_active = len(active)
        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        for symbol in active:
            s_data = coin_signals[symbol]
            df = s_data['df']
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
            curr = df.loc[ts]
            prev = df.iloc[curr_idx - 1]
            op = curr['open']
            long_lev = s_data['long_lev']
            short_lev = s_data['short_lev']

            long_signal = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
            short_signal = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
            pos = positions.get(symbol)

            # 롱 우선 로직 (기존 bitget 로직)
            if long_signal:
                # 숏 청산 후 롱 진입
                if pos and pos['side'] == 'short':
                    pr = op / pos['entry_price'] - 1
                    pnl = pos['margin'] * (-pr) * pos['lev']
                    settled = pos['margin'] + pnl - pos['cum_funding']
                    exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                    settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                    cash += max(settled, 0)
                    del positions[symbol]
                    total_trades += 1
                    pos = None

                if pos is None:
                    total_eq = cash + sum(p['margin'] for p in positions.values())
                    alloc = min(total_eq / n_active, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * long_lev
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'long', 'entry_price': op, 'margin': margin,
                                                 'cum_funding': 0, 'lev': long_lev}
                            total_trades += 1
                            long_trades += 1

            elif short_signal and (pos is None or pos.get('side') != 'long'):
                # 숏 진입 (롱 없을 때만)
                if pos is None:
                    total_eq = cash + sum(p['margin'] for p in positions.values())
                    alloc = min(total_eq / n_active, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * short_lev
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'short', 'entry_price': op, 'margin': margin,
                                                 'cum_funding': 0, 'lev': short_lev}
                            total_trades += 1
                            short_trades += 1
            else:
                # 신호 없음 → 기존 포지션 청산
                if pos and pos['side'] == 'long':
                    pr = op / pos['entry_price'] - 1
                    pnl = pos['margin'] * pr * pos['lev']
                    settled = pos['margin'] + pnl - pos['cum_funding']
                    exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                    settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                    cash += max(settled, 0)
                    del positions[symbol]
                    total_trades += 1
                elif pos and pos['side'] == 'short':
                    pr = op / pos['entry_price'] - 1
                    pnl = pos['margin'] * (-pr) * pos['lev']
                    settled = pos['margin'] + pnl - pos['cum_funding']
                    exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                    settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                    cash += max(settled, 0)
                    del positions[symbol]
                    total_trades += 1

            # 펀딩 + 청산
            pos = positions.get(symbol)
            if pos and ts in df.index:
                pos['cum_funding'] += pos['margin'] * pos['lev'] * funding_per_bar
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
                close_p = df.loc[ts, 'close']
                if pos['side'] == 'long':
                    val = pos['margin'] + pos['margin'] * (close_p / pos['entry_price'] - 1) * pos['lev'] - pos['cum_funding']
                else:
                    val = pos['margin'] + pos['margin'] * (-(close_p / pos['entry_price'] - 1)) * pos['lev'] - pos['cum_funding']
                unrealized += max(val, 0)
            else:
                unrealized += pos['margin']
        equity_curve.append(cash + unrealized)
        eq_timestamps.append(ts)

    # 마지막 청산
    for sym, pos in list(positions.items()):
        df = coin_signals[sym]['df']
        last_p = df.iloc[-1]['close']
        if pos['side'] == 'long':
            pnl = pos['margin'] * (last_p / pos['entry_price'] - 1) * pos['lev']
        else:
            pnl = pos['margin'] * (-(last_p / pos['entry_price'] - 1)) * pos['lev']
        settled = pos['margin'] + pnl - pos['cum_funding']
        exit_not = abs(pos['margin'] * pos['lev'] * (last_p / pos['entry_price']))
        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(settled, 0)

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf:
        perf['trades'] = total_trades
        perf['short_trades'] = short_trades
        perf['long_trades'] = long_trades
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


def plot_all_charts(short_perf, long_perf, combined_perf, save_dir):
    short_df = build_equity_df(short_perf)
    long_df = build_equity_df(long_perf)
    comb_df = build_equity_df(combined_perf)

    colors = {'short': '#E91E63', 'long': '#4CAF50', 'combined': '#2196F3'}

    # ── 1. 에쿼티 + 드로다운 ──
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={'hspace': 0.05})
    ax1 = axes[0]
    ax1.semilogy(short_df.index, short_df['equity'], color=colors['short'], linewidth=1.3,
                 label=f'SHORT only (CAGR={short_perf["cagr"]:.0f}%)', alpha=0.9)
    ax1.semilogy(long_df.index, long_df['equity'], color=colors['long'], linewidth=1.3,
                 label=f'LONG only (CAGR={long_perf["cagr"]:.0f}%)', alpha=0.9)
    ax1.semilogy(comb_df.index, comb_df['equity'], color=colors['combined'], linewidth=1.8,
                 label=f'COMBINED (CAGR={combined_perf["cagr"]:.0f}%)', alpha=0.95)
    ax1.set_ylabel('Portfolio Equity (log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.set_title('6 Coins (BTC ETH XRP SOL DOGE ADA) - SHORT / LONG / COMBINED Effect',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2 = axes[1]
    for lbl, eq_df, col in [('SHORT', short_df, colors['short']),
                              ('LONG', long_df, colors['long']),
                              ('COMBINED', comb_df, colors['combined'])]:
        peak = eq_df['equity'].cummax()
        dd = (eq_df['equity'] - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, color=col, alpha=0.15)
        ax2.plot(dd.index, dd.values, color=col, linewidth=0.6, alpha=0.7, label=f'{lbl} DD')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path1 = os.path.join(save_dir, '6coin_long_short_effect_equity.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1] 에쿼티+드로다운: {path1}")

    # ── 2. 연간 수익률 비교 ──
    def get_yearly_returns(eq_df):
        yearly = eq_df['equity'].resample('YE').last().dropna()
        rets = yearly.pct_change().dropna() * 100
        first_ret = (yearly.iloc[0] / eq_df['equity'].iloc[0] - 1) * 100
        rets = pd.concat([pd.Series({yearly.index[0]: first_ret}), rets])
        return rets

    short_yr = get_yearly_returns(short_df)
    long_yr = get_yearly_returns(long_df)
    comb_yr = get_yearly_returns(comb_df)
    all_years = sorted(set(short_yr.index.year) | set(long_yr.index.year) | set(comb_yr.index.year))

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_years))
    w = 0.25
    s_vals = [short_yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]
    l_vals = [long_yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]
    c_vals = [comb_yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]

    bars_s = ax.bar(x - w, s_vals, w, label='SHORT only', color=colors['short'], alpha=0.8)
    bars_l = ax.bar(x, l_vals, w, label='LONG only', color=colors['long'], alpha=0.8)
    bars_c = ax.bar(x + w, c_vals, w, label='COMBINED', color=colors['combined'], alpha=0.8)

    for bars in [bars_s, bars_l, bars_c]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 5:
                ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 5),
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(all_years, fontsize=11)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.set_title('6 Coins - Annual Returns: SHORT / LONG / COMBINED', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    path2 = os.path.join(save_dir, '6coin_long_short_effect_yearly.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 연간 수익률: {path2}")

    # ── 3. 요약 대시보드 ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = ['SHORT\nonly', 'LONG\nonly', 'SHORT+\nLONG']
    all_perfs = [short_perf, long_perf, combined_perf]
    cols = [colors['short'], colors['long'], colors['combined']]

    ax = axes[0]
    vals = [p['cagr'] for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(bar.get_height()*0.02, 10),
                f'{val:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('CAGR (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    vals = [abs(p['mdd']) for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'-{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('Max Drawdown (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    vals = [p['sharpe'] for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[3]
    vals = [p['trades'] for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Total Trades', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('6 Coins (BTC ETH XRP SOL DOGE ADA) - Long/Short Effect ($10K)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path3 = os.path.join(save_dir, '6coin_long_short_effect_summary.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3] 요약 대시보드: {path3}")

    # ── 4. 월별 히트맵 3단 ──
    fig, axes_hm = plt.subplots(3, 1, figsize=(16, 14))
    for ax, (lbl, eq_df, col) in zip(axes_hm, [('SHORT only', short_df, colors['short']),
                                                 ('LONG only', long_df, colors['long']),
                                                 ('COMBINED', comb_df, colors['combined'])]):
        monthly = eq_df['equity'].resample('ME').last().dropna()
        rets = monthly.pct_change().dropna() * 100
        ret_df = pd.DataFrame({'return': rets})
        ret_df['year'] = ret_df.index.year
        ret_df['month'] = ret_df.index.month
        pivot = ret_df.pivot_table(values='return', index='year', columns='month', aggfunc='sum')
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_labels[m-1] for m in pivot.columns]
        pivot['Year'] = pivot.sum(axis=1)

        vmax = min(pivot.iloc[:, :-1].max().max(), 200)
        vmin = max(pivot.iloc[:, :-1].min().min(), -80)
        if vmin >= 0:
            vmin = -1
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.imshow(pivot.iloc[:, :-1].values, cmap='RdYlGn', aspect='auto', norm=norm)
        ax.set_xticks(range(len(pivot.columns) - 1))
        ax.set_xticklabels(pivot.columns[:-1], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns) - 1):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ct = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8, color=ct)
        for i, yr in enumerate(pivot['Year']):
            ax.text(len(pivot.columns) - 1.5, i, f' {yr:.0f}%', ha='left', va='center',
                    fontsize=9, fontweight='bold', color='green' if yr > 0 else 'red')

        ax.set_title(f'{lbl} - Monthly Returns (%)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    path4 = os.path.join(save_dir, '6coin_long_short_effect_monthly.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [4] 월별 히트맵: {path4}")


# ============================================================
# 개별 코인 분석
# ============================================================
def run_single_coin_backtest(coin_signals, symbol, mode='combined'):
    """개별 코인 백테스트 (short/long/combined)"""
    if symbol not in coin_signals:
        return None
    funding_per_bar = FUNDING_RATE_8H * 0.5
    s_data = coin_signals[symbol]
    df = s_data['df']
    long_lev = s_data['long_lev']
    short_lev = s_data['short_lev']
    timeline = df.index.tolist()

    cash = CAPITAL
    position = None
    equity_curve = []
    trades = 0

    for t_idx, ts in enumerate(timeline):
        if t_idx == 0:
            equity_curve.append(cash)
            continue
        curr = df.loc[ts]
        prev = df.iloc[t_idx - 1]
        op = curr['open']

        long_signal = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
        short_signal = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])

        if mode == 'short':
            long_signal = False
        elif mode == 'long':
            short_signal = False

        # 롱 우선 로직
        if long_signal:
            if position and position['side'] == 'short':
                pr = op / position['entry_price'] - 1
                pnl = position['margin'] * (-pr) * position['lev']
                settled = position['margin'] + pnl - position['cum_funding']
                exit_not = abs(position['margin'] * position['lev'] * (op / position['entry_price']))
                settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                cash += max(settled, 0)
                position = None
                trades += 1
            if position is None and cash > 1:
                alloc = cash * 0.995
                margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * long_lev
                if margin > 0:
                    cash -= alloc
                    position = {'side': 'long', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': long_lev}
                    trades += 1
        elif short_signal and (position is None or position['side'] != 'long'):
            if position is None and cash > 1:
                alloc = cash * 0.995
                margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * short_lev
                if margin > 0:
                    cash -= alloc
                    position = {'side': 'short', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': short_lev}
                    trades += 1
        else:
            if position:
                pr = op / position['entry_price'] - 1
                if position['side'] == 'long':
                    pnl = position['margin'] * pr * position['lev']
                else:
                    pnl = position['margin'] * (-pr) * position['lev']
                settled = position['margin'] + pnl - position['cum_funding']
                exit_not = abs(position['margin'] * position['lev'] * (op / position['entry_price']))
                settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                cash += max(settled, 0)
                position = None
                trades += 1

        if position:
            position['cum_funding'] += position['margin'] * position['lev'] * funding_per_bar
            if position['side'] == 'long' and position['lev'] > 0:
                if curr['low'] <= position['entry_price'] * (1 - 1 / position['lev']):
                    position = None
                    trades += 1
            elif position and position['side'] == 'short' and position['lev'] > 0:
                if curr['high'] >= position['entry_price'] * (1 + 1 / position['lev']):
                    position = None
                    trades += 1

        if position:
            close_p = curr['close']
            if position['side'] == 'long':
                val = position['margin'] + position['margin'] * (close_p / position['entry_price'] - 1) * position['lev'] - position['cum_funding']
            else:
                val = position['margin'] + position['margin'] * (-(close_p / position['entry_price'] - 1)) * position['lev'] - position['cum_funding']
            equity_curve.append(cash + max(val, 0))
        else:
            equity_curve.append(cash)

    final_eq = equity_curve[-1] if equity_curve else cash
    initial = CAPITAL
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    if years > 0 and final_eq > 0:
        cagr = (final_eq / initial) ** (1 / years) - 1
    else:
        cagr = 0

    peak = pd.Series(equity_curve).cummax()
    dd = (pd.Series(equity_curve) - peak) / peak * 100
    mdd = dd.min()

    return {'cagr': cagr * 100, 'mdd': mdd, 'final_equity': final_eq, 'trades': trades}


def print_coin_table(coin_signals):
    """개별 코인 SHORT/LONG/COMBINED 비교표"""
    print(f"\n  {'코인':<12} {'SHORT CAGR':>12} {'LONG CAGR':>12} {'COMBINED':>12} {'S MDD':>8} {'L MDD':>8} {'C MDD':>8}")
    print(f"  {'-'*74}")

    for config in BITGET_CONFIGS:
        symbol = config['symbol']
        s = run_single_coin_backtest(coin_signals, symbol, 'short')
        l = run_single_coin_backtest(coin_signals, symbol, 'long')
        c = run_single_coin_backtest(coin_signals, symbol, 'combined')
        if s and l and c:
            best = max(s['cagr'], l['cagr'], c['cagr'])
            s_mark = " 🏆" if s['cagr'] == best else ""
            l_mark = " 🏆" if l['cagr'] == best else ""
            c_mark = " 🏆" if c['cagr'] == best else ""
            print(f"  {symbol:<12} {s['cagr']:>10.1f}%{s_mark} {l['cagr']:>10.1f}%{l_mark} {c['cagr']:>10.1f}%{c_mark} "
                  f"{s['mdd']:>7.1f}% {l['mdd']:>7.1f}% {c['mdd']:>7.1f}%")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 70)
    print("  6코인 포트폴리오 롱/숏/롱+숏 효과 분석")
    print("  (BTC, ETH, XRP, SOL, DOGE, ADA)")
    print("=" * 70)
    start_time = time.time()

    # 데이터 로드
    print("\n  --- 데이터 로드 ---")
    coin_data_cache = {}
    for config in BITGET_CONFIGS:
        symbol = config['symbol']
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data_cache[symbol] = data
            print(f"  {symbol}: {data['days']}일")
    print(f"  로드: {len(coin_data_cache)}개 코인")

    all_ends = [d['df_4h']['timestamp'].max() for d in coin_data_cache.values()]
    end_date = min(all_ends)
    print(f"  종료일: {end_date}")

    # 신호 준비
    coin_signals = prepare_6coin_signals(coin_data_cache, end_date)
    print(f"  신호 준비: {len(coin_signals)}개 코인")

    # 개별 코인 분석
    print("\n  --- 개별 코인 SHORT/LONG/COMBINED ---")
    print_coin_table(coin_signals)

    # 포트폴리오 백테스트
    print("\n  --- SHORT only 포트폴리오 ---")
    short_perf = run_short_only_6coin(coin_signals)

    print("\n  --- LONG only 포트폴리오 ---")
    long_perf = run_long_only_6coin(coin_signals)

    print("\n  --- COMBINED (롱우선 숏+롱) 포트폴리오 ---")
    combined_perf = run_combined_6coin(coin_signals)

    if not short_perf or not long_perf or not combined_perf:
        print("  백테스트 실패!")
        return

    # 결과 출력
    print("\n" + "=" * 70)
    print("  포트폴리오 결과 비교 (초기 자본: $10,000)")
    print("=" * 70)
    print(f"  {'항목':<20} {'SHORT only':>15} {'LONG only':>15} {'COMBINED':>15}")
    print(f"  {'-'*65}")
    print(f"  {'CAGR':<20} {short_perf['cagr']:>14.1f}% {long_perf['cagr']:>14.1f}% {combined_perf['cagr']:>14.1f}%")
    print(f"  {'MDD':<20} {short_perf['mdd']:>14.1f}% {long_perf['mdd']:>14.1f}% {combined_perf['mdd']:>14.1f}%")
    print(f"  {'Sharpe':<20} {short_perf['sharpe']:>15.3f} {long_perf['sharpe']:>15.3f} {combined_perf['sharpe']:>15.3f}")
    print(f"  {'Total Return':<20} {short_perf['total_return']:>13.0f}% {long_perf['total_return']:>13.0f}% {combined_perf['total_return']:>13.0f}%")
    print(f"  {'총 거래 수':<20} {short_perf['trades']:>15,} {long_perf['trades']:>15,} {combined_perf['trades']:>15,}")
    print(f"  {'최종 자산':<20} ${short_perf['final_equity']:>13,.0f} ${long_perf['final_equity']:>13,.0f} ${combined_perf['final_equity']:>13,.0f}")

    if 'short_trades' in combined_perf:
        st = combined_perf['short_trades']
        lt = combined_perf['long_trades']
        if st + lt > 0:
            print(f"\n  COMBINED 구성: 숏 {st} ({st/(st+lt)*100:.0f}%) + 롱 {lt} ({lt/(st+lt)*100:.0f}%)")

    synergy = combined_perf['cagr'] - max(short_perf['cagr'], long_perf['cagr'])
    print(f"\n  시너지: COMBINED - max(SHORT,LONG) = {synergy:+.1f}% CAGR")

    # 차트
    print("\n  --- 차트 생성 ---")
    plot_all_charts(short_perf, long_perf, combined_perf, SAVE_DIR)

    elapsed = time.time() - start_time
    print(f"\n  완료! ({elapsed:.0f}초)")


if __name__ == '__main__':
    main()
