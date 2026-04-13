"""
================================================================================
HYBRID 288코인 롱/숏/롱+숏 효과 분석 백테스트
================================================================================
- SHORT only: 숏 포지션만 운영 (288코인)
- LONG only: 롱 포지션만 운영 (288코인)
- COMBINED: 숏+롱 코인별 우선순위 운영 (288코인)
- 4H 시가 기반, 비용 반영, 동적 배분
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
    calculate_portfolio_performance,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H
)

SAVE_DIR = os.path.expanduser("~/Downloads")
CAPITAL = 100000.0


def load_configs_from_file(bot_path):
    with open(bot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    ns = {}
    s = content.index('SHORT_TRADING_CONFIGS = [')
    e = content.index(']', s) + 1
    exec(content[s:e], {}, ns)
    short_configs = ns.get('SHORT_TRADING_CONFIGS', [])

    ns2 = {}
    s2 = content.index('LONG_TRADING_CONFIGS = [')
    e2 = content.index(']', s2) + 1
    exec(content[s2:e2], {}, ns2)
    long_configs = ns2.get('LONG_TRADING_CONFIGS', [])

    priority = {}
    if 'COIN_PRIORITY' in content:
        p_start = content.index('COIN_PRIORITY = {')
        p_end = content.index('}', p_start) + 1
        ns3 = {}
        exec(content[p_start:p_end], {}, ns3)
        priority = ns3.get('COIN_PRIORITY', {})

    return short_configs, long_configs, priority


def prepare_short_signals(short_configs, coin_data_cache, end_date):
    short_signals = {}
    for config in short_configs:
        symbol = config['symbol']
        data = coin_data_cache.get(symbol)
        if data is None:
            continue
        df_sig = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                                 config['ma_period'], config['stoch_k_period'],
                                 config['stoch_k_smooth'], config['stoch_d_period'], use_1h=False)
        df_bt = df_sig.dropna(subset=['ma', 'prev_slow_k', 'prev_slow_d'])
        if end_date is not None:
            df_bt = df_bt[df_bt['timestamp'] <= end_date]
        df_bt = df_bt.set_index('timestamp').sort_index()
        if len(df_bt) >= 100:
            short_signals[symbol] = {'df': df_bt, 'lev': config['leverage']}
    return short_signals


def prepare_long_signals(long_configs, coin_data_cache, end_date):
    long_signals = {}
    for config in long_configs:
        symbol = config['symbol']
        data = coin_data_cache.get(symbol)
        if data is None:
            continue
        df_sf = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                                config['short_ma'], config['short_sk'],
                                config['short_sks'], config['short_sd'], use_1h=False)
        df_ls = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                                config['long_ma'], config['long_sk'],
                                config['long_sks'], config['long_sd'], use_1h=False)
        df_bt = df_sf[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                        'prev_slow_k', 'prev_slow_d']].copy()
        df_bt.rename(columns={'ma': 'sf_ma', 'prev_slow_k': 'sf_k', 'prev_slow_d': 'sf_d'}, inplace=True)
        df_bt['ls_ma'] = df_ls['ma'].values
        df_bt['ls_k'] = df_ls['prev_slow_k'].values
        df_bt['ls_d'] = df_ls['prev_slow_d'].values
        df_bt = df_bt.dropna(subset=['sf_ma', 'sf_k', 'sf_d', 'ls_ma', 'ls_k', 'ls_d'])
        if end_date is not None:
            df_bt = df_bt[df_bt['timestamp'] <= end_date]
        df_bt = df_bt.set_index('timestamp').sort_index()
        if len(df_bt) >= 100:
            long_signals[symbol] = {'df': df_bt, 'lev': config['long_lev']}
    return long_signals


def run_short_only(short_signals, label="SHORT"):
    """숏 포지션만 운영"""
    funding_per_bar = FUNDING_RATE_8H * 0.5
    all_ts = set()
    for s_data in short_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    if not all_ts:
        return None
    timeline = sorted(all_ts)
    print(f"  [{label}] {len(short_signals)}코인, {len(timeline)}봉 ({timeline[0].date()} ~ {timeline[-1].date()})")

    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0
    progress_step = max(1, len(timeline) // 10)

    for t_idx, ts in enumerate(timeline):
        if t_idx % progress_step == 0:
            port_eq = cash + sum(p['margin'] for p in positions.values())
            print(f"    [{label}] {t_idx/len(timeline)*100:.0f}% ({ts.date()}) ${port_eq:,.0f}")

        active = [s for s, d in short_signals.items() if ts in d['df'].index]
        n_active = len(active)
        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        total_eq = cash + sum(p['margin'] for p in positions.values())
        max_per_slot = total_eq / n_active

        for symbol in active:
            df = short_signals[symbol]['df']
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
            curr = df.loc[ts]
            prev = df.iloc[curr_idx - 1]
            op = curr['open']
            lev = short_signals[symbol]['lev']
            short_signal = (op < prev['ma']) and (prev['prev_slow_k'] < prev['prev_slow_d'])
            pos = positions.get(symbol)

            if short_signal:
                if pos is None and cash > 1:
                    alloc = min(max_per_slot, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'short', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': lev}
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
        for sym in list(positions.keys()):
            pos = positions[sym]
            df = short_signals[sym]['df']
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            pos['cum_funding'] += pos['margin'] * pos['lev'] * funding_per_bar
            if pos['lev'] > 0 and curr['high'] >= pos['entry_price'] * (1 + 1 / pos['lev']):
                del positions[sym]
                total_trades += 1

        unrealized = 0
        for sym, pos in positions.items():
            df = short_signals[sym]['df']
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
        df = short_signals[sym]['df']
        last_p = df.iloc[-1]['close']
        pnl = pos['margin'] * (-(last_p / pos['entry_price'] - 1)) * pos['lev']
        settled = pos['margin'] + pnl - pos['cum_funding']
        exit_not = abs(pos['margin'] * pos['lev'] * (last_p / pos['entry_price']))
        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(settled, 0)

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf:
        perf['trades'] = total_trades
        perf['equity_curve'] = equity_curve
        perf['timestamps'] = eq_timestamps
    return perf


def run_long_only(long_signals, label="LONG"):
    """롱 포지션만 운영"""
    funding_per_bar = FUNDING_RATE_8H * 0.5
    all_ts = set()
    for s_data in long_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    if not all_ts:
        return None
    timeline = sorted(all_ts)
    print(f"  [{label}] {len(long_signals)}코인, {len(timeline)}봉 ({timeline[0].date()} ~ {timeline[-1].date()})")

    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0
    progress_step = max(1, len(timeline) // 10)

    for t_idx, ts in enumerate(timeline):
        if t_idx % progress_step == 0:
            port_eq = cash + sum(p['margin'] for p in positions.values())
            print(f"    [{label}] {t_idx/len(timeline)*100:.0f}% ({ts.date()}) ${port_eq:,.0f}")

        active = [s for s, d in long_signals.items() if ts in d['df'].index]
        n_active = len(active)
        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        total_eq = cash + sum(p['margin'] for p in positions.values())
        max_per_slot = total_eq / n_active

        for symbol in active:
            df = long_signals[symbol]['df']
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
            curr = df.loc[ts]
            prev = df.iloc[curr_idx - 1]
            op = curr['open']
            lev = long_signals[symbol]['lev']
            sf_active = (op < prev['sf_ma']) and (prev['sf_k'] < prev['sf_d'])
            ls_active = (op > prev['ls_ma']) and (prev['ls_k'] > prev['ls_d'])
            long_signal = (not sf_active) and ls_active
            pos = positions.get(symbol)

            if long_signal:
                if pos is None and cash > 1:
                    alloc = min(max_per_slot, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'long', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': lev}
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
        for sym in list(positions.keys()):
            pos = positions[sym]
            df = long_signals[sym]['df']
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            pos['cum_funding'] += pos['margin'] * pos['lev'] * funding_per_bar
            if pos['lev'] > 0 and curr['low'] <= pos['entry_price'] * (1 - 1 / pos['lev']):
                del positions[sym]
                total_trades += 1

        unrealized = 0
        for sym, pos in positions.items():
            df = long_signals[sym]['df']
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
        df = long_signals[sym]['df']
        last_p = df.iloc[-1]['close']
        pnl = pos['margin'] * (last_p / pos['entry_price'] - 1) * pos['lev']
        settled = pos['margin'] + pnl - pos['cum_funding']
        exit_not = abs(pos['margin'] * pos['lev'] * (last_p / pos['entry_price']))
        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(settled, 0)

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf:
        perf['trades'] = total_trades
        perf['equity_curve'] = equity_curve
        perf['timestamps'] = eq_timestamps
    return perf


def run_combined(short_signals, long_signals, coin_priority, label="COMBINED"):
    """숏+롱 코인별 우선순위 운영"""
    funding_per_bar = FUNDING_RATE_8H * 0.5
    all_ts = set()
    for s_data in short_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    for s_data in long_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    if not all_ts:
        return None
    timeline = sorted(all_ts)
    print(f"  [{label}] 숏 {len(short_signals)}, 롱 {len(long_signals)}코인, {len(timeline)}봉")

    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0
    short_trades = 0
    long_trades = 0
    progress_step = max(1, len(timeline) // 10)

    for t_idx, ts in enumerate(timeline):
        if t_idx % progress_step == 0:
            port_eq = cash + sum(p['margin'] for p in positions.values())
            print(f"    [{label}] {t_idx/len(timeline)*100:.0f}% ({ts.date()}) ${port_eq:,.0f}")

        active_short = [s for s, d in short_signals.items() if ts in d['df'].index]
        active_long = [s for s, d in long_signals.items() if ts in d['df'].index]
        n_active = len(set(active_short) | set(active_long))

        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        total_eq = cash + sum(p['margin'] for p in positions.values())
        max_per_slot = total_eq / n_active

        all_symbols = set(active_short) | set(active_long)
        for symbol in all_symbols:
            priority = coin_priority.get(symbol, 'short')
            pos = positions.get(symbol)

            # 숏 신호
            s_signal = False
            if symbol in short_signals and ts in short_signals[symbol]['df'].index:
                df = short_signals[symbol]['df']
                ci = df.index.get_loc(ts)
                if ci > 0:
                    prev = df.iloc[ci - 1]
                    s_signal = (df.loc[ts, 'open'] < prev['ma']) and (prev['prev_slow_k'] < prev['prev_slow_d'])

            # 롱 신호
            l_signal = False
            if symbol in long_signals and ts in long_signals[symbol]['df'].index:
                df = long_signals[symbol]['df']
                ci = df.index.get_loc(ts)
                if ci > 0:
                    prev = df.iloc[ci - 1]
                    op = df.loc[ts, 'open']
                    sf_act = (op < prev['sf_ma']) and (prev['sf_k'] < prev['sf_d'])
                    ls_act = (op > prev['ls_ma']) and (prev['ls_k'] > prev['ls_d'])
                    l_signal = (not sf_act) and ls_act

            if priority == 'long':
                first_sig, second_sig = l_signal, s_signal
                first_side, second_side = 'long', 'short'
            else:
                first_sig, second_sig = s_signal, l_signal
                first_side, second_side = 'short', 'long'

            def get_entry_price(side):
                if side == 'short' and symbol in short_signals:
                    return short_signals[symbol]['df'].loc[ts, 'open'] if ts in short_signals[symbol]['df'].index else None
                elif side == 'long' and symbol in long_signals:
                    return long_signals[symbol]['df'].loc[ts, 'open'] if ts in long_signals[symbol]['df'].index else None
                return None

            def get_lev(side):
                if side == 'short':
                    return short_signals[symbol]['lev']
                else:
                    return long_signals[symbol]['lev']

            def close_pos(pos, price):
                nonlocal cash, total_trades
                pr = price / pos['entry_price'] - 1
                pnl = pos['margin'] * ((-pr) if pos['side'] == 'short' else pr) * pos['lev']
                settled = pos['margin'] + pnl - pos['cum_funding']
                exit_not = abs(pos['margin'] * pos['lev'] * (price / pos['entry_price']))
                settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                cash += max(settled, 0)
                del positions[symbol]
                total_trades += 1

            def enter_pos(side, price):
                nonlocal cash, total_trades, short_trades, long_trades
                lev = get_lev(side)
                alloc = min(max_per_slot, cash * 0.995)
                if alloc > 1:
                    margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                    if margin > 0:
                        cash -= alloc
                        positions[symbol] = {'side': side, 'entry_price': price, 'margin': margin,
                                             'cum_funding': 0, 'lev': lev, 'strategy': side}
                        total_trades += 1
                        if side == 'short':
                            short_trades += 1
                        else:
                            long_trades += 1

            if first_sig:
                if pos is None and cash > 1:
                    ep = get_entry_price(first_side)
                    if ep:
                        enter_pos(first_side, ep)
                elif pos and pos['strategy'] == second_side:
                    ep = get_entry_price(second_side)
                    if ep:
                        close_pos(pos, ep)
                    ep2 = get_entry_price(first_side)
                    if ep2 and cash > 1:
                        enter_pos(first_side, ep2)
            elif second_sig:
                if pos is None and cash > 1:
                    ep = get_entry_price(second_side)
                    if ep:
                        enter_pos(second_side, ep)
                elif pos and pos['strategy'] == first_side:
                    ep = get_entry_price(first_side)
                    if ep:
                        close_pos(pos, ep)
                    ep2 = get_entry_price(second_side)
                    if ep2 and cash > 1:
                        enter_pos(second_side, ep2)
            else:
                if pos:
                    if pos['strategy'] == 'short' and symbol in short_signals and ts in short_signals[symbol]['df'].index:
                        close_pos(pos, short_signals[symbol]['df'].loc[ts, 'open'])
                    elif pos['strategy'] == 'long' and symbol in long_signals and ts in long_signals[symbol]['df'].index:
                        close_pos(pos, long_signals[symbol]['df'].loc[ts, 'open'])

        # 펀딩 + 청산
        for sym in list(positions.keys()):
            pos = positions[sym]
            if pos['strategy'] == 'short' and sym in short_signals:
                df = short_signals[sym]['df']
            elif pos['strategy'] == 'long' and sym in long_signals:
                df = long_signals[sym]['df']
            else:
                continue
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            pos['cum_funding'] += pos['margin'] * pos['lev'] * funding_per_bar
            if pos['side'] == 'short' and pos['lev'] > 0:
                if curr['high'] >= pos['entry_price'] * (1 + 1 / pos['lev']):
                    del positions[sym]
                    total_trades += 1
            elif pos['side'] == 'long' and pos['lev'] > 0:
                if curr['low'] <= pos['entry_price'] * (1 - 1 / pos['lev']):
                    del positions[sym]
                    total_trades += 1

        unrealized = 0
        for sym, pos in positions.items():
            if pos['strategy'] == 'short' and sym in short_signals:
                df = short_signals[sym]['df']
            elif pos['strategy'] == 'long' and sym in long_signals:
                df = long_signals[sym]['df']
            else:
                unrealized += pos['margin']
                continue
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
        if pos['strategy'] == 'short' and sym in short_signals:
            df = short_signals[sym]['df']
        elif pos['strategy'] == 'long' and sym in long_signals:
            df = long_signals[sym]['df']
        else:
            continue
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
    ax1.set_title('HYBRID 288 Coins - SHORT / LONG / COMBINED Effect', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2 = axes[1]
    for label_name, eq_df, col in [('SHORT', short_df, colors['short']),
                                    ('LONG', long_df, colors['long']),
                                    ('COMBINED', comb_df, colors['combined'])]:
        peak = eq_df['equity'].cummax()
        dd = (eq_df['equity'] - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, color=col, alpha=0.15)
        ax2.plot(dd.index, dd.values, color=col, linewidth=0.6, alpha=0.7, label=f'{label_name} DD')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path1 = os.path.join(save_dir, 'hybrid_long_short_effect_equity.png')
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

    fig, ax = plt.subplots(figsize=(16, 7))
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
            if abs(h) > 10:
                ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 20),
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(all_years, fontsize=11)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.set_title('HYBRID 288 Coins - Annual Returns: SHORT / LONG / COMBINED', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    path2 = os.path.join(save_dir, 'hybrid_long_short_effect_yearly.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 연간 수익률: {path2}")

    # ── 3. 요약 대시보드 ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = ['SHORT\nonly', 'LONG\nonly', 'SHORT+\nLONG']
    all_perfs = [short_perf, long_perf, combined_perf]
    cols = [colors['short'], colors['long'], colors['combined']]

    # CAGR
    ax = axes[0]
    vals = [p['cagr'] for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(bar.get_height()*0.02, 20),
                f'{val:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('CAGR (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # MDD
    ax = axes[1]
    vals = [abs(p['mdd']) for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'-{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('Max Drawdown (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Sharpe
    ax = axes[2]
    vals = [p['sharpe'] for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Total Trades
    ax = axes[3]
    vals = [p['trades'] for p in all_perfs]
    bars = ax.bar(names, vals, color=cols, alpha=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Total Trades', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('HYBRID 288 Coins - Long/Short Position Effect Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path3 = os.path.join(save_dir, 'hybrid_long_short_effect_summary.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3] 요약 대시보드: {path3}")

    # ── 4. 월별 히트맵 (COMBINED) ──
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

        vmax = min(pivot.iloc[:, :-1].max().max(), 300)
        vmin = max(pivot.iloc[:, :-1].min().min(), -100)
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
                    color_t = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7, color=color_t)
        for i, yr in enumerate(pivot['Year']):
            ax.text(len(pivot.columns) - 1.5, i, f' {yr:.0f}%', ha='left', va='center',
                    fontsize=8, fontweight='bold', color='green' if yr > 0 else 'red')

        ax.set_title(f'{lbl} - Monthly Returns', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    path4 = os.path.join(save_dir, 'hybrid_long_short_effect_monthly.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [4] 월별 히트맵: {path4}")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 70)
    print("  HYBRID 288코인 롱/숏/롱+숏 효과 분석")
    print("=" * 70)
    start_time = time.time()

    # 설정 로드
    bot_path = os.path.join(SAVE_DIR, 'binance_bot-5.py')
    short_configs, long_configs, coin_priority = load_configs_from_file(bot_path)
    lp = sum(1 for v in coin_priority.values() if v == 'long')
    sp = sum(1 for v in coin_priority.values() if v == 'short')
    print(f"  숏 {len(short_configs)}개, 롱 {len(long_configs)}개")
    print(f"  Priority: 롱우선 {lp}개, 숏우선 {sp}개")

    # 데이터 로드
    print("\n  --- 데이터 로드 ---")
    all_symbols = set(c['symbol'] for c in short_configs) | set(c['symbol'] for c in long_configs)
    coin_data_cache = {}
    for idx, symbol in enumerate(sorted(all_symbols), 1):
        if idx % 50 == 0:
            print(f"  진행: {idx}/{len(all_symbols)}...")
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data_cache[symbol] = data
    print(f"  로드: {len(coin_data_cache)}개 코인")

    all_ends = [d['df_4h']['timestamp'].max() for d in coin_data_cache.values()]
    end_date = min(all_ends)
    print(f"  종료일: {end_date}")

    # 신호 준비 (공통)
    print("\n  --- 신호 준비 ---")
    short_signals = prepare_short_signals(short_configs, coin_data_cache, end_date)
    long_signals = prepare_long_signals(long_configs, coin_data_cache, end_date)
    print(f"  숏 신호: {len(short_signals)}코인, 롱 신호: {len(long_signals)}코인")

    # 백테스트 실행
    print("\n  --- SHORT only 백테스트 ---")
    short_perf = run_short_only(short_signals)

    print("\n  --- LONG only 백테스트 ---")
    long_perf = run_long_only(long_signals)

    print("\n  --- COMBINED (숏+롱) 백테스트 ---")
    combined_perf = run_combined(short_signals, long_signals, coin_priority)

    if not short_perf or not long_perf or not combined_perf:
        print("  백테스트 실패!")
        return

    # 결과 출력
    print("\n" + "=" * 70)
    print("  결과 비교")
    print("=" * 70)
    print(f"  {'항목':<20} {'SHORT only':>15} {'LONG only':>15} {'COMBINED':>15}")
    print(f"  {'-'*65}")
    print(f"  {'CAGR':<20} {short_perf['cagr']:>14.1f}% {long_perf['cagr']:>14.1f}% {combined_perf['cagr']:>14.1f}%")
    print(f"  {'MDD':<20} {short_perf['mdd']:>14.1f}% {long_perf['mdd']:>14.1f}% {combined_perf['mdd']:>14.1f}%")
    print(f"  {'Sharpe':<20} {short_perf['sharpe']:>15.3f} {long_perf['sharpe']:>15.3f} {combined_perf['sharpe']:>15.3f}")
    print(f"  {'Total Return':<20} {short_perf['total_return']:>13.0f}% {long_perf['total_return']:>13.0f}% {combined_perf['total_return']:>13.0f}%")
    print(f"  {'총 거래 수':<20} {short_perf['trades']:>15,} {long_perf['trades']:>15,} {combined_perf['trades']:>15,}")
    print(f"  {'최종 자산':<20} ${short_perf['final_equity']:>13,.0f} ${long_perf['final_equity']:>13,.0f} ${combined_perf['final_equity']:>13,.0f}")

    # COMBINED의 롱/숏 거래 비율
    if 'short_trades' in combined_perf:
        st = combined_perf['short_trades']
        lt = combined_perf['long_trades']
        print(f"\n  COMBINED 거래 구성: 숏 {st:,} ({st/(st+lt)*100:.0f}%) + 롱 {lt:,} ({lt/(st+lt)*100:.0f}%)")

    # 시너지 효과
    synergy = combined_perf['cagr'] - max(short_perf['cagr'], long_perf['cagr'])
    print(f"\n  시너지 효과 (COMBINED - max(SHORT,LONG)): {synergy:+.1f}% CAGR")
    mdd_benefit = combined_perf['mdd'] - min(short_perf['mdd'], long_perf['mdd'])
    print(f"  MDD 변화 (COMBINED vs worst): {mdd_benefit:+.1f}%p")

    # 차트 생성
    print("\n  --- 차트 생성 ---")
    plot_all_charts(short_perf, long_perf, combined_perf, SAVE_DIR)

    elapsed = time.time() - start_time
    print(f"\n  완료! ({elapsed/60:.1f}분)")


if __name__ == '__main__':
    main()
