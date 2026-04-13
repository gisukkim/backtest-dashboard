"""
================================================================================
코인별 OLD vs NEW 파라미터 체리피킹 → HYBRID 설정 생성
================================================================================
- 289개 코인 각각 OLD/NEW 개별 백테스트
- 코인별 승자 선택 (CAGR 기준, 동률이면 Sharpe)
- HYBRID 설정으로 binance_bot-5.py 업데이트
- OLD vs NEW vs HYBRID 3자 포트폴리오 비교
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
    prepare_coin_data, prepare_signals, calculate_stochastic,
    calculate_portfolio_performance,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H
)

SAVE_DIR = os.path.expanduser("~/Downloads")
CAPITAL = 100000.0
RESULT_CSV = os.path.join(SAVE_DIR, 'hybrid_coin_comparison.csv')


# ============================================================
# 봇 파일에서 설정 로드
# ============================================================
def load_configs_from_file(bot_path):
    """봇 파일에서 SHORT/LONG TRADING_CONFIGS + COIN_PRIORITY 로드"""
    with open(bot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    short_start = content.index('SHORT_TRADING_CONFIGS = [')
    short_end = content.index(']', short_start) + 1
    local_ns = {}
    exec(content[short_start:short_end], {}, local_ns)
    short_configs = local_ns.get('SHORT_TRADING_CONFIGS', [])

    long_start = content.index('LONG_TRADING_CONFIGS = [')
    long_end = content.index(']', long_start) + 1
    local_ns2 = {}
    exec(content[long_start:long_end], {}, local_ns2)
    long_configs = local_ns2.get('LONG_TRADING_CONFIGS', [])

    priority = {}
    if 'COIN_PRIORITY' in content:
        p_start = content.index('COIN_PRIORITY = {')
        p_end = content.index('}', p_start) + 1
        local_ns3 = {}
        exec(content[p_start:p_end], {}, local_ns3)
        priority = local_ns3.get('COIN_PRIORITY', {})

    return short_configs, long_configs, priority


# ============================================================
# 개별 코인 백테스트 (숏+롱 결합)
# ============================================================
def backtest_single_coin(symbol, short_config, long_config, priority, coin_data, end_date):
    """
    단일 코인 백테스트 - 포지션 하나만 (숏 or 롱)
    priority: 'short' or 'long'
    Returns: dict with cagr, mdd, sharpe, trades or None
    """
    funding_per_bar = FUNDING_RATE_8H * 0.5  # 4H

    data = coin_data
    if data is None:
        return None

    # 숏 신호
    df_sig_s = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                               short_config['ma_period'], short_config['stoch_k_period'],
                               short_config['stoch_k_smooth'], short_config['stoch_d_period'],
                               use_1h=False)
    df_s = df_sig_s.dropna(subset=['ma', 'prev_slow_k', 'prev_slow_d'])
    if end_date:
        df_s = df_s[df_s['timestamp'] <= end_date]
    df_s = df_s.set_index('timestamp').sort_index()

    # 롱 신호
    has_long = long_config is not None
    df_l = None
    if has_long:
        df_sf = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                                long_config['short_ma'], long_config['short_sk'],
                                long_config['short_sks'], long_config['short_sd'],
                                use_1h=False)
        df_ls = prepare_signals(data['df_4h'], data['df_4h'], data['df_daily'],
                                long_config['long_ma'], long_config['long_sk'],
                                long_config['long_sks'], long_config['long_sd'],
                                use_1h=False)
        df_bt = df_sf[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                        'prev_slow_k', 'prev_slow_d']].copy()
        df_bt.rename(columns={'ma': 'sf_ma', 'prev_slow_k': 'sf_k', 'prev_slow_d': 'sf_d'}, inplace=True)
        df_bt['ls_ma'] = df_ls['ma'].values
        df_bt['ls_k'] = df_ls['prev_slow_k'].values
        df_bt['ls_d'] = df_ls['prev_slow_d'].values
        df_bt = df_bt.dropna(subset=['sf_ma', 'sf_k', 'sf_d', 'ls_ma', 'ls_k', 'ls_d'])
        if end_date:
            df_bt = df_bt[df_bt['timestamp'] <= end_date]
        df_l = df_bt.set_index('timestamp').sort_index()

    if len(df_s) < 100:
        return None

    # 타임라인
    all_ts = set(df_s.index.tolist())
    if has_long and df_l is not None and len(df_l) >= 100:
        all_ts.update(df_l.index.tolist())
    else:
        has_long = False
    timeline = sorted(all_ts)

    # 시뮬레이션 (단일 코인, 초기자본 10000)
    init_cap = 10000.0
    cash = init_cap
    pos = None
    equity_curve = []
    eq_timestamps = []
    trades = 0
    short_lev = short_config['leverage']
    long_lev = long_config['long_lev'] if has_long else 1

    for ts in timeline:
        # 숏 신호
        short_signal = False
        if ts in df_s.index:
            curr_idx = df_s.index.get_loc(ts)
            if curr_idx > 0:
                curr = df_s.loc[ts]
                prev = df_s.iloc[curr_idx - 1]
                short_signal = (curr['open'] < prev['ma']) and (prev['prev_slow_k'] < prev['prev_slow_d'])

        # 롱 신호
        long_signal = False
        if has_long and ts in df_l.index:
            curr_idx = df_l.index.get_loc(ts)
            if curr_idx > 0:
                curr = df_l.loc[ts]
                prev = df_l.iloc[curr_idx - 1]
                op = curr['open']
                sf_active = (op < prev['sf_ma']) and (prev['sf_k'] < prev['sf_d'])
                ls_active = (op > prev['ls_ma']) and (prev['ls_k'] > prev['ls_d'])
                long_signal = (not sf_active) and ls_active

        # 우선순위 결정
        if priority == 'long':
            first_sig, second_sig = long_signal, short_signal
            first_side, second_side = 'long', 'short'
            first_lev, second_lev = long_lev, short_lev
        else:
            first_sig, second_sig = short_signal, long_signal
            first_side, second_side = 'short', 'long'
            first_lev, second_lev = short_lev, long_lev

        # 가격
        op = None
        if ts in df_s.index:
            op = df_s.loc[ts, 'open']
        elif has_long and ts in df_l.index:
            op = df_l.loc[ts, 'open']

        if op is None:
            equity_curve.append(cash + (pos['margin'] if pos else 0))
            eq_timestamps.append(ts)
            continue

        # 포지션 관리
        def close_pos(price):
            nonlocal cash, pos, trades
            if pos is None:
                return
            pr = price / pos['entry_price'] - 1
            if pos['side'] == 'short':
                pnl = pos['margin'] * (-pr) * pos['lev']
            else:
                pnl = pos['margin'] * pr * pos['lev']
            settled = pos['margin'] + pnl - pos['cum_funding']
            exit_not = abs(pos['margin'] * pos['lev'] * (price / pos['entry_price']))
            settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
            cash += max(settled, 0)
            pos = None
            trades += 1

        def open_pos(side, lev, price):
            nonlocal cash, pos, trades
            if cash <= 1:
                return
            alloc = cash * 0.995
            margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
            if margin > 0:
                cash -= alloc
                pos = {'side': side, 'entry_price': price, 'margin': margin,
                       'cum_funding': 0, 'lev': lev}
                trades += 1

        if first_sig:
            if pos is None:
                open_pos(first_side, first_lev, op)
            elif pos['side'] != first_side:
                close_pos(op)
                open_pos(first_side, first_lev, op)
        elif second_sig:
            if pos is None:
                open_pos(second_side, second_lev, op)
            elif pos['side'] != second_side:
                close_pos(op)
                open_pos(second_side, second_lev, op)
        else:
            if pos is not None:
                close_pos(op)

        # 펀딩 + 청산
        if pos:
            pos['cum_funding'] += pos['margin'] * pos['lev'] * funding_per_bar
            close_p = None
            high_p = None
            low_p = None
            if ts in df_s.index:
                close_p = df_s.loc[ts, 'close']
                high_p = df_s.loc[ts, 'high']
                low_p = df_s.loc[ts, 'low']
            elif has_long and ts in df_l.index:
                close_p = df_l.loc[ts, 'close']
                high_p = df_l.loc[ts, 'high']
                low_p = df_l.loc[ts, 'low']

            if pos and high_p is not None:
                if pos['side'] == 'short' and pos['lev'] > 0:
                    if high_p >= pos['entry_price'] * (1 + 1 / pos['lev']):
                        pos = None
                        trades += 1
                elif pos['side'] == 'long' and pos['lev'] > 0:
                    if low_p <= pos['entry_price'] * (1 - 1 / pos['lev']):
                        pos = None
                        trades += 1

        # 에쿼티
        if pos and close_p:
            if pos['side'] == 'long':
                val = pos['margin'] + pos['margin'] * (close_p / pos['entry_price'] - 1) * pos['lev'] - pos['cum_funding']
            else:
                val = pos['margin'] + pos['margin'] * (-(close_p / pos['entry_price'] - 1)) * pos['lev'] - pos['cum_funding']
            equity_curve.append(cash + max(val, 0))
        else:
            equity_curve.append(cash + (pos['margin'] if pos else 0))
        eq_timestamps.append(ts)

    # 마지막 청산
    if pos:
        last_p = None
        if len(df_s) > 0:
            last_p = df_s.iloc[-1]['close']
        if has_long and df_l is not None and len(df_l) > 0:
            last_p2 = df_l.iloc[-1]['close']
            if last_p is None or (df_l.index[-1] > df_s.index[-1]):
                last_p = last_p2
        if last_p:
            pr = last_p / pos['entry_price'] - 1
            if pos['side'] == 'short':
                pnl = pos['margin'] * (-pr) * pos['lev']
            else:
                pnl = pos['margin'] * pr * pos['lev']
            settled = pos['margin'] + pnl - pos['cum_funding']
            exit_not = abs(pos['margin'] * pos['lev'] * (last_p / pos['entry_price']))
            settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
            cash += max(settled, 0)

    if len(equity_curve) < 50:
        return None

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf is None:
        return None
    perf['trades'] = trades
    return perf


# ============================================================
# 포트폴리오 백테스트 (compare_old_vs_new_binance.py와 동일)
# ============================================================
def run_portfolio_backtest(short_configs, long_configs, coin_priority, coin_data_cache,
                           end_date=None, label=""):
    funding_per_bar = FUNDING_RATE_8H * 0.5

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

    print(f"  [{label}] 숏 {len(short_signals)}코인, 롱 {len(long_signals)}코인")

    all_ts = set()
    for s_data in short_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    for s_data in long_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    if not all_ts:
        return None
    timeline = sorted(all_ts)
    print(f"  [{label}] 타임라인: {len(timeline)}개 봉 ({timeline[0].date()} ~ {timeline[-1].date()})")

    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0
    short_trade_count = 0
    long_trade_count = 0
    progress_step = max(1, len(timeline) // 5)

    for t_idx, ts in enumerate(timeline):
        if t_idx % progress_step == 0:
            pct = t_idx / len(timeline) * 100
            port_eq = cash + sum(p['margin'] for p in positions.values())
            print(f"    [{label}] {pct:.0f}% ({ts.date()}) ${port_eq:,.0f} pos={len(positions)}")

        active_short = [s for s, d in short_signals.items() if ts in d['df'].index]
        active_long = [s for s, d in long_signals.items() if ts in d['df'].index]
        n_active = len(set(active_short) | set(active_long))
        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        total_eq_for_alloc = cash + sum(p['margin'] for p in positions.values())
        max_per_slot = total_eq_for_alloc / n_active
        all_active_symbols = set(active_short) | set(active_long)

        for symbol in all_active_symbols:
            pri = coin_priority.get(symbol, 'short')
            pos = positions.get(symbol)

            short_signal = False
            if symbol in short_signals and ts in short_signals[symbol]['df'].index:
                df = short_signals[symbol]['df']
                curr_idx = df.index.get_loc(ts)
                if curr_idx > 0:
                    curr = df.loc[ts]
                    prev = df.iloc[curr_idx - 1]
                    short_signal = (curr['open'] < prev['ma']) and (prev['prev_slow_k'] < prev['prev_slow_d'])

            long_signal = False
            if symbol in long_signals and ts in long_signals[symbol]['df'].index:
                df = long_signals[symbol]['df']
                curr_idx = df.index.get_loc(ts)
                if curr_idx > 0:
                    curr = df.loc[ts]
                    prev = df.iloc[curr_idx - 1]
                    op = curr['open']
                    sf_active = (op < prev['sf_ma']) and (prev['sf_k'] < prev['sf_d'])
                    ls_active = (op > prev['ls_ma']) and (prev['ls_k'] > prev['ls_d'])
                    long_signal = (not sf_active) and ls_active

            if pri == 'long':
                first_signal, second_signal = long_signal, short_signal
                first_side, second_side = 'long', 'short'
            else:
                first_signal, second_signal = short_signal, long_signal
                first_side, second_side = 'short', 'long'

            def get_open_price(side):
                if side == 'short' and symbol in short_signals and ts in short_signals[symbol]['df'].index:
                    return short_signals[symbol]['df'].loc[ts, 'open']
                elif side == 'long' and symbol in long_signals and ts in long_signals[symbol]['df'].index:
                    return long_signals[symbol]['df'].loc[ts, 'open']
                if symbol in short_signals and ts in short_signals[symbol]['df'].index:
                    return short_signals[symbol]['df'].loc[ts, 'open']
                if symbol in long_signals and ts in long_signals[symbol]['df'].index:
                    return long_signals[symbol]['df'].loc[ts, 'open']
                return None

            def close_position(sym, price):
                nonlocal cash, total_trades
                p = positions.get(sym)
                if p is None:
                    return
                pr = price / p['entry_price'] - 1
                pnl = p['margin'] * ((-pr) if p['side'] == 'short' else pr) * p['lev']
                settled = p['margin'] + pnl - p['cum_funding']
                exit_not = abs(p['margin'] * p['lev'] * (price / p['entry_price']))
                settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                cash += max(settled, 0)
                del positions[sym]
                total_trades += 1

            def open_position(sym, side, lev, price):
                nonlocal cash, total_trades, short_trade_count, long_trade_count
                if cash <= 1:
                    return
                alloc = min(max_per_slot, cash * 0.995)
                if alloc <= 1:
                    return
                margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                if margin <= 0:
                    return
                cash -= alloc
                positions[sym] = {'side': side, 'entry_price': price, 'margin': margin,
                                   'cum_funding': 0, 'lev': lev, 'strategy': side}
                total_trades += 1
                if side == 'short':
                    short_trade_count += 1
                else:
                    long_trade_count += 1

            if first_signal:
                op = get_open_price(first_side)
                if op:
                    if pos is None:
                        lev = (short_signals[symbol]['lev'] if first_side == 'short'
                               else long_signals[symbol]['lev'])
                        open_position(symbol, first_side, lev, op)
                    elif pos['strategy'] == second_side:
                        close_position(symbol, op)
                        lev = (short_signals[symbol]['lev'] if first_side == 'short'
                               else long_signals[symbol]['lev'])
                        open_position(symbol, first_side, lev, op)
            elif second_signal:
                op = get_open_price(second_side)
                if op:
                    if pos is None:
                        lev = (short_signals[symbol]['lev'] if second_side == 'short'
                               else long_signals[symbol]['lev'])
                        open_position(symbol, second_side, lev, op)
                    elif pos['strategy'] == first_side:
                        close_position(symbol, op)
                        lev = (short_signals[symbol]['lev'] if second_side == 'short'
                               else long_signals[symbol]['lev'])
                        open_position(symbol, second_side, lev, op)
            else:
                if pos:
                    op = get_open_price(pos['strategy'])
                    if op:
                        close_position(symbol, op)

        # 펀딩 + 청산
        for sym in list(positions.keys()):
            p = positions[sym]
            if p['strategy'] == 'short' and sym in short_signals:
                df = short_signals[sym]['df']
            elif p['strategy'] == 'long' and sym in long_signals:
                df = long_signals[sym]['df']
            else:
                continue
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            p['cum_funding'] += p['margin'] * p['lev'] * funding_per_bar
            if p['side'] == 'short' and p['lev'] > 0:
                if curr['high'] >= p['entry_price'] * (1 + 1 / p['lev']):
                    del positions[sym]
                    total_trades += 1
            elif p['side'] == 'long' and p['lev'] > 0:
                if curr['low'] <= p['entry_price'] * (1 - 1 / p['lev']):
                    del positions[sym]
                    total_trades += 1

        unrealized_total = 0
        for sym, p in positions.items():
            if p['strategy'] == 'short' and sym in short_signals:
                df = short_signals[sym]['df']
            elif p['strategy'] == 'long' and sym in long_signals:
                df = long_signals[sym]['df']
            else:
                unrealized_total += p['margin']
                continue
            if ts in df.index:
                close_p = df.loc[ts, 'close']
                if p['side'] == 'long':
                    val = p['margin'] + p['margin'] * (close_p / p['entry_price'] - 1) * p['lev'] - p['cum_funding']
                else:
                    val = p['margin'] + p['margin'] * (-(close_p / p['entry_price'] - 1)) * p['lev'] - p['cum_funding']
                unrealized_total += max(val, 0)
            else:
                unrealized_total += p['margin']
        equity_curve.append(cash + unrealized_total)
        eq_timestamps.append(ts)

    for sym, p in list(positions.items()):
        if p['strategy'] == 'short' and sym in short_signals:
            df = short_signals[sym]['df']
        elif p['strategy'] == 'long' and sym in long_signals:
            df = long_signals[sym]['df']
        else:
            continue
        last_p = df.iloc[-1]['close']
        pnl_r = last_p / p['entry_price'] - 1
        pnl = p['margin'] * ((-pnl_r) if p['side'] == 'short' else pnl_r) * p['lev']
        settled = p['margin'] + pnl - p['cum_funding']
        exit_not = abs(p['margin'] * p['lev'] * (last_p / p['entry_price']))
        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(settled, 0)
    positions.clear()

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf is None:
        return None
    perf['trades'] = total_trades
    perf['short_trades'] = short_trade_count
    perf['long_trades'] = long_trade_count
    perf['short_coins'] = len(short_signals)
    perf['long_coins'] = len(long_signals)
    perf['equity_curve'] = equity_curve
    perf['timestamps'] = eq_timestamps
    return perf


# ============================================================
# 차트: 3자 비교
# ============================================================
def build_equity_df(perf):
    df = pd.DataFrame({'timestamp': perf['timestamps'], 'equity': perf['equity_curve']})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp').sort_index()


def plot_triple_comparison(old_perf, new_perf, hyb_perf, save_dir):
    old_df = build_equity_df(old_perf)
    new_df = build_equity_df(new_perf)
    hyb_df = build_equity_df(hyb_perf)

    # 1. 에쿼티 + 드로다운
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={'hspace': 0.05})
    ax1 = axes[0]
    ax1.semilogy(old_df.index, old_df['equity'], color='#2196F3', linewidth=1.2,
                 label=f'OLD (v4) CAGR={old_perf["cagr"]:.0f}%', alpha=0.9)
    ax1.semilogy(new_df.index, new_df['equity'], color='#FF5722', linewidth=1.2,
                 label=f'NEW (v5) CAGR={new_perf["cagr"]:.0f}%', alpha=0.9)
    ax1.semilogy(hyb_df.index, hyb_df['equity'], color='#4CAF50', linewidth=1.5,
                 label=f'HYBRID CAGR={hyb_perf["cagr"]:.0f}%', alpha=0.95)
    ax1.set_ylabel('Portfolio Equity (log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.set_title('Binance 288 Coins - OLD vs NEW vs HYBRID', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2 = axes[1]
    for df_eq, color, lbl in [(old_df, '#2196F3', 'OLD'), (new_df, '#FF5722', 'NEW'), (hyb_df, '#4CAF50', 'HYBRID')]:
        peak = df_eq['equity'].cummax()
        dd = (df_eq['equity'] - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.2, label=f'{lbl} DD')
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.5, alpha=0.7)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hybrid_equity_drawdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1] 에쿼티+드로다운 저장")

    # 2. 요약 대시보드
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = ['OLD\n(v4)', 'NEW\n(v5)', 'HYBRID']
    colors = ['#2196F3', '#FF5722', '#4CAF50']

    for ax, title, vals, fmt in [
        (axes[0], 'CAGR (%)', [old_perf['cagr'], new_perf['cagr'], hyb_perf['cagr']], '{:.0f}%'),
        (axes[1], 'Max Drawdown (%)', [abs(old_perf['mdd']), abs(new_perf['mdd']), abs(hyb_perf['mdd'])], '-{:.1f}%'),
        (axes[2], 'Sharpe Ratio', [old_perf['sharpe'], new_perf['sharpe'], hyb_perf['sharpe']], '{:.2f}'),
    ]:
        bars = ax.bar(names, vals, color=colors, alpha=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02 + 1,
                    fmt.format(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    ax = axes[3]
    vals = [old_perf['trades'], new_perf['trades'], hyb_perf['trades']]
    bars = ax.bar(names, vals, color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                f'{val:,}', ha='center', va='bottom', fontsize=11)
    ax.set_title('Total Trades', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('OLD vs NEW vs HYBRID - Performance Summary', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hybrid_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 요약 대시보드 저장")

    # 3. 연간 수익률
    def get_yearly_returns(equity_df):
        yearly = equity_df['equity'].resample('YE').last().dropna()
        returns = yearly.pct_change().dropna() * 100
        first = (yearly.iloc[0] / equity_df['equity'].iloc[0] - 1) * 100
        returns = pd.concat([pd.Series({yearly.index[0]: first}), returns])
        return returns

    old_yr = get_yearly_returns(old_df)
    new_yr = get_yearly_returns(new_df)
    hyb_yr = get_yearly_returns(hyb_df)
    all_years = sorted(set(old_yr.index.year) | set(new_yr.index.year) | set(hyb_yr.index.year))

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(all_years))
    w = 0.25
    for i, (yr, color, lbl) in enumerate([(old_yr, '#2196F3', 'OLD'), (new_yr, '#FF5722', 'NEW'), (hyb_yr, '#4CAF50', 'HYBRID')]):
        vals = [yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=color, alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 5),
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_years, fontsize=11)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.set_title('Annual Returns: OLD vs NEW vs HYBRID', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hybrid_yearly.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3] 연간 수익률 저장")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 70)
    print("  코인별 OLD vs NEW 체리피킹 → HYBRID 설정 생성")
    print("=" * 70)
    start_time = time.time()

    # 설정 로드
    print("\n  --- 설정 로드 ---")
    old_bot = os.path.join(SAVE_DIR, 'binance_bot-4.py')
    new_bot = os.path.join(SAVE_DIR, 'binance_bot-5.py')

    old_short, old_long, old_priority = load_configs_from_file(old_bot)
    new_short, new_long, new_priority = load_configs_from_file(new_bot)
    print(f"  OLD: 숏 {len(old_short)}개, 롱 {len(old_long)}개")
    print(f"  NEW: 숏 {len(new_short)}개, 롱 {len(new_long)}개")

    # 코인별 dict 변환
    old_short_dict = {c['symbol']: c for c in old_short}
    old_long_dict = {c['symbol']: c for c in old_long}
    new_short_dict = {c['symbol']: c for c in new_short}
    new_long_dict = {c['symbol']: c for c in new_long}

    all_symbols = sorted(set(c['symbol'] for c in old_short))
    print(f"  비교 대상: {len(all_symbols)}개 코인")

    # 데이터 로드
    print("\n  --- 데이터 로드 ---")
    coin_data_cache = {}
    for idx, symbol in enumerate(all_symbols, 1):
        if idx % 50 == 0:
            print(f"  진행: {idx}/{len(all_symbols)}...")
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data_cache[symbol] = data
    print(f"  로드 완료: {len(coin_data_cache)}개")

    all_ends = [d['df_4h']['timestamp'].max() for d in coin_data_cache.values()]
    end_date = min(all_ends)
    print(f"  공통 종료일: {end_date}")

    # 코인별 개별 백테스트
    print("\n  --- 코인별 OLD vs NEW 백테스트 ---")
    results = []
    hybrid_short_configs = []
    hybrid_long_configs = []
    hybrid_priority = {}

    old_win = 0
    new_win = 0

    for idx, symbol in enumerate(all_symbols, 1):
        if idx % 50 == 0:
            print(f"  진행: {idx}/{len(all_symbols)} (OLD승: {old_win}, NEW승: {new_win})")

        data = coin_data_cache.get(symbol)
        if data is None:
            continue

        # OLD 백테스트
        old_sc = old_short_dict.get(symbol)
        old_lc = old_long_dict.get(symbol)
        old_pri = old_priority.get(symbol, 'short')
        old_perf = None
        if old_sc:
            old_perf = backtest_single_coin(symbol, old_sc, old_lc, old_pri, data, end_date)

        # NEW 백테스트
        new_sc = new_short_dict.get(symbol)
        new_lc = new_long_dict.get(symbol)
        new_pri = new_priority.get(symbol, 'short')
        new_perf = None
        if new_sc:
            new_perf = backtest_single_coin(symbol, new_sc, new_lc, new_pri, data, end_date)

        # 승자 결정
        old_cagr = old_perf['cagr'] if old_perf else -999
        new_cagr = new_perf['cagr'] if new_perf else -999
        old_sharpe = old_perf['sharpe'] if old_perf else -999
        new_sharpe = new_perf['sharpe'] if new_perf else -999

        if old_cagr > new_cagr:
            winner = 'OLD'
            old_win += 1
            hybrid_short_configs.append(old_sc)
            if old_lc:
                hybrid_long_configs.append(old_lc)
            hybrid_priority[symbol] = old_pri
        elif new_cagr > old_cagr:
            winner = 'NEW'
            new_win += 1
            hybrid_short_configs.append(new_sc)
            if new_lc:
                hybrid_long_configs.append(new_lc)
            hybrid_priority[symbol] = new_pri
        else:
            # 동률: Sharpe 비교
            if old_sharpe >= new_sharpe:
                winner = 'OLD'
                old_win += 1
                hybrid_short_configs.append(old_sc)
                if old_lc:
                    hybrid_long_configs.append(old_lc)
                hybrid_priority[symbol] = old_pri
            else:
                winner = 'NEW'
                new_win += 1
                hybrid_short_configs.append(new_sc)
                if new_lc:
                    hybrid_long_configs.append(new_lc)
                hybrid_priority[symbol] = new_pri

        results.append({
            'Symbol': symbol,
            'OLD_CAGR': old_cagr if old_cagr != -999 else None,
            'OLD_MDD': old_perf['mdd'] if old_perf else None,
            'OLD_Sharpe': old_sharpe if old_sharpe != -999 else None,
            'OLD_Trades': old_perf['trades'] if old_perf else None,
            'NEW_CAGR': new_cagr if new_cagr != -999 else None,
            'NEW_MDD': new_perf['mdd'] if new_perf else None,
            'NEW_Sharpe': new_sharpe if new_sharpe != -999 else None,
            'NEW_Trades': new_perf['trades'] if new_perf else None,
            'Winner': winner,
            'CAGR_Diff': (new_cagr - old_cagr) if old_cagr != -999 and new_cagr != -999 else None,
            'Priority': hybrid_priority.get(symbol, 'short'),
        })

    print(f"\n  체리피킹 결과: OLD 승 {old_win}개, NEW 승 {new_win}개")
    print(f"  HYBRID: 숏 {len(hybrid_short_configs)}개, 롱 {len(hybrid_long_configs)}개")

    # 결과 CSV 저장
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
    print(f"  코인별 비교 결과: {RESULT_CSV}")

    # 포트폴리오 백테스트 3자 비교
    print("\n  --- 포트폴리오 백테스트: OLD vs NEW vs HYBRID ---")

    print("\n  [OLD]")
    old_port = run_portfolio_backtest(old_short, old_long, old_priority, coin_data_cache,
                                      end_date=end_date, label="OLD")
    print("\n  [NEW]")
    new_port = run_portfolio_backtest(new_short, new_long, new_priority, coin_data_cache,
                                      end_date=end_date, label="NEW")
    print("\n  [HYBRID]")
    hyb_port = run_portfolio_backtest(hybrid_short_configs, hybrid_long_configs, hybrid_priority,
                                      coin_data_cache, end_date=end_date, label="HYBRID")

    if not all([old_port, new_port, hyb_port]):
        print("  백테스트 실패!")
        return

    # 결과 출력
    print("\n" + "=" * 80)
    print("  포트폴리오 3자 비교 결과")
    print("=" * 80)
    print(f"  {'항목':<20} {'OLD (v4)':>15} {'NEW (v5)':>15} {'HYBRID':>15}")
    print(f"  {'-'*65}")
    print(f"  {'CAGR':<20} {old_port['cagr']:>14.1f}% {new_port['cagr']:>14.1f}% {hyb_port['cagr']:>14.1f}%")
    print(f"  {'MDD':<20} {old_port['mdd']:>14.1f}% {new_port['mdd']:>14.1f}% {hyb_port['mdd']:>14.1f}%")
    print(f"  {'Sharpe':<20} {old_port['sharpe']:>15.3f} {new_port['sharpe']:>15.3f} {hyb_port['sharpe']:>15.3f}")
    print(f"  {'Total Return':<20} {old_port['total_return']:>13.0f}% {new_port['total_return']:>13.0f}% {hyb_port['total_return']:>13.0f}%")
    print(f"  {'거래 수':<20} {old_port['trades']:>15,} {new_port['trades']:>15,} {hyb_port['trades']:>15,}")
    print(f"  {'숏 거래':<20} {old_port['short_trades']:>15,} {new_port['short_trades']:>15,} {hyb_port['short_trades']:>15,}")
    print(f"  {'롱 거래':<20} {old_port['long_trades']:>15,} {new_port['long_trades']:>15,} {hyb_port['long_trades']:>15,}")
    print(f"  {'숏 코인':<20} {old_port['short_coins']:>15} {new_port['short_coins']:>15} {hyb_port['short_coins']:>15}")
    print(f"  {'롱 코인':<20} {old_port['long_coins']:>15} {new_port['long_coins']:>15} {hyb_port['long_coins']:>15}")
    print(f"  {'최종 자산':<20} ${old_port['final_equity']:>13,.0f} ${new_port['final_equity']:>13,.0f} ${hyb_port['final_equity']:>13,.0f}")

    best = max([(old_port['cagr'], 'OLD'), (new_port['cagr'], 'NEW'), (hyb_port['cagr'], 'HYBRID')])[1]
    print(f"\n  🏆 Winner: {best}")
    print(f"  체리피킹: OLD {old_win}코인 + NEW {new_win}코인 = HYBRID {old_win + new_win}코인")

    # 차트
    print("\n  --- 차트 생성 ---")
    plot_triple_comparison(old_port, new_port, hyb_port, SAVE_DIR)

    elapsed = time.time() - start_time
    print(f"\n  완료! ({elapsed/60:.1f}분)")


if __name__ == '__main__':
    main()
