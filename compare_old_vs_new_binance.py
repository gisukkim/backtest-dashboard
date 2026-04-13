"""
================================================================================
바이낸스 289개 코인 기존 vs 신규 파라미터 비교 백테스트
================================================================================
- OLD: binance_bot-4.py (숏우선 고정, 기존 파라미터)
- NEW: binance_bot-5.py (코인별 롱/숏 우선, 최적화 파라미터)
- 4H 시가 기반, 비용 반영 (수수료 0.04%, 슬리피지 0.05%, 펀딩비 0.01%/8h)
- 포트폴리오 시뮬레이션: 총자산 / 활성코인수 동적 배분
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


# ============================================================
# 봇 파일에서 설정 로드
# ============================================================
def load_configs_from_file(bot_path):
    """봇 파일에서 SHORT/LONG TRADING_CONFIGS 로드"""
    with open(bot_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # SHORT_TRADING_CONFIGS
    short_start = content.index('SHORT_TRADING_CONFIGS = [')
    short_end = content.index(']', short_start) + 1
    local_ns = {}
    exec(content[short_start:short_end], {}, local_ns)
    short_configs = local_ns.get('SHORT_TRADING_CONFIGS', [])

    # LONG_TRADING_CONFIGS
    long_start = content.index('LONG_TRADING_CONFIGS = [')
    long_end = content.index(']', long_start) + 1
    local_ns2 = {}
    exec(content[long_start:long_end], {}, local_ns2)
    long_configs = local_ns2.get('LONG_TRADING_CONFIGS', [])

    # COIN_PRIORITY (신규 봇에만 있음)
    priority = {}
    if 'COIN_PRIORITY' in content:
        p_start = content.index('COIN_PRIORITY = {')
        p_end = content.index('}', p_start) + 1
        local_ns3 = {}
        exec(content[p_start:p_end], {}, local_ns3)
        priority = local_ns3.get('COIN_PRIORITY', {})

    return short_configs, long_configs, priority


# ============================================================
# 포트폴리오 백테스트 (코인별 우선순위 지원)
# ============================================================
def run_portfolio_backtest(short_configs, long_configs, coin_priority, coin_data_cache,
                           end_date=None, label=""):
    """
    포트폴리오 백테스트 실행
    coin_priority: {'SYMBOL': 'long' or 'short'} - 빈 dict이면 전체 숏우선
    """
    funding_per_bar = FUNDING_RATE_8H * 0.5  # 4H 기준

    # 숏 신호 준비
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

    # 롱 신호 준비
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

    # 타임라인
    all_ts = set()
    for s_data in short_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    for s_data in long_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    if not all_ts:
        return None
    timeline = sorted(all_ts)
    print(f"  [{label}] 타임라인: {len(timeline)}개 봉 ({timeline[0].date()} ~ {timeline[-1].date()})")

    # 시뮬레이션
    cash = CAPITAL
    positions = {}
    equity_curve = []
    eq_timestamps = []
    total_trades = 0
    short_trade_count = 0
    long_trade_count = 0
    progress_step = max(1, len(timeline) // 10)

    for t_idx, ts in enumerate(timeline):
        if t_idx % progress_step == 0:
            pct = t_idx / len(timeline) * 100
            port_eq = cash + sum(p['margin'] for p in positions.values())
            print(f"    [{label}] {pct:.0f}% ({ts.date()}) ${port_eq:,.0f} 포지션={len(positions)}")

        active_short = [s for s, d in short_signals.items() if ts in d['df'].index]
        active_long = [s for s, d in long_signals.items() if ts in d['df'].index]
        n_active = len(set(active_short) | set(active_long))

        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        total_eq_for_alloc = cash + sum(p['margin'] for p in positions.values())
        max_per_slot = total_eq_for_alloc / n_active

        # 코인별 우선순위에 따라 처리 순서 결정
        all_active_symbols = set(active_short) | set(active_long)

        for symbol in all_active_symbols:
            priority = coin_priority.get(symbol, 'short')  # 기본: 숏 우선
            pos = positions.get(symbol)

            # 숏 신호 계산
            short_signal = False
            if symbol in short_signals and ts in short_signals[symbol]['df'].index:
                df = short_signals[symbol]['df']
                curr_idx = df.index.get_loc(ts)
                if curr_idx > 0:
                    curr = df.loc[ts]
                    prev = df.iloc[curr_idx - 1]
                    short_signal = (curr['open'] < prev['ma']) and (prev['prev_slow_k'] < prev['prev_slow_d'])

            # 롱 신호 계산
            long_signal = False
            sf_active = False
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

            # 우선순위에 따른 1차/2차 신호 결정
            if priority == 'long':
                first_signal, second_signal = long_signal, short_signal
                first_side, second_side = 'long', 'short'
            else:
                first_signal, second_signal = short_signal, long_signal
                first_side, second_side = 'short', 'long'

            # 의사결정
            if first_signal:
                if pos is None and cash > 1:
                    # 1차 우선 진입
                    lev = (short_signals[symbol]['lev'] if first_side == 'short'
                           else long_signals[symbol]['lev'])
                    alloc = min(max_per_slot, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                        if margin > 0:
                            s_df = (short_signals[symbol]['df'] if first_side == 'short'
                                    else long_signals[symbol]['df'])
                            entry_price = s_df.loc[ts, 'open']
                            cash -= alloc
                            positions[symbol] = {
                                'side': first_side, 'entry_price': entry_price,
                                'margin': margin, 'cum_funding': 0, 'lev': lev,
                                'strategy': first_side
                            }
                            total_trades += 1
                            if first_side == 'short':
                                short_trade_count += 1
                            else:
                                long_trade_count += 1
                elif pos and pos['strategy'] == second_side:
                    # 반대 포지션 청산 → 1차 진입
                    s_df = (short_signals[symbol]['df'] if second_side == 'short'
                            else long_signals[symbol]['df'])
                    if ts in s_df.index:
                        op = s_df.loc[ts, 'open']
                        pr = op / pos['entry_price'] - 1
                        if pos['side'] == 'short':
                            pnl = pos['margin'] * (-pr) * pos['lev']
                        else:
                            pnl = pos['margin'] * pr * pos['lev']
                        settled = pos['margin'] + pnl - pos['cum_funding']
                        exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                        cash += max(settled, 0)
                        del positions[symbol]
                        total_trades += 1

                    # 1차 포지션 진입
                    if cash > 1:
                        lev = (short_signals[symbol]['lev'] if first_side == 'short'
                               else long_signals[symbol]['lev'])
                        alloc = min(max_per_slot, cash * 0.995)
                        if alloc > 1:
                            margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                            if margin > 0:
                                s_df2 = (short_signals[symbol]['df'] if first_side == 'short'
                                         else long_signals[symbol]['df'])
                                entry_price = s_df2.loc[ts, 'open'] if ts in s_df2.index else op
                                cash -= alloc
                                positions[symbol] = {
                                    'side': first_side, 'entry_price': entry_price,
                                    'margin': margin, 'cum_funding': 0, 'lev': lev,
                                    'strategy': first_side
                                }
                                total_trades += 1
                                if first_side == 'short':
                                    short_trade_count += 1
                                else:
                                    long_trade_count += 1

            elif second_signal:
                if pos is None and cash > 1:
                    # 2차 진입
                    lev = (short_signals[symbol]['lev'] if second_side == 'short'
                           else long_signals[symbol]['lev'])
                    alloc = min(max_per_slot, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                        if margin > 0:
                            s_df = (short_signals[symbol]['df'] if second_side == 'short'
                                    else long_signals[symbol]['df'])
                            entry_price = s_df.loc[ts, 'open']
                            cash -= alloc
                            positions[symbol] = {
                                'side': second_side, 'entry_price': entry_price,
                                'margin': margin, 'cum_funding': 0, 'lev': lev,
                                'strategy': second_side
                            }
                            total_trades += 1
                            if second_side == 'short':
                                short_trade_count += 1
                            else:
                                long_trade_count += 1
                elif pos and pos['strategy'] == first_side:
                    # 1차 포지션 청산 → 2차 진입
                    s_df = (short_signals[symbol]['df'] if first_side == 'short'
                            else long_signals[symbol]['df'])
                    if ts in s_df.index:
                        op = s_df.loc[ts, 'open']
                        pr = op / pos['entry_price'] - 1
                        if pos['side'] == 'short':
                            pnl = pos['margin'] * (-pr) * pos['lev']
                        else:
                            pnl = pos['margin'] * pr * pos['lev']
                        settled = pos['margin'] + pnl - pos['cum_funding']
                        exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                        settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                        cash += max(settled, 0)
                        del positions[symbol]
                        total_trades += 1

                    if cash > 1:
                        lev = (short_signals[symbol]['lev'] if second_side == 'short'
                               else long_signals[symbol]['lev'])
                        alloc = min(max_per_slot, cash * 0.995)
                        if alloc > 1:
                            margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                            if margin > 0:
                                s_df2 = (short_signals[symbol]['df'] if second_side == 'short'
                                         else long_signals[symbol]['df'])
                                entry_price = s_df2.loc[ts, 'open'] if ts in s_df2.index else op
                                cash -= alloc
                                positions[symbol] = {
                                    'side': second_side, 'entry_price': entry_price,
                                    'margin': margin, 'cum_funding': 0, 'lev': lev,
                                    'strategy': second_side
                                }
                                total_trades += 1
                                if second_side == 'short':
                                    short_trade_count += 1
                                else:
                                    long_trade_count += 1

            else:
                # 둘 다 OFF → 기존 포지션 청산
                if pos:
                    if pos['strategy'] == 'short' and symbol in short_signals:
                        s_df = short_signals[symbol]['df']
                    elif pos['strategy'] == 'long' and symbol in long_signals:
                        s_df = long_signals[symbol]['df']
                    else:
                        continue
                    if ts in s_df.index:
                        op = s_df.loc[ts, 'open']
                        pr = op / pos['entry_price'] - 1
                        if pos['side'] == 'short':
                            pnl = pos['margin'] * (-pr) * pos['lev']
                        else:
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

        # 에쿼티
        unrealized_total = 0
        for sym, pos in positions.items():
            if pos['strategy'] == 'short' and sym in short_signals:
                df = short_signals[sym]['df']
            elif pos['strategy'] == 'long' and sym in long_signals:
                df = long_signals[sym]['df']
            else:
                unrealized_total += pos['margin']
                continue
            if ts in df.index:
                close_p = df.loc[ts, 'close']
                if pos['side'] == 'long':
                    val = pos['margin'] + pos['margin'] * (close_p / pos['entry_price'] - 1) * pos['lev'] - pos['cum_funding']
                else:
                    val = pos['margin'] + pos['margin'] * (-(close_p / pos['entry_price'] - 1)) * pos['lev'] - pos['cum_funding']
                unrealized_total += max(val, 0)
            else:
                unrealized_total += pos['margin']
        equity_curve.append(cash + unrealized_total)
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
# 차트 생성
# ============================================================
def build_equity_df(perf):
    df = pd.DataFrame({'timestamp': perf['timestamps'], 'equity': perf['equity_curve']})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df


def compute_drawdown(equity_series):
    peak = equity_series.cummax()
    return (equity_series - peak) / peak * 100


def plot_comparison_charts(old_perf, new_perf, save_dir):
    """3개 차트 생성: 에쿼티+드로다운, 연간수익률, 요약 대시보드"""
    old_df = build_equity_df(old_perf)
    new_df = build_equity_df(new_perf)

    # ── 1. 에쿼티 + 드로다운 ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                             sharex=True, gridspec_kw={'hspace': 0.05})
    ax1 = axes[0]
    ax1.semilogy(old_df.index, old_df['equity'], color='#2196F3', linewidth=1.2,
                 label=f'OLD (v4) CAGR={old_perf["cagr"]:.0f}%', alpha=0.9)
    ax1.semilogy(new_df.index, new_df['equity'], color='#FF5722', linewidth=1.2,
                 label=f'NEW (v5) CAGR={new_perf["cagr"]:.0f}%', alpha=0.9)
    ax1.set_ylabel('Portfolio Equity (log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.set_title('Binance 289 Coins - OLD vs NEW Parameters', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2 = axes[1]
    dd_old = compute_drawdown(old_df['equity'])
    dd_new = compute_drawdown(new_df['equity'])
    ax2.fill_between(dd_old.index, dd_old.values, 0, color='#2196F3', alpha=0.3, label='OLD DD')
    ax2.fill_between(dd_new.index, dd_new.values, 0, color='#FF5722', alpha=0.3, label='NEW DD')
    ax2.plot(dd_old.index, dd_old.values, color='#2196F3', linewidth=0.5, alpha=0.7)
    ax2.plot(dd_new.index, dd_new.values, color='#FF5722', linewidth=0.5, alpha=0.7)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path1 = os.path.join(save_dir, 'binance_old_vs_new_equity.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1] 에쿼티+드로다운: {path1}")

    # ── 2. 연간 수익률 비교 ──
    def get_yearly_returns(equity_df):
        yearly = equity_df['equity'].resample('YE').last().dropna()
        returns = yearly.pct_change().dropna() * 100
        first_year_end = yearly.iloc[0]
        first_equity = equity_df['equity'].iloc[0]
        first_ret = (first_year_end / first_equity - 1) * 100
        returns = pd.concat([pd.Series({yearly.index[0]: first_ret}), returns])
        return returns

    old_yr = get_yearly_returns(old_df)
    new_yr = get_yearly_returns(new_df)
    all_years = sorted(set(old_yr.index.year) | set(new_yr.index.year))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_years))
    width = 0.35
    old_vals = [old_yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]
    new_vals = [new_yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]
    bars1 = ax.bar(x - width/2, old_vals, width, label='OLD (v4)', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, new_vals, width, label='NEW (v5)', color='#FF5722', alpha=0.8)
    for bar in bars1:
        h = bar.get_height()
        if h != 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 5),
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h != 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 5),
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_years, fontsize=11)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.set_title('Binance 289 Coins - Annual Returns: OLD vs NEW', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    path2 = os.path.join(save_dir, 'binance_old_vs_new_yearly.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 연간 수익률: {path2}")

    # ── 3. 요약 대시보드 ──
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    names = ['OLD\n(v4)', 'NEW\n(v5)']

    # CAGR
    ax = axes[0]
    vals = [old_perf['cagr'], new_perf['cagr']]
    bars = ax.bar(names, vals, color=['#2196F3', '#FF5722'], alpha=0.8, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title('CAGR (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # MDD
    ax = axes[1]
    vals = [abs(old_perf['mdd']), abs(new_perf['mdd'])]
    bars = ax.bar(names, vals, color=['#2196F3', '#FF5722'], alpha=0.8, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'-{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title('Max Drawdown (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Sharpe
    ax = axes[2]
    vals = [old_perf['sharpe'], new_perf['sharpe']]
    bars = ax.bar(names, vals, color=['#2196F3', '#FF5722'], alpha=0.8, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Trades
    ax = axes[3]
    x_pos = np.arange(2)
    w = 0.3
    old_s, old_l = old_perf['short_trades'], old_perf['long_trades']
    new_s, new_l = new_perf['short_trades'], new_perf['long_trades']
    ax.bar(x_pos - w/2, [old_s, new_s], w, label='Short', color='#E91E63', alpha=0.7)
    ax.bar(x_pos + w/2, [old_l, new_l], w, label='Long', color='#4CAF50', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['OLD', 'NEW'])
    ax.set_title('Trade Count', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (s, l) in enumerate([(old_s, old_l), (new_s, new_l)]):
        ax.text(i, max(s, l) + 100, f'Total: {s+l}', ha='center', fontsize=9)

    fig.suptitle('Binance 289 Coins - OLD vs NEW Parameters Summary', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path3 = os.path.join(save_dir, 'binance_old_vs_new_summary.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3] 요약 대시보드: {path3}")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 70)
    print("  바이낸스 289코인 기존 vs 신규 파라미터 비교 백테스트")
    print("=" * 70)
    print(f"  비용: 수수료 {FEE_RATE*100:.2f}%, 슬리피지 {SLIPPAGE_PCT*100:.2f}%, "
          f"펀딩 {FUNDING_RATE_8H*100:.3f}%/8h")
    print(f"  초기 자본: ${CAPITAL:,.0f}")
    start_time = time.time()

    # 설정 로드
    print("\n  --- 설정 로드 ---")
    old_bot_path = os.path.join(SAVE_DIR, 'binance_bot-4.py')
    new_bot_path = os.path.join(SAVE_DIR, 'binance_bot-5.py')

    old_short, old_long, old_priority = load_configs_from_file(old_bot_path)
    print(f"  OLD (v4): 숏 {len(old_short)}개, 롱 {len(old_long)}개, priority {len(old_priority)}개")

    new_short, new_long, new_priority = load_configs_from_file(new_bot_path)
    print(f"  NEW (v5): 숏 {len(new_short)}개, 롱 {len(new_long)}개, priority {len(new_priority)}개")
    long_p = sum(1 for v in new_priority.values() if v == 'long')
    short_p = sum(1 for v in new_priority.values() if v == 'short')
    print(f"  NEW priority: 롱우선 {long_p}개, 숏우선 {short_p}개")

    # 데이터 로드 (공통)
    print("\n  --- 데이터 로드 ---")
    all_symbols = set()
    for c in old_short + old_long + new_short + new_long:
        all_symbols.add(c['symbol'])
    print(f"  전체 심볼: {len(all_symbols)}개")

    coin_data_cache = {}
    for idx, symbol in enumerate(sorted(all_symbols), 1):
        if idx % 50 == 0:
            print(f"  진행: {idx}/{len(all_symbols)}...")
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data_cache[symbol] = data
    print(f"  로드 완료: {len(coin_data_cache)}개 코인")

    # 공통 end_date
    all_ends = [d['df_4h']['timestamp'].max() for d in coin_data_cache.values()]
    end_date = min(all_ends)
    print(f"  공통 종료일: {end_date}")

    # 백테스트 실행
    print("\n  --- OLD (v4) 백테스트 ---")
    old_perf = run_portfolio_backtest(old_short, old_long, old_priority, coin_data_cache,
                                      end_date=end_date, label="OLD")

    print("\n  --- NEW (v5) 백테스트 ---")
    new_perf = run_portfolio_backtest(new_short, new_long, new_priority, coin_data_cache,
                                      end_date=end_date, label="NEW")

    if not old_perf or not new_perf:
        print("  백테스트 실패!")
        return

    # 결과 출력
    print("\n" + "=" * 70)
    print("  결과 비교")
    print("=" * 70)
    print(f"  {'항목':<20} {'OLD (v4)':>15} {'NEW (v5)':>15} {'차이':>15}")
    print(f"  {'-'*65}")
    print(f"  {'CAGR':<20} {old_perf['cagr']:>14.1f}% {new_perf['cagr']:>14.1f}% {new_perf['cagr']-old_perf['cagr']:>+14.1f}%")
    print(f"  {'MDD':<20} {old_perf['mdd']:>14.1f}% {new_perf['mdd']:>14.1f}% {new_perf['mdd']-old_perf['mdd']:>+14.1f}%")
    print(f"  {'Sharpe':<20} {old_perf['sharpe']:>15.3f} {new_perf['sharpe']:>15.3f} {new_perf['sharpe']-old_perf['sharpe']:>+15.3f}")
    print(f"  {'Total Return':<20} {old_perf['total_return']:>13.0f}% {new_perf['total_return']:>13.0f}% {new_perf['total_return']-old_perf['total_return']:>+13.0f}%")
    print(f"  {'총 거래 수':<20} {old_perf['trades']:>15,} {new_perf['trades']:>15,}")
    print(f"  {'숏 거래':<20} {old_perf['short_trades']:>15,} {new_perf['short_trades']:>15,}")
    print(f"  {'롱 거래':<20} {old_perf['long_trades']:>15,} {new_perf['long_trades']:>15,}")
    print(f"  {'숏 코인 수':<20} {old_perf['short_coins']:>15} {new_perf['short_coins']:>15}")
    print(f"  {'롱 코인 수':<20} {old_perf['long_coins']:>15} {new_perf['long_coins']:>15}")
    print(f"  {'최종 자산':<20} ${old_perf['final_equity']:>13,.0f} ${new_perf['final_equity']:>13,.0f}")

    winner = "NEW (v5)" if new_perf['cagr'] > old_perf['cagr'] else "OLD (v4)"
    print(f"\n  🏆 Winner: {winner}")

    # 차트 생성
    print("\n  --- 차트 생성 ---")
    plot_comparison_charts(old_perf, new_perf, SAVE_DIR)

    elapsed = time.time() - start_time
    print(f"\n  완료! ({elapsed/60:.1f}분)")


if __name__ == '__main__':
    main()
