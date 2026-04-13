"""
================================================================================
기존 Bitget 파라미터 vs 신규 최적화 파라미터 비교 분석
================================================================================
대상: BTC, ETH, SOL, ADA, DOGE, XRP (6개)

비교:
  - OLD: backtest_bots.py BITGET_CONFIGS (롱우선 전략)
  - NEW: all_coins_long_vs_short_results.csv 최적화 결과 (승자 전략 적용)

분석 항목:
  1) 코인별 CAGR / MDD / Sharpe / 거래수 비교 (개별 코인 단독)
  2) 6코인 포트폴리오 백테스트 비교 (동적 배분)
  3) 에쿼티 커브 시각화 (코인별 + 포트폴리오)
  4) 파라미터 변화 테이블

비용 모델: 수수료 0.04%, 슬리피지 0.05%, 펀딩 0.01%/8h
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
import warnings

warnings.filterwarnings('ignore')

# backtest_bots.py 로직 임포트
from backtest_bots import (
    prepare_coin_data, prepare_signals,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H,
    calculate_portfolio_performance,
)

SAVE_DIR = os.path.expanduser("~/Downloads")
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5

# ==========================================
# 기존 Bitget 파라미터 (롱우선)
# ==========================================
OLD_CONFIGS = {
    'BTCUSDT': {
        'priority': 'long',
        'long_ma': 219, 'long_sk': 27, 'long_sks': 33, 'long_sd': 4, 'long_lev': 5,
        'short_ma': 247, 'short_sk': 35, 'short_sks': 52, 'short_sd': 29, 'short_lev': 1,
    },
    'ETHUSDT': {
        'priority': 'long',
        'long_ma': 152, 'long_sk': 19, 'long_sks': 28, 'long_sd': 14, 'long_lev': 5,
        'short_ma': 248, 'short_sk': 53, 'short_sks': 40, 'short_sd': 11, 'short_lev': 2,
    },
    'SOLUSDT': {
        'priority': 'long',
        'long_ma': 284, 'long_sk': 37, 'long_sks': 15, 'long_sd': 36, 'long_lev': 3,
        'short_ma': 65, 'short_sk': 39, 'short_sks': 28, 'short_sd': 50, 'short_lev': 1,
    },
    'ADAUSDT': {
        'priority': 'long',
        'long_ma': 113, 'long_sk': 91, 'long_sks': 45, 'long_sd': 14, 'long_lev': 3,
        'short_ma': 216, 'short_sk': 107, 'short_sks': 23, 'short_sd': 19, 'short_lev': 1,
    },
    'DOGEUSDT': {
        'priority': 'long',
        'long_ma': 236, 'long_sk': 36, 'long_sks': 34, 'long_sd': 50, 'long_lev': 2,
        'short_ma': 101, 'short_sk': 83, 'short_sks': 17, 'short_sd': 6, 'short_lev': 2,
    },
    'XRPUSDT': {
        'priority': 'long',
        'long_ma': 337, 'long_sk': 16, 'long_sks': 12, 'long_sd': 14, 'long_lev': 4,
        'short_ma': 37, 'short_sk': 116, 'short_sks': 60, 'short_sd': 20, 'short_lev': 1,
    },
}

# ==========================================
# 신규 최적화 파라미터 (CSV에서 승자 기준)
# ==========================================
def load_new_configs():
    """all_coins_long_vs_short_results.csv에서 6코인 승자 파라미터 로드"""
    csv_path = os.path.join(SAVE_DIR, "all_coins_long_vs_short_results.csv")
    df = pd.read_csv(csv_path)

    target = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
    configs = {}

    for symbol in target:
        row = df[df['Symbol'] == symbol]
        if row.empty:
            print(f"  ⚠️ {symbol}: CSV에 없음")
            continue
        row = row.iloc[0]
        winner = row['Winner']

        if winner == '롱 우선':
            configs[symbol] = {
                'priority': 'long',
                'long_ma': int(row['LF_Long_MA']), 'long_sk': int(row['LF_Long_SK']),
                'long_sks': int(row['LF_Long_SKs']), 'long_sd': int(row['LF_Long_SD']),
                'long_lev': int(row['LF_Long_Lev']),
                'short_ma': int(row['LF_Short_MA']), 'short_sk': int(row['LF_Short_SK']),
                'short_sks': int(row['LF_Short_SKs']), 'short_sd': int(row['LF_Short_SD']),
                'short_lev': int(row['LF_Short_Lev']),
                'csv_cagr': row['LF_Combined_CAGR'],
                'csv_mdd': row['LF_Combined_MDD'],
                'csv_sharpe': row['LF_Combined_Sharpe'],
            }
        else:
            configs[symbol] = {
                'priority': 'short',
                'short_ma': int(row['SF_Short_MA']), 'short_sk': int(row['SF_Short_SK']),
                'short_sks': int(row['SF_Short_SKs']), 'short_sd': int(row['SF_Short_SD']),
                'short_lev': int(row['SF_Short_Lev']),
                'long_ma': int(row['SF_Long_MA']), 'long_sk': int(row['SF_Long_SK']),
                'long_sks': int(row['SF_Long_SKs']), 'long_sd': int(row['SF_Long_SD']),
                'long_lev': int(row['SF_Long_Lev']),
                'csv_cagr': row['SF_Combined_CAGR'],
                'csv_mdd': row['SF_Combined_MDD'],
                'csv_sharpe': row['SF_Combined_Sharpe'],
            }
    return configs


# ==========================================
# 코인 개별 백테스트 (결합 전략)
# ==========================================
def backtest_coin(data, config):
    """
    단일 코인 결합(롱+숏) 백테스트
    config에 priority, long_*, short_* 파라미터 포함
    반환: equity_curve, timestamps, perf dict
    """
    df_4h = data['df_4h']
    df_daily = data['df_daily']
    priority = config.get('priority', 'long')

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
    df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d',
                                  'ma_short', 'short_k', 'short_d']).reset_index(drop=True)

    if len(df_bt) < 100:
        return None

    long_lev = config['long_lev']
    short_lev = config['short_lev']

    initial_capital = 10000
    equity = initial_capital
    in_long = False
    in_short = False
    entry_price = 0.0
    entry_margin = 0.0
    cum_funding = 0.0
    current_lev = 0
    trade_count = 0
    long_trade_count = 0
    short_trade_count = 0
    equity_curve = [equity]
    timestamps = [df_bt.iloc[0]['timestamp']]

    for i in range(1, len(df_bt)):
        prev = df_bt.iloc[i - 1]
        curr = df_bt.iloc[i]
        op = curr['open']

        long_signal = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
        short_signal = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])

        def close_pos():
            nonlocal equity, in_long, in_short, entry_price, entry_margin, cum_funding, trade_count, current_lev
            if entry_price > 0:
                pr = op / entry_price - 1
                if in_short:
                    pr = -pr
                pnl = entry_margin * pr * current_lev
                equity = entry_margin + pnl - cum_funding
                exit_not = abs(entry_margin * current_lev * (op / entry_price))
                equity -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                equity = max(equity, 0)
            in_long = False
            in_short = False
            entry_price = 0.0
            entry_margin = 0.0
            cum_funding = 0.0
            trade_count += 1

        def enter_pos(direction, lev):
            nonlocal equity, in_long, in_short, entry_price, entry_margin, cum_funding
            nonlocal trade_count, long_trade_count, short_trade_count, current_lev
            fee = equity * FEE_RATE * lev
            slip = equity * SLIPPAGE_PCT * lev
            equity -= (fee + slip)
            entry_margin = equity
            entry_price = op
            cum_funding = 0.0
            current_lev = lev
            trade_count += 1
            if direction == 'long':
                in_long = True
                long_trade_count += 1
            else:
                in_short = True
                short_trade_count += 1

        # 포지션 전환 로직
        if priority == 'long':
            if long_signal:
                if in_short:
                    close_pos()
                if not in_long and equity > 0:
                    enter_pos('long', long_lev)
            elif short_signal and not in_long:
                if not in_short and equity > 0:
                    enter_pos('short', short_lev)
            else:
                if in_long:
                    close_pos()
                if in_short:
                    close_pos()
        else:  # short priority
            if short_signal:
                if in_long:
                    close_pos()
                if not in_short and equity > 0:
                    enter_pos('short', short_lev)
            elif long_signal and not in_short:
                if not in_long and equity > 0:
                    enter_pos('long', long_lev)
            else:
                if in_short:
                    close_pos()
                if in_long:
                    close_pos()

        # 포지션 보유 중
        if (in_long or in_short) and entry_price > 0:
            pr = curr['close'] / entry_price - 1
            if in_short:
                pr = -pr
            unrealized = entry_margin * pr * current_lev
            cum_funding += entry_margin * current_lev * FUNDING_PER_4H
            display = entry_margin + unrealized - cum_funding

            # 강제청산
            if current_lev > 0:
                if in_long:
                    liq = entry_price * (1 - 1 / current_lev)
                    triggered = curr['low'] <= liq
                else:
                    liq = entry_price * (1 + 1 / current_lev)
                    triggered = curr['high'] >= liq
                if triggered:
                    equity = 0
                    in_long = False
                    in_short = False
                    entry_price = 0.0
                    entry_margin = 0.0
                    cum_funding = 0.0
                    equity_curve.append(0)
                    timestamps.append(curr['timestamp'])
                    break
            equity_curve.append(max(display, 0))
        else:
            equity_curve.append(max(equity, 0))
        timestamps.append(curr['timestamp'])

    # 마지막 정리
    if (in_long or in_short) and entry_price > 0:
        lp = df_bt.iloc[-1]['close']
        pr = lp / entry_price - 1
        if in_short:
            pr = -pr
        pnl = entry_margin * pr * current_lev
        equity = entry_margin + pnl - cum_funding
        exit_not = abs(entry_margin * current_lev * (lp / entry_price))
        equity -= exit_not * (FEE_RATE + SLIPPAGE_PCT)

    eq_arr = np.array(equity_curve)
    total_days = (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 0

    if len(eq_arr) < 2 or total_days < 30:
        return None

    eq_arr = np.maximum(eq_arr, 0.01)
    total_return = eq_arr[-1] / eq_arr[0]
    years = total_days / 365.25
    cagr = (total_return ** (1 / years) - 1) * 100 if total_return > 0 else -999

    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    mdd = dd.min()

    rets = np.diff(eq_arr) / eq_arr[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 6) if len(rets) > 10 and rets.std() > 0 else 0

    return {
        'cagr': cagr, 'mdd': mdd, 'sharpe': sharpe,
        'trades': trade_count, 'long_trades': long_trade_count, 'short_trades': short_trade_count,
        'equity_curve': eq_arr.tolist(), 'timestamps': timestamps,
        'final_equity': eq_arr[-1], 'initial_equity': eq_arr[0],
        'days': total_days,
    }


# ==========================================
# 6코인 포트폴리오 백테스트
# ==========================================
def run_portfolio(configs_dict, coin_data_cache, initial_capital=10000):
    """
    6코인 동적 배분 포트폴리오 시뮬레이션
    configs_dict: {symbol: config_dict, ...}
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
        df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d',
                                      'ma_short', 'short_k', 'short_d'])
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
    long_trades = 0
    short_trades = 0

    for ts in timeline:
        active_coins = [s for s, d in coin_signals.items() if ts in d['df'].index]
        n_active = len(active_coins)

        if n_active == 0:
            total_equity = cash + sum(p['margin'] for p in positions.values())
            equity_curve.append(total_equity)
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
                nonlocal cash, total_trades, long_trades, short_trades
                total_eq = cash + sum(p['margin'] for p in positions.values())
                alloc = min(total_eq / n_active, cash * 0.995)
                if alloc > 1:
                    cost = alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                    margin = alloc - cost
                    if margin > 0:
                        cash -= alloc
                        positions[symbol] = {
                            'side': side, 'entry_price': op,
                            'margin': margin, 'cum_funding': 0, 'lev': lev
                        }
                        total_trades += 1
                        if side == 'long':
                            long_trades += 1
                        else:
                            short_trades += 1

            # 포지션 전환 로직 (priority 기반)
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
                    liq = pos['entry_price'] * (1 - 1 / pos['lev'])
                    if curr['low'] <= liq:
                        del positions[symbol]
                        total_trades += 1
                elif pos['side'] == 'short' and pos['lev'] > 0:
                    liq = pos['entry_price'] * (1 + 1 / pos['lev'])
                    if curr['high'] >= liq:
                        del positions[symbol]
                        total_trades += 1

        # 에쿼티 계산
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
    if perf:
        perf['trades'] = total_trades
        perf['long_trades'] = long_trades
        perf['short_trades'] = short_trades
        perf['coins'] = len(coin_signals)
        perf['equity_curve'] = equity_curve
        perf['timestamps'] = eq_timestamps
    return perf


# ==========================================
# 시각화
# ==========================================

def plot_coin_comparison(old_results, new_results, save_path):
    """코인별 에쿼티 커브 비교 (2×3 그리드)"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT']
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, symbol in enumerate(symbols):
        ax = axes[idx]
        old = old_results.get(symbol)
        new = new_results.get(symbol)

        if old:
            ts_old = pd.to_datetime(old['timestamps'])
            ax.semilogy(ts_old, old['equity_curve'], color='#2196F3', linewidth=1.2,
                        label=f"OLD CAGR:{old['cagr']:.0f}%", alpha=0.9)
        if new:
            ts_new = pd.to_datetime(new['timestamps'])
            ax.semilogy(ts_new, new['equity_curve'], color='#FF5722', linewidth=1.2,
                        label=f"NEW CAGR:{new['cagr']:.0f}%", alpha=0.9)

        ax.set_title(f"{symbol}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3, which='both')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    fig.suptitle('코인별 에쿼티 커브: OLD(파랑) vs NEW(주황)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 코인별 에쿼티: {save_path}")


def plot_portfolio_comparison(old_perf, new_perf, save_path):
    """포트폴리오 에쿼티 + 드로다운"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                              sharex=True, gridspec_kw={'hspace': 0.05})

    old_ts = pd.to_datetime(old_perf['timestamps'])
    new_ts = pd.to_datetime(new_perf['timestamps'])
    old_eq = np.array(old_perf['equity_curve'])
    new_eq = np.array(new_perf['equity_curve'])

    ax1 = axes[0]
    ax1.semilogy(old_ts, old_eq, color='#2196F3', linewidth=1.5,
                 label=f"OLD (롱우선 고정) CAGR:{old_perf['cagr']:.0f}%", alpha=0.9)
    ax1.semilogy(new_ts, new_eq, color='#FF5722', linewidth=1.5,
                 label=f"NEW (최적 방향) CAGR:{new_perf['cagr']:.0f}%", alpha=0.9)
    ax1.set_ylabel('Portfolio Equity (log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_title('6코인 포트폴리오: OLD vs NEW 파라미터', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # 드로다운
    ax2 = axes[1]
    for label, ts, eq, color in [
        ('OLD', old_ts, old_eq, '#2196F3'),
        ('NEW', new_ts, new_eq, '#FF5722'),
    ]:
        eq_s = pd.Series(eq, index=ts)
        peak = eq_s.cummax()
        dd = (eq_s - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.3, label=f'{label} DD')
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.5, alpha=0.7)

    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 포트폴리오 에쿼티: {save_path}")


def plot_summary_bars(old_results, new_results, old_port, new_port, save_path):
    """요약 바차트: CAGR, MDD, Sharpe 비교"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'Portfolio']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    old_cagrs = [old_results[s]['cagr'] if s in old_results else 0 for s in symbols[:-1]] + [old_port['cagr']]
    new_cagrs = [new_results[s]['cagr'] if s in new_results else 0 for s in symbols[:-1]] + [new_port['cagr']]
    old_mdds = [abs(old_results[s]['mdd']) if s in old_results else 0 for s in symbols[:-1]] + [abs(old_port['mdd'])]
    new_mdds = [abs(new_results[s]['mdd']) if s in new_results else 0 for s in symbols[:-1]] + [abs(new_port['mdd'])]
    old_sharpes = [old_results[s]['sharpe'] if s in old_results else 0 for s in symbols[:-1]] + [old_port['sharpe']]
    new_sharpes = [new_results[s]['sharpe'] if s in new_results else 0 for s in symbols[:-1]] + [new_port['sharpe']]

    x = np.arange(len(symbols))
    w = 0.35

    # CAGR
    ax = axes[0]
    ax.bar(x - w/2, old_cagrs, w, label='OLD', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, new_cagrs, w, label='NEW', color='#FF5722', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('USDT','') for s in symbols], rotation=45, fontsize=10)
    ax.set_title('CAGR (%)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # MDD
    ax = axes[1]
    ax.bar(x - w/2, old_mdds, w, label='OLD', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, new_mdds, w, label='NEW', color='#FF5722', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('USDT','') for s in symbols], rotation=45, fontsize=10)
    ax.set_title('MDD (% absolute)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Sharpe
    ax = axes[2]
    ax.bar(x - w/2, old_sharpes, w, label='OLD', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, new_sharpes, w, label='NEW', color='#FF5722', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('USDT','') for s in symbols], rotation=45, fontsize=10)
    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('OLD vs NEW 파라미터 비교', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 요약 바차트: {save_path}")


# ==========================================
# 메인
# ==========================================
def main():
    print("=" * 100)
    print("📊 기존 Bitget 파라미터 vs 신규 최적화 파라미터 비교 분석")
    print("=" * 100)

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT']

    # 신규 파라미터 로드
    print("\n📥 신규 최적화 파라미터 로드...")
    new_configs = load_new_configs()
    for s in symbols:
        if s in new_configs:
            c = new_configs[s]
            print(f"  {s}: {c['priority']}우선")

    # 데이터 로드
    print("\n📥 코인 데이터 로드...")
    coin_data = {}
    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data[symbol] = data
            print(f"✅ {data['days']}일")
        else:
            print("❌ 실패")

    # ==========================================
    # 1) 코인별 개별 백테스트
    # ==========================================
    print(f"\n{'=' * 100}")
    print("📊 Part 1: 코인별 개별 백테스트 비교")
    print(f"{'=' * 100}")

    old_results = {}
    new_results = {}

    for symbol in symbols:
        data = coin_data.get(symbol)
        if data is None:
            continue

        old_config = OLD_CONFIGS.get(symbol)
        new_config = new_configs.get(symbol)

        if old_config:
            old_results[symbol] = backtest_coin(data, old_config)
        if new_config:
            new_results[symbol] = backtest_coin(data, new_config)

    # 파라미터 변화 테이블
    print(f"\n{'─' * 120}")
    print(f"{'코인':<12} {'항목':<8} {'OLD':>50} {'NEW':>50}")
    print(f"{'─' * 120}")

    for symbol in symbols:
        oc = OLD_CONFIGS.get(symbol, {})
        nc = new_configs.get(symbol, {})

        old_pri = oc.get('priority', 'long')
        new_pri = nc.get('priority', 'long')
        pri_change = f" → {new_pri}" if old_pri != new_pri else ""

        old_long = f"MA:{oc.get('long_ma','-')} SK:{oc.get('long_sk','-')}/{oc.get('long_sks','-')}/{oc.get('long_sd','-')} Lev:{oc.get('long_lev','-')}x"
        new_long = f"MA:{nc.get('long_ma','-')} SK:{nc.get('long_sk','-')}/{nc.get('long_sks','-')}/{nc.get('long_sd','-')} Lev:{nc.get('long_lev','-')}x"
        old_short = f"MA:{oc.get('short_ma','-')} SK:{oc.get('short_sk','-')}/{oc.get('short_sks','-')}/{oc.get('short_sd','-')} Lev:{oc.get('short_lev','-')}x"
        new_short = f"MA:{nc.get('short_ma','-')} SK:{nc.get('short_sk','-')}/{nc.get('short_sks','-')}/{nc.get('short_sd','-')} Lev:{nc.get('short_lev','-')}x"

        print(f"{symbol:<12} {'우선순위':<8} {'롱 우선':>50} {new_pri + ' 우선' + pri_change:>50}")
        print(f"{'':12} {'롱 파라':<8} {old_long:>50} {new_long:>50}")
        print(f"{'':12} {'숏 파라':<8} {old_short:>50} {new_short:>50}")
        print(f"{'─' * 120}")

    # 성과 비교 테이블
    print(f"\n{'=' * 110}")
    print(f"{'코인':<12} {'':5} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'거래수':>6} {'롱':>5} {'숏':>5} {'우선순위':>8}")
    print(f"{'=' * 110}")

    for symbol in symbols:
        old = old_results.get(symbol)
        new = new_results.get(symbol)
        nc = new_configs.get(symbol, {})

        if old:
            print(f"{symbol:<12} {'OLD':5} {old['cagr']:>7.1f}% {old['mdd']:>7.1f}% {old['sharpe']:>8.2f} {old['trades']:>6} {old['long_trades']:>5} {old['short_trades']:>5} {'롱 우선':>8}")
        if new:
            diff_cagr = new['cagr'] - old['cagr'] if old else 0
            marker = '🔺' if diff_cagr > 0 else '🔻'
            print(f"{'':12} {'NEW':5} {new['cagr']:>7.1f}% {new['mdd']:>7.1f}% {new['sharpe']:>8.2f} {new['trades']:>6} {new['long_trades']:>5} {new['short_trades']:>5} {nc.get('priority','long') + ' 우선':>8}")
            print(f"{'':12} {marker:5} {diff_cagr:>+7.1f}% {new['mdd']-old['mdd']:>+7.1f}% {new['sharpe']-old['sharpe']:>+8.2f}")
        print(f"{'─' * 110}")

    # 승패 집계
    old_wins = 0
    new_wins = 0
    for symbol in symbols:
        old = old_results.get(symbol)
        new = new_results.get(symbol)
        if old and new:
            if new['cagr'] > old['cagr']:
                new_wins += 1
            else:
                old_wins += 1

    print(f"\n  코인별 승패: OLD {old_wins}승 vs NEW {new_wins}승")

    # ==========================================
    # 2) 포트폴리오 백테스트
    # ==========================================
    print(f"\n{'=' * 100}")
    print("📊 Part 2: 6코인 포트폴리오 백테스트 비교")
    print(f"{'=' * 100}")

    print("  OLD 포트폴리오...", end=" ", flush=True)
    old_port = run_portfolio(OLD_CONFIGS, coin_data)
    if old_port:
        print(f"✅ CAGR:{old_port['cagr']:.1f}% MDD:{old_port['mdd']:.1f}% Sharpe:{old_port['sharpe']:.2f}")
    else:
        print("❌ 실패")

    print("  NEW 포트폴리오...", end=" ", flush=True)
    new_port = run_portfolio(new_configs, coin_data)
    if new_port:
        print(f"✅ CAGR:{new_port['cagr']:.1f}% MDD:{new_port['mdd']:.1f}% Sharpe:{new_port['sharpe']:.2f}")
    else:
        print("❌ 실패")

    if old_port and new_port:
        print(f"\n  {'항목':<20} {'OLD':>15} {'NEW':>15} {'변화':>15}")
        print(f"  {'─' * 65}")
        print(f"  {'CAGR':<20} {old_port['cagr']:>14.1f}% {new_port['cagr']:>14.1f}% {new_port['cagr']-old_port['cagr']:>+14.1f}%")
        print(f"  {'MDD':<20} {old_port['mdd']:>14.1f}% {new_port['mdd']:>14.1f}% {new_port['mdd']-old_port['mdd']:>+14.1f}%")
        print(f"  {'Sharpe':<20} {old_port['sharpe']:>15.2f} {new_port['sharpe']:>15.2f} {new_port['sharpe']-old_port['sharpe']:>+15.2f}")
        print(f"  {'거래수':<20} {old_port['trades']:>15} {new_port['trades']:>15}")
        print(f"  {'롱 거래':<20} {old_port['long_trades']:>15} {new_port['long_trades']:>15}")
        print(f"  {'숏 거래':<20} {old_port['short_trades']:>15} {new_port['short_trades']:>15}")

        if old_port.get('initial_equity') and new_port.get('initial_equity'):
            print(f"\n  {'최종 자산':<20} ${old_port['final_equity']:>13,.0f} ${new_port['final_equity']:>13,.0f}")

    # ==========================================
    # 3) 시각화
    # ==========================================
    print(f"\n{'=' * 100}")
    print("📊 Part 3: 차트 생성")
    print(f"{'=' * 100}")

    plot_coin_comparison(
        old_results, new_results,
        os.path.join(SAVE_DIR, 'compare_old_vs_new_coins.png')
    )

    if old_port and new_port:
        plot_portfolio_comparison(
            old_port, new_port,
            os.path.join(SAVE_DIR, 'compare_old_vs_new_portfolio.png')
        )

        plot_summary_bars(
            old_results, new_results, old_port, new_port,
            os.path.join(SAVE_DIR, 'compare_old_vs_new_summary.png')
        )

    # ==========================================
    # 4) 최종 판정
    # ==========================================
    print(f"\n{'=' * 100}")
    print("📊 최종 판정")
    print(f"{'=' * 100}")

    if old_port and new_port:
        cagr_better = "NEW" if new_port['cagr'] > old_port['cagr'] else "OLD"
        mdd_better = "NEW" if new_port['mdd'] > old_port['mdd'] else "OLD"  # MDD는 높을수록(덜 음수) 좋음
        sharpe_better = "NEW" if new_port['sharpe'] > old_port['sharpe'] else "OLD"

        score_new = sum([
            1 if new_port['cagr'] > old_port['cagr'] else 0,
            1 if new_port['mdd'] > old_port['mdd'] else 0,
            1 if new_port['sharpe'] > old_port['sharpe'] else 0,
        ])

        print(f"  포트폴리오 CAGR:  {cagr_better} 우세 ({old_port['cagr']:.1f}% → {new_port['cagr']:.1f}%)")
        print(f"  포트폴리오 MDD:   {mdd_better} 우세 ({old_port['mdd']:.1f}% → {new_port['mdd']:.1f}%)")
        print(f"  포트폴리오 Sharpe: {sharpe_better} 우세 ({old_port['sharpe']:.2f} → {new_port['sharpe']:.2f})")
        print(f"  코인별 승패: OLD {old_wins} vs NEW {new_wins}")
        print(f"\n  → 종합: {'NEW 파라미터 우세 ✅' if score_new >= 2 else 'OLD 파라미터 유지 권장 ⚠️'}")

    print(f"\n✅ 비교 분석 완료!")
    print(f"  차트 저장: {SAVE_DIR}/compare_old_vs_new_*.png")


if __name__ == "__main__":
    main()
