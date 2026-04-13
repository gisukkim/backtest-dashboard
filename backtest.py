"""
================================================================================
자동매매 전략 통합 백테스트 시스템
================================================================================
전략 1: TQQQ Sniper (종가 기준, 2011~2026, 수수료 0.07%)
전략 2: 바이낸스 Spot (시가 기준, 전체기간, 수수료 0.075%)
전략 3: 바이낸스 Futures 숏/롱 (시가 기준, 전체기간, 수수료 0.04%, 펀딩비)
전략 4: 비트겟 Futures 숏/롱 (시가 기준, 전체기간, 수수료 0.04%, 펀딩비)
================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

from all_coin_configs import SPOT_ALL_COINS, FUTURES_SHORT_ALL, FUTURES_LONG_ALL

# ═══════════════════════════════════════════════════════════════════════════════
# 공통 설정
# ═══════════════════════════════════════════════════════════════════════════════

SLIPPAGE_PCT = 0.05  # 슬리피지 0.05%
FUNDING_RATE_8H = 0.01  # 8시간 펀딩비 0.01% (연 ~10.95%)
INITIAL_CAPITAL = 10000  # 초기 자본금 $10,000

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 다운로드 유틸리티
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol, interval, start_date=None, limit=1500):
    """바이낸스 Public API로 캔들 데이터 가져오기"""
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []

    if start_date:
        start_ms = int(start_date.timestamp() * 1000)
    else:
        start_ms = None

    while True:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_ms:
            params['startTime'] = start_ms

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [!] {symbol} {interval} 데이터 다운로드 실패: {e}")
            break

        if not data:
            break

        all_data.extend(data)

        if len(data) < limit:
            break

        start_ms = data[-1][0] + 1
        time.sleep(0.1)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df[~df.index.duplicated(keep='first')]
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_binance_futures_klines(symbol, interval, start_date=None, limit=1500):
    """바이낸스 Futures Public API로 캔들 데이터 가져오기"""
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []

    if start_date:
        start_ms = int(start_date.timestamp() * 1000)
    else:
        start_ms = None

    while True:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_ms:
            params['startTime'] = start_ms

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [!] {symbol} futures {interval} 데이터 다운로드 실패: {e}")
            break

        if not data:
            break

        all_data.extend(data)

        if len(data) < limit:
            break

        start_ms = data[-1][0] + 1
        time.sleep(0.1)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df[~df.index.duplicated(keep='first')]
    return df[['open', 'high', 'low', 'close', 'volume']]


def get_cached_data(symbol, interval, fetcher, start_date=None, prefix="spot"):
    """캐시된 데이터 로드 또는 다운로드"""
    safe_symbol = symbol.replace('/', '')
    cache_file = os.path.join(DATA_CACHE_DIR, f"{prefix}_{safe_symbol}_{interval}.csv")

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) > 0:
            return df

    print(f"  다운로드: {prefix} {symbol} {interval}...")

    # 심볼 정리: 슬래시 제거 (FLOKI/USDT -> FLOKIUSDT)
    api_symbol = symbol.replace('/', '')

    df = fetcher(api_symbol, interval, start_date=start_date)

    if df is not None and len(df) > 0:
        df.to_csv(cache_file)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 지표 계산 공통 함수
# ═══════════════════════════════════════════════════════════════════════════════

def calc_sma(series, period):
    return series.rolling(window=period).mean()


def calc_stochastic(df, k_period, k_smooth, d_period):
    """Slow Stochastic 계산 - (Fast%K -> Slow%K = SMA(Fast%K) -> Slow%D = SMA(Slow%K))"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    fast_k = ((df['close'] - low_min) / denom) * 100
    slow_k = fast_k.rolling(window=k_smooth).mean()
    slow_d = slow_k.rolling(window=d_period).mean()
    return slow_k, slow_d


# ═══════════════════════════════════════════════════════════════════════════════
# 성과 지표 계산
# ═══════════════════════════════════════════════════════════════════════════════

def calc_performance(equity_curve, name="Strategy"):
    """성과 지표 계산"""
    if len(equity_curve) < 2:
        return {}

    eq = equity_curve.copy()
    eq = eq.dropna()

    total_days = (eq.index[-1] - eq.index[0]).days
    if total_days == 0:
        return {}

    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    years = total_days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # MDD
    peak = eq.cummax()
    drawdown = (eq - peak) / peak * 100
    mdd = drawdown.min()

    # Daily returns
    daily_ret = eq.pct_change().dropna()

    # Sharpe (risk-free = 0)
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
    else:
        sharpe = 0

    # Sortino
    neg_ret = daily_ret[daily_ret < 0]
    if len(neg_ret) > 0 and neg_ret.std() > 0:
        sortino = (daily_ret.mean() / neg_ret.std()) * np.sqrt(365)
    else:
        sortino = 0

    # Win rate (daily)
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0

    return {
        'name': name,
        'period': f"{eq.index[0].strftime('%Y-%m-%d')} ~ {eq.index[-1].strftime('%Y-%m-%d')}",
        'total_days': total_days,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': win_rate,
        'final_equity': eq.iloc[-1],
    }


def print_performance(perf):
    """성과 출력"""
    if not perf:
        print("  성과 데이터 없음")
        return

    print(f"\n{'='*60}")
    print(f"  {perf['name']}")
    print(f"{'='*60}")
    print(f"  기간: {perf['period']} ({perf['total_days']}일)")
    print(f"  최종 자산: ${perf['final_equity']:,.2f}")
    print(f"  총 수익률: {perf['total_return']:,.2f}%")
    print(f"  CAGR: {perf['cagr']:.2f}%")
    print(f"  MDD: {perf['mdd']:.2f}%")
    print(f"  Sharpe: {perf['sharpe']:.3f}")
    print(f"  Sortino: {perf['sortino']:.3f}")
    print(f"  승률(일별): {perf['win_rate']:.1f}%")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# 전략 1: TQQQ Sniper 백테스트
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_tqqq(initial_capital=INITIAL_CAPITAL):
    """
    TQQQ Sniper 전략 백테스트
    - 종가 기준으로 시그널 판단 및 거래
    - 수수료 0.07%
    - 기간: 2011~2026
    """
    print("\n" + "=" * 80)
    print("전략 1: TQQQ Sniper 백테스트")
    print("=" * 80)

    FEE_RATE = 0.0007  # 0.07%

    # 파라미터 (tqqq_bot.py에서 가져옴)
    stoch_config = {'period': 112, 'k_period': 78, 'd_period': 38}
    ma_periods = [19, 49, 192, 266]

    # 데이터 다운로드
    print("TQQQ 데이터 다운로드 중...")
    ticker = yf.Ticker('TQQQ')
    data = ticker.history(start='2010-01-01', end='2026-12-31', auto_adjust=True)

    if data is None or data.empty:
        print("  [!] TQQQ 데이터 다운로드 실패")
        return None

    df = pd.DataFrame({
        'Open': data['Open'],
        'High': data['High'],
        'Low': data['Low'],
        'Close': data['Close']
    }).dropna()

    print(f"  데이터 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}일)")

    # 지표 계산
    p, k, d_p = stoch_config.values()
    df['HH'] = df['High'].rolling(window=p).max()
    df['LL'] = df['Low'].rolling(window=p).min()
    df['%K'] = ((df['Close'] - df['LL']) / (df['HH'] - df['LL']) * 100).rolling(window=k).mean()
    df['%D'] = df['%K'].rolling(window=d_p).mean()

    for ma in ma_periods:
        df[f'MA{ma}'] = df['Close'].rolling(window=ma).mean()

    df = df.dropna()
    print(f"  지표 계산 후 데이터: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}일)")

    # 백테스트
    equity = initial_capital
    prev_tqqq_ratio = 0.0
    equity_history = []
    trade_count = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]

        # 시그널 계산 (종가 기준)
        is_bullish = curr['%K'] > curr['%D']
        ma_signals = {p: curr['Close'] > curr[f'MA{p}'] for p in ma_periods}

        if is_bullish:
            tqqq_ratio = sum(ma_signals.values()) * 0.25
        else:
            tqqq_ratio = (int(ma_signals[19]) + int(ma_signals[49])) * 0.5

        # 비중 변경 시 거래 실행
        change = tqqq_ratio - prev_tqqq_ratio
        if abs(change) > 0.001:
            # 매매 금액에 대한 수수료 + 슬리피지
            trade_value = abs(change) * equity
            fee = trade_value * FEE_RATE
            slippage = trade_value * (SLIPPAGE_PCT / 100)
            equity -= (fee + slippage)
            trade_count += 1

        # 일일 수익률 반영
        if i > 0:
            daily_return = (curr['Close'] / df.iloc[i - 1]['Close']) - 1
            # TQQQ 비중만큼 수익률 반영
            equity *= (1 + prev_tqqq_ratio * daily_return)

        prev_tqqq_ratio = tqqq_ratio
        equity_history.append({'date': curr.name, 'equity': equity})

    eq_df = pd.DataFrame(equity_history)
    eq_df.set_index('date', inplace=True)

    print(f"  총 거래 횟수: {trade_count}")

    perf = calc_performance(eq_df['equity'], "TQQQ Sniper (종가 기준, 수수료 0.07%)")
    perf['trade_count'] = trade_count
    print_performance(perf)

    return eq_df['equity'], perf


# ═══════════════════════════════════════════════════════════════════════════════
# 전략 2: 바이낸스 Spot 백테스트 (전체 135개 코인)
# ═══════════════════════════════════════════════════════════════════════════════


def backtest_binance_spot_single(symbol, config, fee_rate=0.00075, initial_capital=INITIAL_CAPITAL):
    """바이낸스 Spot 개별 코인 백테스트 (4H 봉 기준 판단)"""

    # 4H 데이터 (MA 계산용) - 2017년부터 전체기간
    start = datetime(2017, 1, 1)
    df_4h = get_cached_data(symbol, '4h', lambda s, i, **kw: fetch_binance_klines(s, i, start_date=start), prefix="spot")
    if df_4h is None or len(df_4h) < config['ma'] + 50:
        return None

    # 1D 데이터 (스토캐스틱 계산용)
    df_1d = get_cached_data(symbol, '1d', lambda s, i, **kw: fetch_binance_klines(s, i, start_date=start), prefix="spot")
    if df_1d is None or len(df_1d) < config['sk'] + config['sks'] + config['sd'] + 20:
        return None

    # 4H MA 계산
    df_4h['ma'] = calc_sma(df_4h['close'], config['ma'])

    # 1D 스토캐스틱 계산
    slow_k, slow_d = calc_stochastic(df_1d, config['sk'], config['sks'], config['sd'])
    df_1d['slow_k'] = slow_k
    df_1d['slow_d'] = slow_d
    df_1d['stoch_signal'] = slow_k > slow_d  # K > D = 매수 시그널

    # 4H 봉에 해당 날짜의 전일 스토캐스틱 시그널 매칭
    df_4h['date'] = df_4h.index.date
    stoch_daily = df_1d['stoch_signal'].copy()
    stoch_daily.index = stoch_daily.index.date

    # 각 4H 봉에 전일 스토캐스틱 매칭 (전일 값 사용)
    df_4h['stoch_buy'] = pd.Series(df_4h['date']).apply(
        lambda d: stoch_daily.shift(1).get(d, np.nan) if d in stoch_daily.shift(1).index else np.nan
    ).values

    # shift(1) 대신 직접 전일 매핑
    stoch_prev = stoch_daily.shift(1)
    date_to_stoch = stoch_prev.to_dict()
    df_4h['stoch_buy'] = df_4h['date'].map(date_to_stoch)

    # MA와 스토캐스틱이 모두 있는 행만 사용
    df_4h = df_4h.dropna(subset=['ma', 'stoch_buy'])
    if len(df_4h) < 10:
        return None

    # 백테스트 (4H 봉 단위)
    equity = initial_capital
    in_position = False
    equity_list = []
    trades = 0

    for i in range(1, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i - 1]

        price = row['open']  # 4H 봉 시가 기준
        ma_val = prev['ma']  # 전 4H 봉의 MA
        stoch_buy = prev['stoch_buy']  # 전일 스토캐스틱

        buy_signal = (price > ma_val) and stoch_buy

        if buy_signal and not in_position:
            fee = equity * fee_rate
            slippage = equity * (SLIPPAGE_PCT / 100)
            equity -= (fee + slippage)
            in_position = True
            trades += 1
        elif not buy_signal and in_position:
            fee = equity * fee_rate
            slippage = equity * (SLIPPAGE_PCT / 100)
            equity -= (fee + slippage)
            in_position = False
            trades += 1

        # 수익률 반영 (4H 봉 시가 -> 시가)
        if in_position:
            price_return = row['open'] / prev['open'] - 1 if prev['open'] > 0 else 0
            equity *= (1 + price_return)

        equity_list.append({'date': df_4h.index[i], 'equity': equity})

    if not equity_list:
        return None

    eq_df = pd.DataFrame(equity_list).set_index('date')
    # 일별로 리샘플링 (마지막 값 사용) - 다른 전략과 합산하기 위해
    eq_daily = eq_df['equity'].resample('D').last().dropna()
    return eq_daily, trades


def backtest_binance_spot(initial_capital=INITIAL_CAPITAL):
    """바이낸스 Spot 통합 백테스트 (균등 분배)"""
    print("\n" + "=" * 80)
    print(f"전략 2: 바이낸스 Spot 백테스트 (전체 {len(SPOT_ALL_COINS)}개 코인)")
    print("=" * 80)

    FEE_RATE = 0.00075  # 0.075%
    per_coin_capital = initial_capital / len(SPOT_ALL_COINS)

    all_equities = {}
    total_trades = 0

    for symbol, config in SPOT_ALL_COINS.items():
        print(f"  [{symbol}] 백테스트 중...")
        result = backtest_binance_spot_single(symbol, config, FEE_RATE, per_coin_capital)

        if result is not None:
            eq, trades = result
            all_equities[symbol] = eq
            total_trades += trades
            print(f"    기간: {eq.index[0].strftime('%Y-%m-%d')} ~ {eq.index[-1].strftime('%Y-%m-%d')}, "
                  f"수익률: {(eq.iloc[-1]/per_coin_capital-1)*100:.1f}%, 거래: {trades}회")
        else:
            print(f"    [!] 데이터 부족으로 스킵")

    if not all_equities:
        print("  [!] 유효한 백테스트 결과 없음")
        return None

    # 통합 자산곡선 (각 코인 자산 합산)
    combined = pd.DataFrame(all_equities)
    combined = combined.ffill()

    # 시작 전 데이터가 없는 코인은 초기 자본 유지
    for col in combined.columns:
        first_valid = combined[col].first_valid_index()
        if first_valid is not None:
            combined.loc[:first_valid, col] = combined.loc[:first_valid, col].fillna(per_coin_capital)
    combined = combined.fillna(per_coin_capital)

    total_equity = combined.sum(axis=1)

    perf = calc_performance(total_equity, f"바이낸스 Spot (시가 기준, 수수료 0.075%, {len(all_equities)}개 코인)")
    perf['trade_count'] = total_trades
    perf['coin_count'] = len(all_equities)
    print_performance(perf)

    return total_equity, perf


# ═══════════════════════════════════════════════════════════════════════════════
# 전략 3: 바이낸스 Futures 숏/롱 백테스트 (전체 코인)
# ═══════════════════════════════════════════════════════════════════════════════


def backtest_futures_short_single(symbol, config, fee_rate=0.0004, initial_capital=INITIAL_CAPITAL):
    """바이낸스 Futures 숏 개별 코인 백테스트 (4H 봉 기준 판단, 고정 마진 방식)"""
    leverage = config['leverage']

    # 4H 데이터 (Futures)
    df_4h = get_cached_data(symbol, '4h', lambda s, i, **kw: fetch_binance_futures_klines(s, i, start_date=datetime(2019, 1, 1)), prefix="futures")
    if df_4h is None or len(df_4h) < config['ma_period'] + 50:
        return None

    # 1D 데이터 (Futures)
    df_1d = get_cached_data(symbol, '1d', lambda s, i, **kw: fetch_binance_futures_klines(s, i, start_date=datetime(2019, 1, 1)), prefix="futures")
    req_len = config['stoch_k_period'] + config['stoch_k_smooth'] + config['stoch_d_period'] + 20
    if df_1d is None or len(df_1d) < req_len:
        return None

    # 4H MA
    df_4h['ma'] = calc_sma(df_4h['close'], config['ma_period'])

    # 1D 스토캐스틱
    slow_k, slow_d = calc_stochastic(df_1d, config['stoch_k_period'], config['stoch_k_smooth'], config['stoch_d_period'])
    df_1d['slow_k'] = slow_k
    df_1d['slow_d'] = slow_d
    df_1d['short_signal'] = slow_k < slow_d  # K < D = 숏 시그널

    # 4H 봉에 전일 스토캐스틱 매칭
    df_4h['date'] = df_4h.index.date
    stoch_daily = df_1d['short_signal'].copy()
    stoch_daily.index = stoch_daily.index.date
    stoch_prev = stoch_daily.shift(1)
    date_to_stoch = stoch_prev.to_dict()
    df_4h['short_signal'] = df_4h['date'].map(date_to_stoch)

    df_4h = df_4h.dropna(subset=['ma', 'short_signal'])
    if len(df_4h) < 10:
        return None

    # 백테스트 (4H 봉 단위, 포지션 내 비복리, 거래 간 복리)
    equity = initial_capital
    in_short = False
    entry_price = 0
    entry_margin = 0
    cum_funding = 0
    equity_list = []
    trades = 0

    # 4H 봉당 펀딩비 (8시간 펀딩비를 4시간 단위로: 4/8 = 0.5)
    funding_per_4h = FUNDING_RATE_8H / 100 * 0.5

    for i in range(1, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i - 1]

        price = row['open']
        ma_val = prev['ma']
        short_sig = prev['short_signal']

        short_condition = (price < ma_val) and short_sig

        if short_condition and not in_short:
            fee = equity * fee_rate * leverage
            slippage = equity * (SLIPPAGE_PCT / 100) * leverage
            equity -= (fee + slippage)
            entry_margin = equity
            entry_price = price
            cum_funding = 0
            in_short = True
            trades += 1
        elif not short_condition and in_short:
            if entry_price > 0:
                price_return = price / entry_price - 1
                trade_pnl = entry_margin * (-price_return) * leverage
                equity = entry_margin + trade_pnl - cum_funding
            fee = max(equity, 0) * fee_rate * leverage
            slippage = max(equity, 0) * (SLIPPAGE_PCT / 100) * leverage
            equity -= (fee + slippage)
            in_short = False
            cum_funding = 0
            trades += 1

        if in_short and entry_price > 0:
            price_return = price / entry_price - 1
            unrealized_pnl = entry_margin * (-price_return) * leverage
            cum_funding += entry_margin * funding_per_4h
            equity = entry_margin + unrealized_pnl - cum_funding

        if equity <= initial_capital * 0.01:
            equity = initial_capital * 0.01
            in_short = False
            break

        equity_list.append({'date': df_4h.index[i], 'equity': max(equity, 0)})

    if not equity_list:
        return None

    eq_df = pd.DataFrame(equity_list).set_index('date')
    eq_daily = eq_df['equity'].resample('D').last().dropna()
    return eq_daily, trades


def backtest_futures_long_single(symbol, config, fee_rate=0.0004, initial_capital=INITIAL_CAPITAL):
    """바이낸스 Futures 롱 개별 코인 백테스트 (4H 봉 기준 판단, 고정 마진 방식)"""
    leverage = config['long_lev']

    # 4H 데이터
    max_ma = max(config['short_ma'], config['long_ma'])
    df_4h = get_cached_data(symbol, '4h', lambda s, i, **kw: fetch_binance_futures_klines(s, i, start_date=datetime(2019, 1, 1)), prefix="futures")
    if df_4h is None or len(df_4h) < max_ma + 50:
        return None

    # 1D 데이터
    df_1d = get_cached_data(symbol, '1d', lambda s, i, **kw: fetch_binance_futures_klines(s, i, start_date=datetime(2019, 1, 1)), prefix="futures")
    max_stoch = max(config['short_sk'] + config['short_sks'] + config['short_sd'],
                    config['long_sk'] + config['long_sks'] + config['long_sd'])
    if df_1d is None or len(df_1d) < max_stoch + 20:
        return None

    # 4H MA (숏필터용 + 롱 진입용)
    df_4h['short_ma'] = calc_sma(df_4h['close'], config['short_ma'])
    df_4h['long_ma'] = calc_sma(df_4h['close'], config['long_ma'])

    # 1D 스토캐스틱 (숏필터용 + 롱 진입용)
    short_k, short_d = calc_stochastic(df_1d, config['short_sk'], config['short_sks'], config['short_sd'])
    long_k, long_d = calc_stochastic(df_1d, config['long_sk'], config['long_sks'], config['long_sd'])

    df_1d['short_filter'] = short_k < short_d
    df_1d['long_signal'] = long_k > long_d

    # 4H 봉에 전일 스토캐스틱 매칭
    df_4h['date'] = df_4h.index.date
    short_filter_daily = df_1d['short_filter'].copy()
    short_filter_daily.index = short_filter_daily.index.date
    long_signal_daily = df_1d['long_signal'].copy()
    long_signal_daily.index = long_signal_daily.index.date

    sf_prev = short_filter_daily.shift(1).to_dict()
    ls_prev = long_signal_daily.shift(1).to_dict()

    df_4h['short_filter'] = df_4h['date'].map(sf_prev)
    df_4h['long_signal'] = df_4h['date'].map(ls_prev)

    df_4h = df_4h.dropna(subset=['short_ma', 'long_ma', 'short_filter', 'long_signal'])
    if len(df_4h) < 10:
        return None

    # 백테스트 (4H 봉 단위, 포지션 내 비복리, 거래 간 복리)
    equity = initial_capital
    in_long = False
    entry_price = 0
    entry_margin = 0
    cum_funding = 0
    equity_list = []
    trades = 0
    funding_per_4h = FUNDING_RATE_8H / 100 * 0.5

    for i in range(1, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i - 1]

        price = row['open']

        short_filter_active = (price < prev['short_ma']) and prev['short_filter']
        long_signal_active = (price > prev['long_ma']) and prev['long_signal']
        long_condition = (not short_filter_active) and long_signal_active

        if long_condition and not in_long:
            fee = equity * fee_rate * leverage
            slippage = equity * (SLIPPAGE_PCT / 100) * leverage
            equity -= (fee + slippage)
            entry_margin = equity
            entry_price = price
            cum_funding = 0
            in_long = True
            trades += 1
        elif not long_condition and in_long:
            if entry_price > 0:
                price_return = price / entry_price - 1
                trade_pnl = entry_margin * price_return * leverage
                equity = entry_margin + trade_pnl - cum_funding
            fee = max(equity, 0) * fee_rate * leverage
            slippage = max(equity, 0) * (SLIPPAGE_PCT / 100) * leverage
            equity -= (fee + slippage)
            in_long = False
            cum_funding = 0
            trades += 1

        if in_long and entry_price > 0:
            price_return = price / entry_price - 1
            unrealized_pnl = entry_margin * price_return * leverage
            cum_funding += entry_margin * funding_per_4h
            equity = entry_margin + unrealized_pnl - cum_funding

        if equity <= initial_capital * 0.01:
            equity = initial_capital * 0.01
            in_long = False
            break

        equity_list.append({'date': df_4h.index[i], 'equity': max(equity, 0)})

    if not equity_list:
        return None

    eq_df = pd.DataFrame(equity_list).set_index('date')
    eq_daily = eq_df['equity'].resample('D').last().dropna()
    return eq_daily, trades


def backtest_binance_futures(initial_capital=INITIAL_CAPITAL):
    """바이낸스 Futures 숏+롱 통합 백테스트"""
    print("\n" + "=" * 80)
    print(f"전략 3: 바이낸스 Futures 숏/롱 백테스트 (숏 {len(FUTURES_SHORT_ALL)}개 + 롱 {len(FUTURES_LONG_ALL)}개 코인)")
    print("=" * 80)

    FEE_RATE = 0.0004  # 0.04%

    # 숏과 롱을 합쳐서 슬롯 수 계산
    all_symbols = set()
    for c in FUTURES_SHORT_ALL:
        all_symbols.add(c['symbol'])
    for c in FUTURES_LONG_ALL:
        all_symbols.add(c['symbol'])

    total_slots = len(all_symbols)
    per_coin_capital = initial_capital / total_slots

    all_equities = {}
    total_trades = 0

    # 숏 백테스트
    print("\n  --- 숏 전략 ---")
    for config in FUTURES_SHORT_ALL:
        symbol = config['symbol']
        print(f"  [숏][{symbol}] 백테스트 중...")
        result = backtest_futures_short_single(symbol, config, FEE_RATE, per_coin_capital)
        if result is not None:
            eq, trades = result
            key = f"{symbol}_short"
            all_equities[key] = eq
            total_trades += trades
            print(f"    수익률: {(eq.iloc[-1]/per_coin_capital-1)*100:.1f}%, 거래: {trades}회")
        else:
            print(f"    [!] 스킵")

    # 롱 백테스트
    print("\n  --- 롱 전략 ---")
    for config in FUTURES_LONG_ALL:
        symbol = config['symbol']
        print(f"  [롱][{symbol}] 백테스트 중...")
        result = backtest_futures_long_single(symbol, config, FEE_RATE, per_coin_capital)
        if result is not None:
            eq, trades = result
            key = f"{symbol}_long"
            all_equities[key] = eq
            total_trades += trades
            print(f"    수익률: {(eq.iloc[-1]/per_coin_capital-1)*100:.1f}%, 거래: {trades}회")
        else:
            print(f"    [!] 스킵")

    if not all_equities:
        print("  [!] 유효한 백테스트 결과 없음")
        return None

    combined = pd.DataFrame(all_equities)
    combined = combined.ffill()
    for col in combined.columns:
        first_valid = combined[col].first_valid_index()
        if first_valid is not None:
            combined.loc[:first_valid, col] = combined.loc[:first_valid, col].fillna(per_coin_capital)
    combined = combined.fillna(per_coin_capital)

    total_equity = combined.sum(axis=1)

    perf = calc_performance(total_equity, f"바이낸스 Futures 숏/롱 (시가 기준, 수수료 0.04%, 펀딩비 적용)")
    perf['trade_count'] = total_trades
    perf['coin_count'] = len(all_equities)
    print_performance(perf)

    return total_equity, perf


# ═══════════════════════════════════════════════════════════════════════════════
# 전략 4: 비트겟 Futures 백테스트
# ═══════════════════════════════════════════════════════════════════════════════

# 비트겟 코인 설정 (bitget_bot.py에서 추출)
BITGET_CONFIGS = {
    'BTCUSDT': {
        'allocation': 0.30,
        'long_ma': 216, 'long_sk': 46, 'long_sks': 37, 'long_sd': 4, 'long_lev': 4,
        'short_ma': 248, 'short_sk': 24, 'short_sks': 20, 'short_sd': 28, 'short_lev': 1,
    },
    'ETHUSDT': {
        'allocation': 0.30,
        'long_ma': 152, 'long_sk': 58, 'long_sks': 23, 'long_sd': 18, 'long_lev': 4,
        'short_ma': 227, 'short_sk': 32, 'short_sks': 43, 'short_sd': 26, 'short_lev': 1,
    },
    'SOLUSDT': {
        'allocation': 0.30,
        'long_ma': 67, 'long_sk': 51, 'long_sks': 20, 'long_sd': 17, 'long_lev': 3,
        'short_ma': 64, 'short_sk': 132, 'short_sks': 25, 'short_sd': 34, 'short_lev': 1,
    },
    'SUIUSDT': {
        'allocation': 0.10,
        'long_ma': 140, 'long_sk': 90, 'long_sks': 40, 'long_sd': 5, 'long_lev': 3,
        'short_ma': 308, 'short_sk': 162, 'short_sks': 68, 'short_sd': 50, 'short_lev': 1,
    },
}


def backtest_bitget_single(symbol, config, fee_rate=0.0004, initial_capital=INITIAL_CAPITAL):
    """비트겟 Futures 개별 코인 백테스트 (4H 봉 기준 판단, 롱 우선)"""

    # 4H 데이터 (바이낸스에서 가져옴 - 실 봇도 바이낸스 시세 참조)
    max_ma = max(config['long_ma'], config['short_ma'])
    df_4h = get_cached_data(symbol, '4h', lambda s, i, **kw: fetch_binance_futures_klines(s, i, start_date=datetime(2019, 1, 1)), prefix="futures")
    if df_4h is None or len(df_4h) < max_ma + 50:
        return None

    # 1D 데이터
    df_1d = get_cached_data(symbol, '1d', lambda s, i, **kw: fetch_binance_futures_klines(s, i, start_date=datetime(2019, 1, 1)), prefix="futures")
    max_stoch = max(config['long_sk'] + config['long_sks'] + config['long_sd'],
                    config['short_sk'] + config['short_sks'] + config['short_sd'])
    if df_1d is None or len(df_1d) < max_stoch + 20:
        return None

    # 4H MA
    df_4h['long_ma'] = calc_sma(df_4h['close'], config['long_ma'])
    df_4h['short_ma'] = calc_sma(df_4h['close'], config['short_ma'])

    # 1D 스토캐스틱
    long_k, long_d = calc_stochastic(df_1d, config['long_sk'], config['long_sks'], config['long_sd'])
    short_k, short_d = calc_stochastic(df_1d, config['short_sk'], config['short_sks'], config['short_sd'])

    df_1d['long_signal'] = long_k > long_d
    df_1d['short_signal'] = short_k < short_d

    # 4H 봉에 전일 스토캐스틱 매칭
    df_4h['date'] = df_4h.index.date
    long_sig_daily = df_1d['long_signal'].copy()
    long_sig_daily.index = long_sig_daily.index.date
    short_sig_daily = df_1d['short_signal'].copy()
    short_sig_daily.index = short_sig_daily.index.date

    ls_prev = long_sig_daily.shift(1).to_dict()
    ss_prev = short_sig_daily.shift(1).to_dict()

    df_4h['long_signal'] = df_4h['date'].map(ls_prev)
    df_4h['short_signal'] = df_4h['date'].map(ss_prev)

    df_4h = df_4h.dropna(subset=['long_ma', 'short_ma', 'long_signal', 'short_signal'])
    if len(df_4h) < 10:
        return None

    # 백테스트 (4H 봉 단위, 포지션 내 비복리, 거래 간 복리, 롱 우선)
    equity = initial_capital
    position = 'CASH'
    entry_price = 0
    entry_margin = 0
    current_lev = 0
    cum_funding = 0
    equity_list = []
    trades = 0
    funding_per_4h = FUNDING_RATE_8H / 100 * 0.5

    for i in range(1, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i - 1]

        price = row['open']

        long_cond = (price > prev['long_ma']) and prev['long_signal']
        short_cond = (price < prev['short_ma']) and prev['short_signal']

        if long_cond:
            new_position = 'LONG'
            new_lev = config['long_lev']
        elif short_cond:
            new_position = 'SHORT'
            new_lev = config['short_lev']
        else:
            new_position = 'CASH'
            new_lev = 0

        if new_position != position:
            if position != 'CASH' and entry_price > 0:
                price_return = price / entry_price - 1
                if position == 'LONG':
                    trade_pnl = entry_margin * price_return * current_lev
                else:
                    trade_pnl = entry_margin * (-price_return) * current_lev
                equity = entry_margin + trade_pnl - cum_funding
                fee = max(equity, 0) * fee_rate * current_lev
                slippage = max(equity, 0) * (SLIPPAGE_PCT / 100) * current_lev
                equity -= (fee + slippage)
                trades += 1

            if new_position != 'CASH':
                current_lev = new_lev
                fee = max(equity, 0) * fee_rate * current_lev
                slippage = max(equity, 0) * (SLIPPAGE_PCT / 100) * current_lev
                equity -= (fee + slippage)
                entry_margin = max(equity, 0)
                entry_price = price
                cum_funding = 0
                trades += 1
            else:
                entry_price = 0
                entry_margin = 0
                current_lev = 0
                cum_funding = 0

            position = new_position

        if position != 'CASH' and entry_price > 0:
            price_return = price / entry_price - 1
            if position == 'LONG':
                unrealized_pnl = entry_margin * price_return * current_lev
            else:
                unrealized_pnl = entry_margin * (-price_return) * current_lev
            cum_funding += entry_margin * funding_per_4h
            equity = entry_margin + unrealized_pnl - cum_funding

        if equity <= initial_capital * 0.01:
            equity = initial_capital * 0.01
            position = 'CASH'
            break

        equity_list.append({'date': df_4h.index[i], 'equity': max(equity, 0)})

    if not equity_list:
        return None

    eq_df = pd.DataFrame(equity_list).set_index('date')
    eq_daily = eq_df['equity'].resample('D').last().dropna()
    return eq_daily, trades


def backtest_bitget_futures(initial_capital=INITIAL_CAPITAL):
    """비트겟 Futures 통합 백테스트"""
    print("\n" + "=" * 80)
    print("전략 4: 비트겟 Futures 롱/숏 백테스트 (BTC, ETH, SOL, SUI)")
    print("=" * 80)

    FEE_RATE = 0.0004  # 0.04%

    all_equities = {}
    total_trades = 0

    for symbol, config in BITGET_CONFIGS.items():
        coin_capital = initial_capital * config['allocation']
        print(f"  [{symbol}] 배분: ${coin_capital:,.0f} ({config['allocation']*100:.0f}%), 백테스트 중...")

        result = backtest_bitget_single(symbol, config, FEE_RATE, coin_capital)

        if result is not None:
            eq, trades = result
            all_equities[symbol] = eq
            total_trades += trades
            ret = (eq.iloc[-1] / coin_capital - 1) * 100
            print(f"    기간: {eq.index[0].strftime('%Y-%m-%d')} ~ {eq.index[-1].strftime('%Y-%m-%d')}")
            print(f"    수익률: {ret:.1f}%, 거래: {trades}회")
        else:
            print(f"    [!] 스킵")

    if not all_equities:
        print("  [!] 유효한 결과 없음")
        return None

    combined = pd.DataFrame(all_equities)
    combined = combined.ffill()

    for symbol, config in BITGET_CONFIGS.items():
        if symbol in combined.columns:
            coin_cap = initial_capital * config['allocation']
            first_valid = combined[symbol].first_valid_index()
            if first_valid is not None:
                combined.loc[:first_valid, symbol] = combined.loc[:first_valid, symbol].fillna(coin_cap)

    # 누락된 값 채우기
    for symbol, config in BITGET_CONFIGS.items():
        if symbol not in combined.columns:
            combined[symbol] = initial_capital * config['allocation']
    combined = combined.ffill().bfill()

    total_equity = combined.sum(axis=1)

    perf = calc_performance(total_equity, "비트겟 Futures 롱/숏 (시가 기준, 수수료 0.04%, 펀딩비 적용)")
    perf['trade_count'] = total_trades
    perf['coin_count'] = len(all_equities)
    print_performance(perf)

    return total_equity, perf


# ═══════════════════════════════════════════════════════════════════════════════
# 차트 생성
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results):
    """모든 전략 결과 차트 생성"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    # 1. 자산곡선 (절대값)
    ax1 = axes[0]
    for idx, (name, eq, perf) in enumerate(results):
        if eq is not None:
            ax1.plot(eq.index, eq.values, label=f"{name}", color=colors[idx], linewidth=1.2)

    ax1.set_title('Strategy Equity Curves', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # 2. 드로다운
    ax2 = axes[1]
    for idx, (name, eq, perf) in enumerate(results):
        if eq is not None:
            peak = eq.cummax()
            dd = (eq - peak) / peak * 100
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color=colors[idx], label=name)

    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # 3. 성과 비교 테이블 (바 차트)
    ax3 = axes[2]
    metrics = ['CAGR (%)', 'MDD (%)', 'Sharpe']
    valid_results = [(name, perf) for name, _, perf in results if perf]

    if valid_results:
        x = np.arange(len(metrics))
        width = 0.8 / len(valid_results)

        for idx, (name, perf) in enumerate(valid_results):
            values = [perf.get('cagr', 0), perf.get('mdd', 0), perf.get('sharpe', 0)]
            bars = ax3.bar(x + idx * width, values, width, label=name, color=colors[idx], alpha=0.8)
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        ax3.set_xticks(x + width * len(valid_results) / 2)
        ax3.set_xticklabels(metrics)
        ax3.set_title('Performance Comparison', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, 'backtest_results.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n차트 저장: {chart_path}")


def save_summary(results):
    """성과 요약 저장"""
    summary = []
    for name, eq, perf in results:
        if perf:
            summary.append(perf)

    summary_path = os.path.join(RESULTS_DIR, 'backtest_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"요약 저장: {summary_path}")

    # 콘솔에 비교표 출력
    print("\n" + "=" * 100)
    print("  전략 성과 비교표")
    print("=" * 100)
    header = f"{'전략':<45} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'거래수':>8} {'최종자산':>12}"
    print(header)
    print("-" * 100)
    for perf in summary:
        line = (f"{perf['name'][:44]:<45} "
                f"{perf.get('cagr',0):>7.1f}% "
                f"{perf.get('mdd',0):>7.1f}% "
                f"{perf.get('sharpe',0):>8.3f} "
                f"{perf.get('sortino',0):>8.3f} "
                f"{perf.get('trade_count',0):>8} "
                f"${perf.get('final_equity',0):>10,.2f}")
        print(line)
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  자동매매 전략 통합 백테스트 시스템")
    print("=" * 80)
    print(f"  초기 자본금: ${INITIAL_CAPITAL:,}")
    print(f"  슬리피지: {SLIPPAGE_PCT}%")
    print(f"  8H 펀딩비: {FUNDING_RATE_8H}%")
    print(f"  결과 저장: {RESULTS_DIR}")
    print("=" * 80)

    results = []

    # 전략 1: TQQQ
    try:
        r = backtest_tqqq()
        if r:
            eq, perf = r
            results.append(("TQQQ Sniper", eq, perf))
    except Exception as e:
        print(f"  [!] TQQQ 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    # 전략 2: 바이낸스 Spot
    try:
        r = backtest_binance_spot()
        if r:
            eq, perf = r
            results.append(("Binance Spot", eq, perf))
    except Exception as e:
        print(f"  [!] 바이낸스 Spot 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    # 전략 3: 바이낸스 Futures
    try:
        r = backtest_binance_futures()
        if r:
            eq, perf = r
            results.append(("Binance Futures", eq, perf))
    except Exception as e:
        print(f"  [!] 바이낸스 Futures 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    # 전략 4: 비트겟 Futures
    try:
        r = backtest_bitget_futures()
        if r:
            eq, perf = r
            results.append(("Bitget Futures", eq, perf))
    except Exception as e:
        print(f"  [!] 비트겟 Futures 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    # 결과 출력 및 저장
    if results:
        plot_results(results)
        save_summary(results)
    else:
        print("\n  [!] 백테스트 결과가 없습니다.")

    print("\n백테스트 완료!")


if __name__ == '__main__':
    main()
