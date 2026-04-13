"""
================================================================================
봇 전략 백테스트: bitget_major_bot.py & binance_futures_bot.py
================================================================================
두 봇의 실제 파라미터로 과거 데이터 백테스트 실행
- 4H / 1H 시가 기반 진입/청산 비교 (MA는 4H봉, 스토캐스틱은 1D)
- 수수료 0.04%, 슬리피지 0.05%, 펀딩비 0.01%/8h (노셔널 기준)
- 시계열 포트폴리오 백테스트: 각 시점의 활성 코인 수로 동적 배분
- 실제 봇 로직 반영: 총자산 / 활성코인수 = 코인당 배분금
================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
import requests
from datetime import datetime

warnings.filterwarnings('ignore')

# ==========================================
# 비용 설정
# ==========================================
FEE_RATE = 0.0004          # 0.04%
SLIPPAGE_PCT = 0.0005      # 0.05%
FUNDING_RATE_8H = 0.0001   # 0.01% per 8h (폴백용)
USE_REAL_FUNDING = True    # True: 바이낸스 실제 펀딩비 사용

MIN_DATA_DAYS = 365

# 포트폴리오 초기 자본
BITGET_CAPITAL = 10000.0
BINANCE_CAPITAL = 100000.0

SAVE_DIR = os.path.expanduser("~/Downloads")
CACHE_DIR = os.path.expanduser("~/binance_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# Bitget Major Bot 설정 (6코인)
# ==========================================
BITGET_CONFIGS = [
    {'symbol': 'BTCUSDT', 'long_ma': 350, 'long_sk': 36, 'long_sks': 32, 'long_sd': 10, 'long_lev': 5,
     'short_ma': 254, 'short_sk': 27, 'short_sks': 23, 'short_sd': 19, 'short_lev': 1},
    {'symbol': 'ETHUSDT', 'long_ma': 322, 'long_sk': 54, 'long_sks': 10, 'long_sd': 36, 'long_lev': 5,
     'short_ma': 220, 'short_sk': 31, 'short_sks': 44, 'short_sd': 26, 'short_lev': 2},
    {'symbol': 'XRPUSDT', 'long_ma': 107, 'long_sk': 14, 'long_sks': 13, 'long_sd': 23, 'long_lev': 5,
     'short_ma': 269, 'short_sk': 121, 'short_sks': 35, 'short_sd': 47, 'short_lev': 1},
    {'symbol': 'SOLUSDT', 'long_ma': 73, 'long_sk': 33, 'long_sks': 16, 'long_sd': 38, 'long_lev': 4,
     'short_ma': 314, 'short_sk': 37, 'short_sks': 34, 'short_sd': 44, 'short_lev': 1},
    {'symbol': 'DOGEUSDT', 'long_ma': 31, 'long_sk': 48, 'long_sks': 50, 'long_sd': 17, 'long_lev': 2,
     'short_ma': 250, 'short_sk': 36, 'short_sks': 15, 'short_sd': 40, 'short_lev': 1},
    {'symbol': 'ADAUSDT', 'long_ma': 296, 'long_sk': 19, 'long_sks': 53, 'long_sd': 15, 'long_lev': 3,
     'short_ma': 80, 'short_sk': 31, 'short_sks': 77, 'short_sd': 46, 'short_lev': 1},
]

# ==========================================
# Binance Futures Bot 설정 로드
# ==========================================
def load_binance_configs():
    bot_path = os.path.join(SAVE_DIR, 'binance_bot-5.py')
    if not os.path.exists(bot_path):
        print(f"  {bot_path} 파일을 찾을 수 없습니다.")
        return [], []
    with open(bot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    short_start = content.index('SHORT_TRADING_CONFIGS = [')
    short_end = content.index(']', short_start) + 1
    long_start = content.index('LONG_TRADING_CONFIGS = [')
    long_end = content.index(']', long_start) + 1
    local_ns = {}
    exec(content[short_start:short_end], {}, local_ns)
    exec(content[long_start:long_end], {}, local_ns)
    short_configs = local_ns.get('SHORT_TRADING_CONFIGS', [])
    long_configs = local_ns.get('LONG_TRADING_CONFIGS', [])
    print(f"  Binance 설정 로드: 숏 {len(short_configs)}개, 롱 {len(long_configs)}개")
    return short_configs, long_configs


# ==========================================
# 데이터 수집
# ==========================================
def get_funding_rates(symbol, silent=False):
    """바이낸스 실제 펀딩비 히스토리 다운로드 (캐시 지원)"""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_funding.csv")
    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mod_time).days < 1:
            return pd.read_csv(cache_file, parse_dates=['timestamp'])

    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    start_time = int(pd.Timestamp('2019-01-01').timestamp() * 1000)
    for _ in range(50):
        params = {'symbol': symbol, 'startTime': start_time, 'limit': 1000}
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(2); continue
            if response.status_code != 200:
                break
            data = response.json()
            if not data or not isinstance(data, list):
                break
            all_data.extend(data)
            if len(data) < 1000:
                break
            start_time = data[-1]['fundingTime'] + 1
            time.sleep(0.1)
        except:
            time.sleep(1); continue
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['funding_rate'] = df['fundingRate'].astype(float)
    df = df[['timestamp', 'funding_rate']].sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


def map_funding_to_4h(df_4h, df_funding):
    """펀딩비를 4H 캔들에 매핑"""
    fallback = FUNDING_RATE_8H * 0.5
    if df_funding.empty:
        df_4h = df_4h.copy()
        df_4h['funding_rate_4h'] = fallback
        return df_4h
    df_4h = df_4h.copy()
    merged = pd.merge_asof(
        df_4h.sort_values('timestamp'),
        df_funding.sort_values('timestamp').rename(columns={'funding_rate': 'funding_rate_8h'}),
        on='timestamp', direction='backward'
    )
    merged['funding_rate_4h'] = merged['funding_rate_8h'] * 0.5
    merged['funding_rate_4h'] = merged['funding_rate_4h'].fillna(fallback)
    return merged


def get_binance_futures_klines(symbol, interval='4h', silent=False):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}_full.csv")
    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mod_time).days < 7:
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            return df
    if not silent:
        print(f"  다운로드: {symbol} ({interval})...")
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    for _ in range(100):
        params = {'symbol': symbol, 'interval': interval, 'limit': 1500, 'endTime': end_time}
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(1)
                continue
            data = response.json()
            if not data or not isinstance(data, list):
                break
            all_data = data + all_data
            if len(data) < 1500:
                break
            end_time = data[0][0] - 1
            time.sleep(0.1)
        except Exception as e:
            if not silent:
                print(f"    API 오류: {e}")
            time.sleep(1)
            continue
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


def prepare_coin_data(symbol, silent=False):
    """코인 데이터 준비 (1H + 4H + 1D)"""
    df_1h = get_binance_futures_klines(symbol, '1h', silent)
    df_4h = get_binance_futures_klines(symbol, '4h', silent)
    df_daily = get_binance_futures_klines(symbol, '1d', silent)
    if df_1h.empty or df_4h.empty or df_daily.empty:
        return None
    if len(df_1h) < 800 or len(df_4h) < 200 or len(df_daily) < 100:
        return None
    start_date = df_4h['timestamp'].min()
    end_date = df_4h['timestamp'].max()
    days = (end_date - start_date).days
    if days < MIN_DATA_DAYS:
        return None
    # 실제 펀딩비 로드
    if USE_REAL_FUNDING:
        df_funding = get_funding_rates(symbol, silent)
        df_4h = map_funding_to_4h(df_4h, df_funding)
    else:
        df_4h = df_4h.copy()
        df_4h['funding_rate_4h'] = FUNDING_RATE_8H * 0.5
    return {
        'symbol': symbol, 'df_1h': df_1h, 'df_4h': df_4h, 'df_daily': df_daily,
        'start_date': start_date, 'end_date': end_date, 'days': days
    }


# ==========================================
# 지표 계산
# ==========================================
def calculate_stochastic(df, k_period, k_smooth, d_period):
    lowest = df['low'].rolling(window=k_period).min()
    highest = df['high'].rolling(window=k_period).max()
    fast_k = ((df['close'] - lowest) / (highest - lowest)) * 100
    fast_k = fast_k.replace([np.inf, -np.inf], np.nan)
    slow_k = fast_k.rolling(window=k_smooth).mean()
    slow_d = slow_k.rolling(window=d_period).mean()
    return slow_k, slow_d


def prepare_signals(df_base, df_4h, df_daily, ma_period, stoch_k, stoch_ks, stoch_d, use_1h=False):
    """
    진입/청산 신호 DataFrame 생성
    - use_1h=False: 4H 시가 기반 (df_base=df_4h), MA도 4H
    - use_1h=True:  1H 시가 기반 (df_base=df_1h), MA는 4H에서 forward-fill
    """
    # 4H MA 계산
    df_4h_ma = df_4h[['timestamp', 'close']].copy().sort_values('timestamp')
    df_4h_ma['ma'] = df_4h_ma['close'].rolling(window=ma_period).mean()

    base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if 'funding_rate_4h' in df_base.columns:
        base_cols.append('funding_rate_4h')

    if use_1h:
        # 1H 기반: 4H MA를 1H에 forward-fill
        df = df_base[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy().sort_values('timestamp')
        df_4h_ma_clean = df_4h_ma[['timestamp', 'ma']].dropna()
        df = pd.merge_asof(df, df_4h_ma_clean, on='timestamp', direction='backward')
        # 1H에는 4H 펀딩비를 forward-fill
        if 'funding_rate_4h' in df_4h.columns:
            df_4h_fr = df_4h[['timestamp', 'funding_rate_4h']].copy().sort_values('timestamp')
            df = pd.merge_asof(df, df_4h_fr, on='timestamp', direction='backward')
    else:
        # 4H 기반: MA 직접 사용
        df = df_base[base_cols].copy().sort_values('timestamp')
        df['ma'] = df['close'].rolling(window=ma_period).mean()

    # 1D 스토캐스틱 (전일값) 매핑
    df['date'] = df['timestamp'].dt.date
    df_d = df_daily.copy()
    slow_k, slow_d = calculate_stochastic(df_d, stoch_k, stoch_ks, stoch_d)
    df_d['slow_k'] = slow_k
    df_d['slow_d'] = slow_d
    df_d['date'] = df_d['timestamp'].dt.date
    df_d['prev_slow_k'] = df_d['slow_k'].shift(1)
    df_d['prev_slow_d'] = df_d['slow_d'].shift(1)
    stoch_map = df_d.dropna(subset=['prev_slow_k', 'prev_slow_d']).set_index('date')[
        ['prev_slow_k', 'prev_slow_d']].to_dict('index')
    df['prev_slow_k'] = df['date'].map(lambda x: stoch_map.get(x, {}).get('prev_slow_k'))
    df['prev_slow_d'] = df['date'].map(lambda x: stoch_map.get(x, {}).get('prev_slow_d'))
    return df


def calculate_portfolio_performance(equity_curve, timestamps):
    eq = np.array(equity_curve, dtype=float)
    ts = np.array(timestamps)
    if len(eq) < 2:
        return None
    total_days = (ts[-1] - ts[0]).days
    if total_days < 30:
        return None
    eq = np.maximum(eq, 0.01)
    total_return = eq[-1] / eq[0]
    years = total_days / 365.25
    cagr = (total_return ** (1 / years) - 1) * 100
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak * 100
    mdd = drawdown.min()
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) > 10 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 6)
    else:
        sharpe = 0
    return {
        'cagr': cagr, 'mdd': mdd, 'sharpe': sharpe,
        'total_return': (total_return - 1) * 100,
        'days': total_days,
        'start': str(pd.Timestamp(ts[0]).date()),
        'end': str(pd.Timestamp(ts[-1]).date()),
        'final_equity': eq[-1],
        'initial_equity': eq[0],
    }


# ==========================================
# Bitget 포트폴리오 시뮬레이션 (범용)
# ==========================================
def run_bitget_portfolio(use_1h, coin_data_cache, end_date=None):
    tf_label = "1H" if use_1h else "4H"
    funding_per_bar = FUNDING_RATE_8H / 8 if use_1h else FUNDING_RATE_8H * 0.5

    coin_signals = {}
    for config in BITGET_CONFIGS:
        symbol = config['symbol']
        data = coin_data_cache.get(symbol)
        if data is None:
            continue

        df_base = data['df_1h'] if use_1h else data['df_4h']
        df_4h = data['df_4h']
        df_daily = data['df_daily']

        df_long = prepare_signals(df_base, df_4h, df_daily, config['long_ma'],
                                  config['long_sk'], config['long_sks'], config['long_sd'], use_1h=use_1h)
        df_short = prepare_signals(df_base, df_4h, df_daily, config['short_ma'],
                                   config['short_sk'], config['short_sks'], config['short_sd'], use_1h=use_1h)

        bt_cols = ['timestamp', 'open', 'high', 'low', 'close', 'ma',
                   'prev_slow_k', 'prev_slow_d']
        if 'funding_rate_4h' in df_long.columns:
            bt_cols.append('funding_rate_4h')
        df_bt = df_long[bt_cols].copy()
        df_bt.rename(columns={'ma': 'ma_long', 'prev_slow_k': 'long_k', 'prev_slow_d': 'long_d'}, inplace=True)
        df_bt['ma_short'] = df_short['ma'].values
        df_bt['short_k'] = df_short['prev_slow_k'].values
        df_bt['short_d'] = df_short['prev_slow_d'].values
        df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d', 'ma_short', 'short_k', 'short_d'])

        if end_date is not None:
            df_bt = df_bt[df_bt['timestamp'] <= end_date]

        df_bt = df_bt.set_index('timestamp').sort_index()
        if len(df_bt) >= 100:
            coin_signals[symbol] = {'df': df_bt, 'long_lev': config['long_lev'], 'short_lev': config['short_lev']}

    if not coin_signals:
        return None

    all_ts = set()
    for s_data in coin_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    timeline = sorted(all_ts)

    cash = BITGET_CAPITAL
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
            long_signal = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
            short_signal = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
            pos = positions.get(symbol)

            if long_signal:
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
                        cost = alloc * (FEE_RATE + SLIPPAGE_PCT) * long_lev
                        margin = alloc - cost
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'long', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': long_lev}
                            total_trades += 1
                            long_trades += 1
            elif short_signal and (pos is None or pos.get('side') != 'long'):
                if pos is None:
                    total_eq = cash + sum(p['margin'] for p in positions.values())
                    alloc = min(total_eq / n_active, cash * 0.995)
                    if alloc > 1:
                        cost = alloc * (FEE_RATE + SLIPPAGE_PCT) * short_lev
                        margin = alloc - cost
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'short', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': short_lev}
                            total_trades += 1
                            short_trades += 1
            else:
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

            pos = positions.get(symbol)
            if pos and ts in df.index:
                # 실제 펀딩비 적용
                fr = curr['funding_rate_4h'] if 'funding_rate_4h' in df.columns else funding_per_bar
                if pos['side'] == 'long':
                    pos['cum_funding'] += pos['margin'] * pos['lev'] * fr       # 양수→비용
                else:
                    pos['cum_funding'] += pos['margin'] * pos['lev'] * (-fr)    # 양수→수익
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

        unrealized_total = 0
        for sym, pos in positions.items():
            df = coin_signals[sym]['df']
            if ts in df.index:
                close_p = df.loc[ts, 'close']
                if pos['side'] == 'long':
                    pr = close_p / pos['entry_price'] - 1
                    val = pos['margin'] + pos['margin'] * pr * pos['lev'] - pos['cum_funding']
                else:
                    pr = close_p / pos['entry_price'] - 1
                    val = pos['margin'] + pos['margin'] * (-pr) * pos['lev'] - pos['cum_funding']
                unrealized_total += max(val, 0)
            else:
                unrealized_total += pos['margin']
        equity_curve.append(cash + unrealized_total)
        eq_timestamps.append(ts)

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
    positions.clear()

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf is None:
        return None
    perf['trades'] = total_trades
    perf['long_trades'] = long_trades
    perf['short_trades'] = short_trades
    perf['coins'] = len(coin_signals)
    perf['equity_curve'] = equity_curve
    perf['timestamps'] = eq_timestamps
    return perf


# ==========================================
# Binance 포트폴리오 시뮬레이션 (범용)
# ==========================================
def run_binance_portfolio(use_1h, short_configs, long_configs, coin_data_cache, end_date=None):
    tf_label = "1H" if use_1h else "4H"
    funding_per_bar = FUNDING_RATE_8H / 8 if use_1h else FUNDING_RATE_8H * 0.5

    # 숏 신호 준비
    short_signals = {}
    for config in short_configs:
        symbol = config['symbol']
        data = coin_data_cache.get(symbol)
        if data is None:
            continue
        df_base = data['df_1h'] if use_1h else data['df_4h']
        df_sig = prepare_signals(df_base, data['df_4h'], data['df_daily'],
                                 config['ma_period'], config['stoch_k_period'],
                                 config['stoch_k_smooth'], config['stoch_d_period'], use_1h=use_1h)
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
        df_base = data['df_1h'] if use_1h else data['df_4h']
        df_4h = data['df_4h']
        df_daily = data['df_daily']
        df_sf = prepare_signals(df_base, df_4h, df_daily, config['short_ma'],
                                config['short_sk'], config['short_sks'], config['short_sd'], use_1h=use_1h)
        df_ls = prepare_signals(df_base, df_4h, df_daily, config['long_ma'],
                                config['long_sk'], config['long_sks'], config['long_sd'], use_1h=use_1h)
        sf_cols = ['timestamp', 'open', 'high', 'low', 'close', 'ma',
                   'prev_slow_k', 'prev_slow_d']
        if 'funding_rate_4h' in df_sf.columns:
            sf_cols.append('funding_rate_4h')
        df_bt = df_sf[sf_cols].copy()
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

    print(f"    [{tf_label}] 숏 {len(short_signals)}코인, 롱 {len(long_signals)}코인")

    # 타임라인
    all_ts = set()
    for s_data in short_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    for s_data in long_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    if not all_ts:
        return None
    timeline = sorted(all_ts)
    print(f"    [{tf_label}] 타임라인: {len(timeline)}개 봉 ({timeline[0].date()} ~ {timeline[-1].date()})")

    # 시뮬레이션
    cash = BINANCE_CAPITAL
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
            print(f"    [{tf_label}] {pct:.0f}% ({ts.date()}) ${port_eq:,.0f} 포지션={len(positions)}")

        active_short = [s for s, d in short_signals.items() if ts in d['df'].index]
        active_long = [s for s, d in long_signals.items() if ts in d['df'].index]
        n_active = len(set(active_short) | set(active_long))

        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue

        total_eq_for_alloc = cash + sum(p['margin'] for p in positions.values())
        max_per_slot = total_eq_for_alloc / n_active

        # 숏
        for symbol in active_short:
            df = short_signals[symbol]['df']
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
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
                            positions[symbol] = {'side': 'short', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': lev, 'strategy': 'short'}
                            total_trades += 1
                            short_trade_count += 1
            else:
                if pos and pos['strategy'] == 'short':
                    pr = op / pos['entry_price'] - 1
                    pnl = pos['margin'] * (-pr) * pos['lev']
                    settled = pos['margin'] + pnl - pos['cum_funding']
                    exit_not = abs(pos['margin'] * pos['lev'] * (op / pos['entry_price']))
                    settled -= exit_not * (FEE_RATE + SLIPPAGE_PCT)
                    cash += max(settled, 0)
                    del positions[symbol]
                    total_trades += 1

        # 롱
        for symbol in active_long:
            df = long_signals[symbol]['df']
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            curr_idx = df.index.get_loc(ts)
            if curr_idx == 0:
                continue
            prev = df.iloc[curr_idx - 1]
            op = curr['open']
            lev = long_signals[symbol]['lev']
            sf_active = (op < prev['sf_ma']) and (prev['sf_k'] < prev['sf_d'])
            ls_active = (op > prev['ls_ma']) and (prev['ls_k'] > prev['ls_d'])
            should_enter = (not sf_active) and ls_active
            should_close = sf_active or (not ls_active)
            pos = positions.get(symbol)
            if should_enter:
                if pos is None and cash > 1:
                    alloc = min(max_per_slot, cash * 0.995)
                    if alloc > 1:
                        margin = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                        if margin > 0:
                            cash -= alloc
                            positions[symbol] = {'side': 'long', 'entry_price': op, 'margin': margin, 'cum_funding': 0, 'lev': lev, 'strategy': 'long'}
                            total_trades += 1
                            long_trade_count += 1
            elif should_close:
                if pos and pos['strategy'] == 'long':
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
            if pos['strategy'] == 'short' and sym in short_signals:
                df = short_signals[sym]['df']
            elif pos['strategy'] == 'long' and sym in long_signals:
                df = long_signals[sym]['df']
            else:
                continue
            if ts not in df.index:
                continue
            curr = df.loc[ts]
            # 실제 펀딩비 적용
            fr = curr['funding_rate_4h'] if 'funding_rate_4h' in df.columns else funding_per_bar
            if pos['side'] == 'long':
                pos['cum_funding'] += pos['margin'] * pos['lev'] * fr
            else:
                pos['cum_funding'] += pos['margin'] * pos['lev'] * (-fr)
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


# ==========================================
# 메인
# ==========================================
def main():
    print("=" * 70)
    print("  봇 전략 백테스트: 4H vs 1H 리밸런싱 비교")
    print("=" * 70)
    print(f"  비용: 수수료 {FEE_RATE*100:.2f}%, 슬리피지 {SLIPPAGE_PCT*100:.2f}%, "
          f"펀딩 {FUNDING_RATE_8H*100:.3f}%/8h")
    print(f"  배분방식: 총자산 / 활성코인수 (동적 배분)")
    start_time = time.time()

    # 데이터 공통 캐시
    print(f"\n  --- Bitget 데이터 로드 ---")
    bitget_cache = {}
    for config in BITGET_CONFIGS:
        symbol = config['symbol']
        print(f"  [{symbol}]...")
        data = prepare_coin_data(symbol, silent=True)
        if data:
            bitget_cache[symbol] = data
            print(f"    1H: {len(data['df_1h'])}봉, 4H: {len(data['df_4h'])}봉")

    short_configs, long_configs = load_binance_configs()

    print(f"\n  --- Binance 데이터 로드 ---")
    binance_cache = {}
    all_symbols = set(c['symbol'] for c in short_configs) | set(c['symbol'] for c in long_configs)
    for idx, symbol in enumerate(sorted(all_symbols), 1):
        if idx % 50 == 0 or idx == 1:
            print(f"  진행: {idx}/{len(all_symbols)}...")
        data = prepare_coin_data(symbol, silent=True)
        if data:
            binance_cache[symbol] = data
    print(f"  데이터 로드 완료: {len(binance_cache)}/{len(all_symbols)} 코인")

    # 공통 end_date 결정 (4H 데이터의 최소 end_date 기준)
    all_4h_ends = []
    for data in bitget_cache.values():
        all_4h_ends.append(data['df_4h']['timestamp'].max())
    for data in binance_cache.values():
        all_4h_ends.append(data['df_4h']['timestamp'].max())
    end_date = min(all_4h_ends) if all_4h_ends else None
    print(f"\n  공통 종료일: {end_date}")

    # ── Bitget 4H vs 1H ──
    print(f"\n{'=' * 70}")
    print(f"  Bitget 포트폴리오 (초기자본 ${BITGET_CAPITAL:,.0f})")
    print(f"{'=' * 70}")

    print(f"  [4H] 시뮬레이션...")
    bitget_4h = run_bitget_portfolio(use_1h=False, coin_data_cache=bitget_cache, end_date=end_date)
    print(f"  [1H] 시뮬레이션...")
    bitget_1h = run_bitget_portfolio(use_1h=True, coin_data_cache=bitget_cache, end_date=end_date)

    # ── Binance 4H vs 1H ──
    print(f"\n{'=' * 70}")
    print(f"  Binance 포트폴리오 (초기자본 ${BINANCE_CAPITAL:,.0f})")
    print(f"{'=' * 70}")

    print(f"\n  [4H] 시뮬레이션...")
    binance_4h = run_binance_portfolio(use_1h=False, short_configs=short_configs, long_configs=long_configs,
                                       coin_data_cache=binance_cache, end_date=end_date)
    print(f"\n  [1H] 시뮬레이션...")
    binance_1h = run_binance_portfolio(use_1h=True, short_configs=short_configs, long_configs=long_configs,
                                       coin_data_cache=binance_cache, end_date=end_date)

    # ── 비교표 ──
    print(f"\n{'=' * 70}")
    print(f"  4H vs 1H 리밸런싱 비교표")
    print(f"{'=' * 70}")
    print(f"  {'봇':<25} {'TF':>4} {'CAGR':>9} {'MDD':>9} {'Sharpe':>8} {'거래':>8} {'기간':>24}")
    print(f"  {'─' * 87}")

    for label, p4h, p1h in [
        ("Bitget (6코인)", bitget_4h, bitget_1h),
        ("Binance (숏+롱)", binance_4h, binance_1h),
    ]:
        for tf, p in [("4H", p4h), ("1H", p1h)]:
            if p:
                print(f"  {label:<25} {tf:>4} {p['cagr']:>8.1f}% {p['mdd']:>8.1f}% "
                      f"{p['sharpe']:>7.3f} {p['trades']:>7}회 {p['start']}~{p['end']}")
            else:
                print(f"  {label:<25} {tf:>4} {'N/A':>9} {'N/A':>9} {'N/A':>8} {'N/A':>8}")
        print(f"  {'─' * 87}")

    # CSV 저장
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(SAVE_DIR, f"backtest_bots_results_{ts_str}.csv")
    rows = []
    for label, bot, p4h, p1h in [
        ("Bitget_Major", "Portfolio(Long+Short)", bitget_4h, bitget_1h),
        ("Binance_Futures", "Portfolio(Short+Long)", binance_4h, binance_1h),
    ]:
        for tf, p in [("4H", p4h), ("1H", p1h)]:
            if p:
                rows.append({
                    'Bot': label, 'Timeframe': tf, 'Strategy': bot,
                    'Initial_Capital': round(p['initial_equity'], 2),
                    'Final_Capital': round(p['final_equity'], 2),
                    'Total_Return(%)': round(p['total_return'], 2),
                    'CAGR(%)': round(p['cagr'], 2),
                    'MDD(%)': round(p['mdd'], 2),
                    'Sharpe': round(p['sharpe'], 3),
                    'Trades': p['trades'],
                    'Long_Trades': p.get('long_trades', 0),
                    'Short_Trades': p.get('short_trades', 0),
                    'Coins': p.get('coins', p.get('short_coins', 0) + p.get('long_coins', 0)),
                    'Days': p['days'],
                    'Start': p['start'],
                    'End': p['end'],
                })
    if rows:
        pd.DataFrame(rows).to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n  결과 저장: {filepath}")

    elapsed = time.time() - start_time
    print(f"\n  총 소요시간: {elapsed/60:.1f}분")
    print("  백테스트 완료!")


if __name__ == '__main__':
    main()
