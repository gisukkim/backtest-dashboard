"""
================================================================================
Binance Futures 메이저 코인 숏+롱 독립 파라미터 최적화 (숏 우선)
================================================================================
대상: BTC, ETH, XRP, SOL, TRX, DOGE, BCH, ADA (8개)

전략 로직 (숏 우선):
  숏 시그널: 시가 < MA_short(4H) AND K_short < D_short (1D, 전일) → 숏 진입
  롱 시그널: 시가 > MA_long(4H) AND K_long > D_long (1D, 전일)
             숏 시그널 없을 때만 롱 진입 (롱은 보조)

최적화 방식: 3단계 독립 최적화
  Stage 1: 숏 파라미터만 최적화 (롱 없이 순수 숏)
  Stage 2: 롱 파라미터 최적화 (숏 파라미터 고정, 결합 백테스트)
  Stage 3: 결합 검증 (숏만 vs 숏+롱 비교)

비용 모델:
  - 수수료: 0.04% (노셔널 기준)
  - 슬리피지: 0.05% (노셔널 기준)
  - 펀딩비: 0.01% / 8h (노셔널 = 마진 × 레버리지 기준)
  - 포지션 내 비복리, 거래 간 복리
  - 강제청산: 롱=entry*(1-1/lev), 숏=entry*(1+1/lev)
================================================================================
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import os
import time
import warnings
import requests
from datetime import datetime

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 설정
# ==========================================

# 대상 코인
TARGET_COINS = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT',
                'TRXUSDT', 'DOGEUSDT', 'BCHUSDT', 'ADAUSDT']

# 1단계 스크리닝 설정
PHASE1_TRIALS = 50

# 2단계 파인튜닝 설정
PHASE2_TRIALS = 300

# 수수료/비용 설정
FEE_RATE = 0.0004          # 0.04%
SLIPPAGE_PCT = 0.0005      # 0.05%
FUNDING_RATE_8H = 0.0001   # 0.01% per 8h
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5  # 4H 봉당 펀딩비 = 0.00005

MIN_DATA_DAYS = 365

# 파라미터 탐색 범위
PARAM_RANGES = {
    'ma_range': (20, 350),
    'stoch_k_range': (14, 150),
    'stoch_smooth_range': (5, 80),
    'stoch_d_range': (3, 50),
    'leverage_range': (1, 5),
}

# 저장 경로
SAVE_DIR = os.path.expanduser("~/Downloads")
CACHE_DIR = os.path.expanduser("~/binance_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ==========================================
# 데이터 수집 함수
# ==========================================

def get_binance_futures_klines(symbol: str, interval: str = '4h', silent: bool = False) -> pd.DataFrame:
    """바이낸스 선물 캔들 데이터 전체 다운로드 (캐시 지원)"""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}_full.csv")

    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mod_time).days < 1:
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            if not silent:
                print(f"  📂 캐시 로드: {symbol} ({interval}) - {len(df)}개")
            return df

    if not silent:
        print(f"  📥 다운로드: {symbol} ({interval})...")

    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)

    for _ in range(100):
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1500,
            'endTime': end_time
        }

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
                print(f"    ⚠️ API 오류: {e}")
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
    if not silent:
        print(f"    ✅ 저장: {len(df)}개 캔들")

    return df


def prepare_coin_data(symbol: str, silent: bool = False) -> dict:
    """코인 데이터 준비 (4H + 1D)"""
    df_4h = get_binance_futures_klines(symbol, '4h', silent)
    df_daily = get_binance_futures_klines(symbol, '1d', silent)

    if df_4h.empty or df_daily.empty:
        return None

    if len(df_4h) < 200 or len(df_daily) < 100:
        return None

    start_date = df_4h['timestamp'].min()
    end_date = df_4h['timestamp'].max()
    days = (end_date - start_date).days

    if days < MIN_DATA_DAYS:
        if not silent:
            print(f"    ⚠️ 데이터 기간 부족: {days}일 < {MIN_DATA_DAYS}일")
        return None

    return {
        'symbol': symbol,
        'df_4h': df_4h,
        'df_daily': df_daily,
        'start_date': start_date,
        'end_date': end_date,
        'days': days
    }


# ==========================================
# 지표 계산
# ==========================================

def calculate_stochastic(df: pd.DataFrame, k_period: int, k_smooth: int, d_period: int) -> tuple:
    """스토캐스틱 Slow K, Slow D 계산"""
    lowest = df['low'].rolling(window=k_period).min()
    highest = df['high'].rolling(window=k_period).max()

    fast_k = ((df['close'] - lowest) / (highest - lowest)) * 100
    fast_k = fast_k.replace([np.inf, -np.inf], np.nan)

    slow_k = fast_k.rolling(window=k_smooth).mean()
    slow_d = slow_k.rolling(window=d_period).mean()

    return slow_k, slow_d


def prepare_signals(df_4h: pd.DataFrame, df_daily: pd.DataFrame,
                    ma_period: int, stoch_k: int, stoch_ks: int, stoch_d: int) -> pd.DataFrame:
    """
    4H MA + 1D 스토캐스틱을 계산하여 4H 프레임에 전일 시그널 매핑
    Returns: df_4h에 'ma', 'prev_slow_k', 'prev_slow_d' 컬럼 추가
    """
    df = df_4h.copy()
    df['ma'] = df['close'].rolling(window=ma_period).mean()
    df['date'] = df['timestamp'].dt.date

    df_d = df_daily.copy()
    slow_k, slow_d = calculate_stochastic(df_d, stoch_k, stoch_ks, stoch_d)
    df_d['slow_k'] = slow_k
    df_d['slow_d'] = slow_d
    df_d['date'] = df_d['timestamp'].dt.date

    # 전일 스토캐스틱 → 오늘 4H에 매핑
    df_d['prev_slow_k'] = df_d['slow_k'].shift(1)
    df_d['prev_slow_d'] = df_d['slow_d'].shift(1)
    stoch_map = df_d.dropna(subset=['prev_slow_k', 'prev_slow_d']).set_index('date')[
        ['prev_slow_k', 'prev_slow_d']].to_dict('index')

    df['prev_slow_k'] = df['date'].map(lambda x: stoch_map.get(x, {}).get('prev_slow_k'))
    df['prev_slow_d'] = df['date'].map(lambda x: stoch_map.get(x, {}).get('prev_slow_d'))

    return df


# ==========================================
# 성과 계산 공통 함수
# ==========================================

def calculate_performance(equity_curve: np.ndarray, total_days: int,
                          trade_count: int, long_count: int = 0, short_count: int = 0) -> dict:
    """에쿼티 커브에서 CAGR, MDD, Sharpe 계산"""
    if len(equity_curve) < 2:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    if equity_curve[-1] <= 0:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': trade_count,
                'long_trades': long_count, 'short_trades': short_count}

    if total_days < 30:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    equity_curve = np.maximum(equity_curve, 0.01)

    total_return = equity_curve[-1] / equity_curve[0]
    if total_return <= 0:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    years = total_days / 365.25
    cagr = (total_return ** (1 / years) - 1) * 100

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    mdd = drawdown.min()

    # Sharpe (4H 수익률 기반, 연율화)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) > 10:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 6) if returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'trades': trade_count,
        'days': total_days,
        'long_trades': long_count,
        'short_trades': short_count,
    }


# ==========================================
# Stage 1: 순수 숏 백테스트 (롱 없이)
# ==========================================

def backtest_short_only(data: dict, short_ma: int, short_sk: int,
                        short_sks: int, short_sd: int, short_lev: int) -> dict:
    """
    순수 숏 전략만 백테스트 (Stage 1용)
    - 숏 진입: open < MA_short AND K_short < D_short (전일)
    - 숏 청산: 조건 미충족
    - 포지션 내 비복리, 거래 간 복리
    """
    df_4h = data['df_4h']
    df_daily = data['df_daily']

    max_stoch = short_sk + short_sks + short_sd + 10
    if len(df_4h) < short_ma + 50 or len(df_daily) < max_stoch:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    df_sig = prepare_signals(df_4h, df_daily, short_ma, short_sk, short_sks, short_sd)
    df_bt = df_sig[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                     'prev_slow_k', 'prev_slow_d']].copy()
    df_bt = df_bt.dropna(subset=['ma', 'prev_slow_k', 'prev_slow_d']).reset_index(drop=True)

    if len(df_bt) < 100:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    # 시뮬레이션
    initial_capital = 10000
    equity = initial_capital
    in_short = False
    entry_price = 0.0
    entry_margin = 0.0
    cum_funding = 0.0
    trade_count = 0
    equity_curve = [equity]
    leverage = short_lev

    for i in range(1, len(df_bt)):
        prev = df_bt.iloc[i - 1]
        curr = df_bt.iloc[i]
        opening_price = curr['open']

        # 숏 시그널: open < MA AND K < D
        short_signal = (opening_price < prev['ma']) and (prev['prev_slow_k'] < prev['prev_slow_d'])

        # 숏 진입
        if short_signal and not in_short:
            fee = equity * FEE_RATE * leverage
            slippage = equity * SLIPPAGE_PCT * leverage
            equity -= (fee + slippage)

            entry_margin = equity
            entry_price = opening_price
            cum_funding = 0.0
            in_short = True
            trade_count += 1

        # 숏 청산
        elif not short_signal and in_short:
            if entry_price > 0:
                price_return = opening_price / entry_price - 1
                trade_pnl = entry_margin * (-price_return) * leverage  # 숏 PnL
                equity = entry_margin + trade_pnl - cum_funding

            # 청산 수수료: 노셔널 기준
            if entry_price > 0:
                exit_notional = abs(entry_margin * leverage * (opening_price / entry_price))
            else:
                exit_notional = abs(equity * leverage)
            fee = exit_notional * FEE_RATE
            slippage_cost = exit_notional * SLIPPAGE_PCT
            equity -= (fee + slippage_cost)
            equity = max(equity, 0)

            in_short = False
            entry_price = 0.0
            entry_margin = 0.0
            cum_funding = 0.0
            trade_count += 1

        # 포지션 보유 중
        if in_short and entry_price > 0:
            price_return = curr['close'] / entry_price - 1
            unrealized_pnl = entry_margin * (-price_return) * leverage  # 숏
            cum_funding += entry_margin * leverage * FUNDING_PER_4H  # 노셔널 기준
            display_equity = entry_margin + unrealized_pnl - cum_funding

            # 강제청산: 숏은 가격 상승 시
            if leverage > 0:
                liquidation_price = entry_price * (1 + 1 / leverage)
                if curr['high'] >= liquidation_price:
                    equity = 0
                    in_short = False
                    entry_price = 0.0
                    entry_margin = 0.0
                    cum_funding = 0.0
                    equity_curve.append(0)
                    break

            equity_curve.append(max(display_equity, 0))
        else:
            equity_curve.append(max(equity, 0))

    # 마지막 포지션 정리
    if in_short and entry_price > 0:
        last_price = df_bt.iloc[-1]['close']
        price_return = last_price / entry_price - 1
        trade_pnl = entry_margin * (-price_return) * leverage
        equity = entry_margin + trade_pnl - cum_funding

        exit_notional = abs(entry_margin * leverage * (last_price / entry_price))
        fee = exit_notional * FEE_RATE
        slippage_cost = exit_notional * SLIPPAGE_PCT
        equity -= (fee + slippage_cost)

    total_days = (df_bt.iloc[-1]['timestamp'] - df_bt.iloc[0]['timestamp']).days
    equity_arr = np.array(equity_curve)

    result = calculate_performance(equity_arr, total_days, trade_count,
                                   long_count=0, short_count=trade_count // 2)
    return result


# ==========================================
# Stage 2/3: 숏+롱 결합 백테스트 (숏 우선)
# ==========================================

def backtest_combined(data: dict,
                      short_ma: int, short_sk: int, short_sks: int, short_sd: int, short_lev: int,
                      long_ma: int, long_sk: int, long_sks: int, long_sd: int, long_lev: int) -> dict:
    """
    숏+롱 결합 백테스트 (숏 우선, 롱 보조)

    전략:
      if 숏 시그널:
          롱 보유 → 롱 청산 → 숏 진입
      elif 롱 시그널 AND 숏 미보유:
          롱 진입
      else:
          보유 포지션 청산
    """
    df_4h = data['df_4h']
    df_daily = data['df_daily']

    # 최소 데이터 확인
    max_stoch_short = short_sk + short_sks + short_sd + 10
    max_stoch_long = long_sk + long_sks + long_sd + 10
    min_ma = max(short_ma, long_ma)

    if len(df_4h) < min_ma + 50 or len(df_daily) < max(max_stoch_short, max_stoch_long):
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    # 숏 시그널 준비
    df_short = prepare_signals(df_4h, df_daily, short_ma, short_sk, short_sks, short_sd)
    # 롱 시그널 준비
    df_long = prepare_signals(df_4h, df_daily, long_ma, long_sk, long_sks, long_sd)

    # 합치기
    df_bt = df_short[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                       'prev_slow_k', 'prev_slow_d']].copy()
    df_bt.rename(columns={
        'ma': 'ma_short',
        'prev_slow_k': 'short_k',
        'prev_slow_d': 'short_d',
    }, inplace=True)

    df_bt['ma_long'] = df_long['ma'].values
    df_bt['long_k'] = df_long['prev_slow_k'].values
    df_bt['long_d'] = df_long['prev_slow_d'].values

    df_bt = df_bt.dropna(subset=['ma_short', 'short_k', 'short_d',
                                  'ma_long', 'long_k', 'long_d']).reset_index(drop=True)

    if len(df_bt) < 100:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    # 시뮬레이션
    initial_capital = 10000
    equity = initial_capital
    in_long = False
    in_short = False
    entry_price = 0.0
    entry_margin = 0.0
    cum_funding = 0.0
    trade_count = 0
    long_trade_count = 0
    short_trade_count = 0
    equity_curve = [equity]
    current_lev = 0

    for i in range(1, len(df_bt)):
        prev = df_bt.iloc[i - 1]
        curr = df_bt.iloc[i]
        opening_price = curr['open']

        # 시그널 계산
        short_signal = (opening_price < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
        long_signal = (opening_price > prev['ma_long']) and (prev['long_k'] > prev['long_d'])

        # ── 포지션 전환 로직 (숏 우선) ──

        if short_signal:
            # 롱 보유 중이면 먼저 청산
            if in_long and entry_price > 0:
                price_return = opening_price / entry_price - 1
                trade_pnl = entry_margin * price_return * current_lev  # 롱 PnL
                equity = entry_margin + trade_pnl - cum_funding

                exit_notional = abs(entry_margin * current_lev * (opening_price / entry_price))
                fee = exit_notional * FEE_RATE
                slippage_cost = exit_notional * SLIPPAGE_PCT
                equity -= (fee + slippage_cost)
                equity = max(equity, 0)

                in_long = False
                entry_price = 0.0
                entry_margin = 0.0
                cum_funding = 0.0
                trade_count += 1

            # 숏 미보유면 진입
            if not in_short and equity > 0:
                fee = equity * FEE_RATE * short_lev
                slippage = equity * SLIPPAGE_PCT * short_lev
                equity -= (fee + slippage)

                entry_margin = equity
                entry_price = opening_price
                cum_funding = 0.0
                in_short = True
                current_lev = short_lev
                trade_count += 1
                short_trade_count += 1

        elif long_signal and not in_short:
            # 롱 미보유면 진입
            if not in_long and equity > 0:
                fee = equity * FEE_RATE * long_lev
                slippage = equity * SLIPPAGE_PCT * long_lev
                equity -= (fee + slippage)

                entry_margin = equity
                entry_price = opening_price
                cum_funding = 0.0
                in_long = True
                current_lev = long_lev
                trade_count += 1
                long_trade_count += 1

        else:
            # 숏 보유 중이고 숏 시그널 없으면 청산
            if in_short and entry_price > 0:
                price_return = opening_price / entry_price - 1
                trade_pnl = entry_margin * (-price_return) * current_lev  # 숏 PnL
                equity = entry_margin + trade_pnl - cum_funding

                exit_notional = abs(entry_margin * current_lev * (opening_price / entry_price))
                fee = exit_notional * FEE_RATE
                slippage_cost = exit_notional * SLIPPAGE_PCT
                equity -= (fee + slippage_cost)
                equity = max(equity, 0)

                in_short = False
                entry_price = 0.0
                entry_margin = 0.0
                cum_funding = 0.0
                trade_count += 1

            # 롱 보유 중이고 롱 시그널 없으면 청산
            if in_long and entry_price > 0:
                price_return = opening_price / entry_price - 1
                trade_pnl = entry_margin * price_return * current_lev
                equity = entry_margin + trade_pnl - cum_funding

                exit_notional = abs(entry_margin * current_lev * (opening_price / entry_price))
                fee = exit_notional * FEE_RATE
                slippage_cost = exit_notional * SLIPPAGE_PCT
                equity -= (fee + slippage_cost)
                equity = max(equity, 0)

                in_long = False
                entry_price = 0.0
                entry_margin = 0.0
                cum_funding = 0.0
                trade_count += 1

        # ── 포지션 보유 중 처리 ──
        if in_short and entry_price > 0:
            price_return = curr['close'] / entry_price - 1
            unrealized_pnl = entry_margin * (-price_return) * current_lev  # 숏
            cum_funding += entry_margin * current_lev * FUNDING_PER_4H
            display_equity = entry_margin + unrealized_pnl - cum_funding

            # 강제청산: 숏
            if current_lev > 0:
                liquidation_price = entry_price * (1 + 1 / current_lev)
                if curr['high'] >= liquidation_price:
                    equity = 0
                    in_short = False
                    entry_price = 0.0
                    entry_margin = 0.0
                    cum_funding = 0.0
                    equity_curve.append(0)
                    break

            equity_curve.append(max(display_equity, 0))

        elif in_long and entry_price > 0:
            price_return = curr['close'] / entry_price - 1
            unrealized_pnl = entry_margin * price_return * current_lev
            cum_funding += entry_margin * current_lev * FUNDING_PER_4H
            display_equity = entry_margin + unrealized_pnl - cum_funding

            # 강제청산: 롱
            if current_lev > 0:
                liquidation_price = entry_price * (1 - 1 / current_lev)
                if curr['low'] <= liquidation_price:
                    equity = 0
                    in_long = False
                    entry_price = 0.0
                    entry_margin = 0.0
                    cum_funding = 0.0
                    equity_curve.append(0)
                    break

            equity_curve.append(max(display_equity, 0))

        else:
            equity_curve.append(max(equity, 0))

    # 마지막 포지션 정리
    if in_short and entry_price > 0:
        last_price = df_bt.iloc[-1]['close']
        price_return = last_price / entry_price - 1
        trade_pnl = entry_margin * (-price_return) * current_lev
        equity = entry_margin + trade_pnl - cum_funding

        exit_notional = abs(entry_margin * current_lev * (last_price / entry_price))
        fee = exit_notional * FEE_RATE
        slippage_cost = exit_notional * SLIPPAGE_PCT
        equity -= (fee + slippage_cost)

    elif in_long and entry_price > 0:
        last_price = df_bt.iloc[-1]['close']
        price_return = last_price / entry_price - 1
        trade_pnl = entry_margin * price_return * current_lev
        equity = entry_margin + trade_pnl - cum_funding

        exit_notional = abs(entry_margin * current_lev * (last_price / entry_price))
        fee = exit_notional * FEE_RATE
        slippage_cost = exit_notional * SLIPPAGE_PCT
        equity -= (fee + slippage_cost)

    total_days = (df_bt.iloc[-1]['timestamp'] - df_bt.iloc[0]['timestamp']).days
    equity_arr = np.array(equity_curve)

    result = calculate_performance(equity_arr, total_days, trade_count,
                                   long_count=long_trade_count, short_count=short_trade_count)

    # 롱/숏 비율
    total_bars = len(df_bt) - 1
    if total_bars > 0:
        result['short_ratio'] = short_trade_count / max(trade_count, 1) * 100
        result['long_ratio'] = long_trade_count / max(trade_count, 1) * 100

    return result


# ==========================================
# Stage 1: 숏 파라미터 최적화
# ==========================================

def optimize_short_phase1(symbol: str, data: dict) -> dict:
    """Stage 1 - Phase 1: 숏만 빠른 스크리닝 (50 trials)"""

    def objective(trial):
        ma = trial.suggest_int('short_ma', *PARAM_RANGES['ma_range'])
        sk = trial.suggest_int('short_sk', *PARAM_RANGES['stoch_k_range'])
        sks = trial.suggest_int('short_sks', *PARAM_RANGES['stoch_smooth_range'])
        sd = trial.suggest_int('short_sd', *PARAM_RANGES['stoch_d_range'])
        lev = trial.suggest_int('short_lev', *PARAM_RANGES['leverage_range'])

        result = backtest_short_only(data, ma, sk, sks, sd, lev)

        cagr = result['cagr']
        mdd = result['mdd']
        sharpe = result['sharpe']

        if cagr < -50 or mdd < -90:
            return -9999

        score = cagr + (mdd * 0.5) + (sharpe * 15)
        return score

    sampler = TPESampler(seed=42, n_startup_trials=min(20, PHASE1_TRIALS // 2))
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=PHASE1_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_short_only(data, best['short_ma'], best['short_sk'],
                                      best['short_sks'], best['short_sd'], best['short_lev'])

    return {
        'short_ma': best['short_ma'],
        'short_sk': best['short_sk'],
        'short_sks': best['short_sks'],
        'short_sd': best['short_sd'],
        'short_lev': best['short_lev'],
        'cagr': best_result['cagr'],
        'mdd': best_result['mdd'],
        'sharpe': best_result['sharpe'],
        'trades': best_result['trades'],
    }


def optimize_short_phase2(symbol: str, data: dict, phase1_result: dict) -> dict:
    """Stage 1 - Phase 2: 숏만 정밀 파인튜닝 (300 trials)"""

    def get_range(center, original_range, ratio=0.3):
        min_val, max_val = original_range
        delta = int((max_val - min_val) * ratio)
        new_min = max(min_val, center - delta)
        new_max = min(max_val, center + delta)
        return (new_min, new_max)

    ranges_p2 = {
        'ma_range': get_range(phase1_result['short_ma'], PARAM_RANGES['ma_range']),
        'stoch_k_range': get_range(phase1_result['short_sk'], PARAM_RANGES['stoch_k_range']),
        'stoch_smooth_range': get_range(phase1_result['short_sks'], PARAM_RANGES['stoch_smooth_range']),
        'stoch_d_range': get_range(phase1_result['short_sd'], PARAM_RANGES['stoch_d_range']),
        'leverage_range': PARAM_RANGES['leverage_range'],
    }

    print(f"    📊 숏 2단계 탐색 범위:")
    print(f"       MA: {ranges_p2['ma_range']} (1단계: {phase1_result['short_ma']})")
    print(f"       SK: {ranges_p2['stoch_k_range']} (1단계: {phase1_result['short_sk']})")
    print(f"       SKs: {ranges_p2['stoch_smooth_range']} (1단계: {phase1_result['short_sks']})")
    print(f"       SD: {ranges_p2['stoch_d_range']} (1단계: {phase1_result['short_sd']})")

    def objective(trial):
        ma = trial.suggest_int('short_ma', *ranges_p2['ma_range'])
        sk = trial.suggest_int('short_sk', *ranges_p2['stoch_k_range'])
        sks = trial.suggest_int('short_sks', *ranges_p2['stoch_smooth_range'])
        sd = trial.suggest_int('short_sd', *ranges_p2['stoch_d_range'])
        lev = trial.suggest_int('short_lev', *ranges_p2['leverage_range'])

        result = backtest_short_only(data, ma, sk, sks, sd, lev)

        cagr = result['cagr']
        mdd = result['mdd']
        sharpe = result['sharpe']

        if cagr < -50 or mdd < -90:
            return -9999

        score = cagr + (mdd * 0.7) + (sharpe * 20)
        return score

    sampler = TPESampler(seed=42, n_startup_trials=min(30, PHASE2_TRIALS // 3))
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Phase1 최적값 시드 삽입
    study.enqueue_trial({
        'short_ma': phase1_result['short_ma'],
        'short_sk': phase1_result['short_sk'],
        'short_sks': phase1_result['short_sks'],
        'short_sd': phase1_result['short_sd'],
        'short_lev': phase1_result['short_lev'],
    })

    study.optimize(objective, n_trials=PHASE2_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_short_only(data, best['short_ma'], best['short_sk'],
                                      best['short_sks'], best['short_sd'], best['short_lev'])

    return {
        'short_ma': best['short_ma'],
        'short_sk': best['short_sk'],
        'short_sks': best['short_sks'],
        'short_sd': best['short_sd'],
        'short_lev': best['short_lev'],
        'cagr': best_result['cagr'],
        'mdd': best_result['mdd'],
        'sharpe': best_result['sharpe'],
        'trades': best_result['trades'],
        'phase1_cagr': phase1_result['cagr'],
        'improvement': best_result['cagr'] - phase1_result['cagr'],
    }


# ==========================================
# Stage 2: 롱 파라미터 최적화 (숏 고정)
# ==========================================

def optimize_long_phase1(symbol: str, data: dict, short_params: dict) -> dict:
    """Stage 2 - Phase 1: 롱 빠른 스크리닝 (숏 고정, 50 trials)"""

    def objective(trial):
        l_ma = trial.suggest_int('long_ma', *PARAM_RANGES['ma_range'])
        l_sk = trial.suggest_int('long_sk', *PARAM_RANGES['stoch_k_range'])
        l_sks = trial.suggest_int('long_sks', *PARAM_RANGES['stoch_smooth_range'])
        l_sd = trial.suggest_int('long_sd', *PARAM_RANGES['stoch_d_range'])
        l_lev = trial.suggest_int('long_lev', *PARAM_RANGES['leverage_range'])

        result = backtest_combined(
            data,
            short_params['short_ma'], short_params['short_sk'],
            short_params['short_sks'], short_params['short_sd'], short_params['short_lev'],
            l_ma, l_sk, l_sks, l_sd, l_lev
        )

        cagr = result['cagr']
        mdd = result['mdd']
        sharpe = result['sharpe']

        if cagr < -50 or mdd < -90:
            return -9999

        score = cagr + (mdd * 0.5) + (sharpe * 15)
        return score

    sampler = TPESampler(seed=42, n_startup_trials=min(20, PHASE1_TRIALS // 2))
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=PHASE1_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_combined(
        data,
        short_params['short_ma'], short_params['short_sk'],
        short_params['short_sks'], short_params['short_sd'], short_params['short_lev'],
        best['long_ma'], best['long_sk'], best['long_sks'], best['long_sd'], best['long_lev']
    )

    return {
        'long_ma': best['long_ma'],
        'long_sk': best['long_sk'],
        'long_sks': best['long_sks'],
        'long_sd': best['long_sd'],
        'long_lev': best['long_lev'],
        'cagr': best_result['cagr'],
        'mdd': best_result['mdd'],
        'sharpe': best_result['sharpe'],
        'trades': best_result['trades'],
        'long_trades': best_result.get('long_trades', 0),
        'short_trades': best_result.get('short_trades', 0),
    }


def optimize_long_phase2(symbol: str, data: dict, short_params: dict, phase1_result: dict) -> dict:
    """Stage 2 - Phase 2: 롱 정밀 파인튜닝 (숏 고정, 300 trials)"""

    def get_range(center, original_range, ratio=0.3):
        min_val, max_val = original_range
        delta = int((max_val - min_val) * ratio)
        new_min = max(min_val, center - delta)
        new_max = min(max_val, center + delta)
        return (new_min, new_max)

    ranges_p2 = {
        'ma_range': get_range(phase1_result['long_ma'], PARAM_RANGES['ma_range']),
        'stoch_k_range': get_range(phase1_result['long_sk'], PARAM_RANGES['stoch_k_range']),
        'stoch_smooth_range': get_range(phase1_result['long_sks'], PARAM_RANGES['stoch_smooth_range']),
        'stoch_d_range': get_range(phase1_result['long_sd'], PARAM_RANGES['stoch_d_range']),
        'leverage_range': PARAM_RANGES['leverage_range'],
    }

    print(f"    📊 롱 2단계 탐색 범위:")
    print(f"       MA: {ranges_p2['ma_range']} (1단계: {phase1_result['long_ma']})")
    print(f"       SK: {ranges_p2['stoch_k_range']} (1단계: {phase1_result['long_sk']})")
    print(f"       SKs: {ranges_p2['stoch_smooth_range']} (1단계: {phase1_result['long_sks']})")
    print(f"       SD: {ranges_p2['stoch_d_range']} (1단계: {phase1_result['long_sd']})")

    def objective(trial):
        l_ma = trial.suggest_int('long_ma', *ranges_p2['ma_range'])
        l_sk = trial.suggest_int('long_sk', *ranges_p2['stoch_k_range'])
        l_sks = trial.suggest_int('long_sks', *ranges_p2['stoch_smooth_range'])
        l_sd = trial.suggest_int('long_sd', *ranges_p2['stoch_d_range'])
        l_lev = trial.suggest_int('long_lev', *ranges_p2['leverage_range'])

        result = backtest_combined(
            data,
            short_params['short_ma'], short_params['short_sk'],
            short_params['short_sks'], short_params['short_sd'], short_params['short_lev'],
            l_ma, l_sk, l_sks, l_sd, l_lev
        )

        cagr = result['cagr']
        mdd = result['mdd']
        sharpe = result['sharpe']

        if cagr < -50 or mdd < -90:
            return -9999

        score = cagr + (mdd * 0.7) + (sharpe * 20)
        return score

    sampler = TPESampler(seed=42, n_startup_trials=min(30, PHASE2_TRIALS // 3))
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Phase1 최적값 시드 삽입
    study.enqueue_trial({
        'long_ma': phase1_result['long_ma'],
        'long_sk': phase1_result['long_sk'],
        'long_sks': phase1_result['long_sks'],
        'long_sd': phase1_result['long_sd'],
        'long_lev': phase1_result['long_lev'],
    })

    study.optimize(objective, n_trials=PHASE2_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_combined(
        data,
        short_params['short_ma'], short_params['short_sk'],
        short_params['short_sks'], short_params['short_sd'], short_params['short_lev'],
        best['long_ma'], best['long_sk'], best['long_sks'], best['long_sd'], best['long_lev']
    )

    return {
        'long_ma': best['long_ma'],
        'long_sk': best['long_sk'],
        'long_sks': best['long_sks'],
        'long_sd': best['long_sd'],
        'long_lev': best['long_lev'],
        'cagr': best_result['cagr'],
        'mdd': best_result['mdd'],
        'sharpe': best_result['sharpe'],
        'trades': best_result['trades'],
        'long_trades': best_result.get('long_trades', 0),
        'short_trades': best_result.get('short_trades', 0),
        'phase1_cagr': phase1_result['cagr'],
        'improvement': best_result['cagr'] - phase1_result['cagr'],
    }


# ==========================================
# 메인 실행
# ==========================================

def main():
    print("=" * 80)
    print("📉 Binance Futures 메이저 코인 숏+롱 독립 파라미터 최적화 (숏 우선)")
    print("=" * 80)
    print(f"📌 대상 코인: {', '.join(TARGET_COINS)} ({len(TARGET_COINS)}개)")
    print(f"📌 수수료: {FEE_RATE*100:.2f}%")
    print(f"📌 슬리피지: {SLIPPAGE_PCT*100:.2f}%")
    print(f"📌 펀딩비: {FUNDING_RATE_8H*100:.3f}% / 8h (노셔널 기준)")
    print(f"📌 방식: 포지션 내 비복리, 거래 간 복리")
    print(f"📌 Stage 1 (숏만): Phase1 {PHASE1_TRIALS} + Phase2 {PHASE2_TRIALS} trials")
    print(f"📌 Stage 2 (롱, 숏 고정): Phase1 {PHASE1_TRIALS} + Phase2 {PHASE2_TRIALS} trials")
    print(f"📌 전략: 숏 우선, 롱 보조 (숏 시그널 시 롱 청산)")
    print(f"📌 레버리지 범위: {PARAM_RANGES['leverage_range'][0]}~{PARAM_RANGES['leverage_range'][1]}배")
    print("=" * 80)

    all_results = []

    for idx, symbol in enumerate(TARGET_COINS, 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(TARGET_COINS)}] {symbol}")
        print(f"{'=' * 80}")

        # 데이터 준비
        data = prepare_coin_data(symbol)
        if not data:
            print(f"  ❌ {symbol}: 데이터 준비 실패")
            continue

        print(f"  📅 기간: {data['start_date'].date()} ~ {data['end_date'].date()} ({data['days']}일)")

        # ────────────────────────────────
        # Stage 1: 숏 파라미터 최적화
        # ────────────────────────────────
        print(f"\n  ── Stage 1: 숏 파라미터 최적화 ──")

        print(f"    Phase 1 ({PHASE1_TRIALS} trials)...", end=" ", flush=True)
        short_p1 = optimize_short_phase1(symbol, data)
        print(f"✅ CAGR:{short_p1['cagr']:.1f}% Lev:{short_p1['short_lev']}x "
              f"MA:{short_p1['short_ma']} SK:{short_p1['short_sk']}/{short_p1['short_sks']}/{short_p1['short_sd']}")

        print(f"    Phase 2 ({PHASE2_TRIALS} trials)...", flush=True)
        short_p2 = optimize_short_phase2(symbol, data, short_p1)
        print(f"    ✅ CAGR:{short_p2['cagr']:.1f}% (개선:{short_p2['improvement']:+.1f}%) "
              f"Lev:{short_p2['short_lev']}x "
              f"MA:{short_p2['short_ma']} SK:{short_p2['short_sk']}/{short_p2['short_sks']}/{short_p2['short_sd']} "
              f"MDD:{short_p2['mdd']:.1f}% Sharpe:{short_p2['sharpe']:.2f}")

        short_only_cagr = short_p2['cagr']
        short_params = {
            'short_ma': short_p2['short_ma'],
            'short_sk': short_p2['short_sk'],
            'short_sks': short_p2['short_sks'],
            'short_sd': short_p2['short_sd'],
            'short_lev': short_p2['short_lev'],
        }

        # ────────────────────────────────
        # Stage 2: 롱 파라미터 최적화 (숏 고정)
        # ────────────────────────────────
        print(f"\n  ── Stage 2: 롱 파라미터 최적화 (숏 고정) ──")

        print(f"    Phase 1 ({PHASE1_TRIALS} trials)...", end=" ", flush=True)
        long_p1 = optimize_long_phase1(symbol, data, short_params)
        print(f"✅ Combined CAGR:{long_p1['cagr']:.1f}% "
              f"MA:{long_p1['long_ma']} SK:{long_p1['long_sk']}/{long_p1['long_sks']}/{long_p1['long_sd']} "
              f"Lev:{long_p1['long_lev']}x "
              f"숏:{long_p1['short_trades']}회 롱:{long_p1['long_trades']}회")

        print(f"    Phase 2 ({PHASE2_TRIALS} trials)...", flush=True)
        long_p2 = optimize_long_phase2(symbol, data, short_params, long_p1)
        print(f"    ✅ Combined CAGR:{long_p2['cagr']:.1f}% (개선:{long_p2['improvement']:+.1f}%) "
              f"MA:{long_p2['long_ma']} SK:{long_p2['long_sk']}/{long_p2['long_sks']}/{long_p2['long_sd']} "
              f"Lev:{long_p2['long_lev']}x MDD:{long_p2['mdd']:.1f}% Sharpe:{long_p2['sharpe']:.2f} "
              f"숏:{long_p2['short_trades']}회 롱:{long_p2['long_trades']}회")

        # ────────────────────────────────
        # Stage 3: 결합 검증
        # ────────────────────────────────
        print(f"\n  ── Stage 3: 결합 검증 ──")
        combined_result = backtest_combined(
            data,
            short_params['short_ma'], short_params['short_sk'],
            short_params['short_sks'], short_params['short_sd'], short_params['short_lev'],
            long_p2['long_ma'], long_p2['long_sk'],
            long_p2['long_sks'], long_p2['long_sd'], long_p2['long_lev']
        )

        combined_cagr = combined_result['cagr']
        diff = combined_cagr - short_only_cagr
        better = "✅ 롱 추가 효과" if diff > 0 else "⚠️ 롱 추가 시 성과 하락"

        print(f"    숏만 CAGR:   {short_only_cagr:.1f}%")
        print(f"    결합 CAGR:   {combined_cagr:.1f}% ({diff:+.1f}%) → {better}")
        print(f"    결합 MDD:    {combined_result['mdd']:.1f}%")
        print(f"    결합 Sharpe: {combined_result['sharpe']:.2f}")
        print(f"    거래수: 총 {combined_result['trades']}회 "
              f"(숏 {combined_result.get('short_trades', 0)}회, "
              f"롱 {combined_result.get('long_trades', 0)}회)")

        # 결과 저장
        result_row = {
            'Symbol': symbol,
            'Short_CAGR': short_only_cagr,
            'Combined_CAGR': combined_cagr,
            'Combined_MDD': combined_result['mdd'],
            'Combined_Sharpe': combined_result['sharpe'],
            'Combined_Trades': combined_result['trades'],
            'Short_Trades': combined_result.get('short_trades', 0),
            'Long_Trades': combined_result.get('long_trades', 0),
            'Days': data['days'],
            'Short_MA': short_params['short_ma'],
            'Short_SK': short_params['short_sk'],
            'Short_SKs': short_params['short_sks'],
            'Short_SD': short_params['short_sd'],
            'Short_Lev': short_params['short_lev'],
            'Long_MA': long_p2['long_ma'],
            'Long_SK': long_p2['long_sk'],
            'Long_SKs': long_p2['long_sks'],
            'Long_SD': long_p2['long_sd'],
            'Long_Lev': long_p2['long_lev'],
            'Short_Ratio': combined_result.get('short_ratio', 0),
            'Long_Ratio': combined_result.get('long_ratio', 0),
        }
        all_results.append(result_row)

    # ── 최종 결과 ──
    if not all_results:
        print("\n❌ 결과가 없습니다.")
        return

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('Combined_CAGR', ascending=False)

    # CSV 저장
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(SAVE_DIR, f"binance_major_coins_short_first_optimized_{ts}.csv")
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 최종 결과 CSV: {csv_path}")

    # ── 결과 요약 테이블 ──
    print(f"\n{'=' * 130}")
    print(f"📊 최종 결과 요약 (숏 우선)")
    print(f"{'=' * 130}")
    print(f"{'Symbol':<12} {'S_Lev':>5} {'S_MA':>5} {'S_SK':>5} {'S_SKs':>5} {'S_SD':>5} "
          f"{'L_Lev':>5} {'L_MA':>5} {'L_SK':>5} {'L_SKs':>5} {'L_SD':>5} "
          f"{'S_CAGR%':>8} {'C_CAGR%':>8} {'C_MDD%':>7} {'Sharpe':>7} "
          f"{'총거래':>6} {'숏':>4} {'롱':>4}")
    print(f"{'─' * 130}")

    for _, row in df_results.iterrows():
        print(f"{row['Symbol']:<12} "
              f"{row['Short_Lev']:>4}x {row['Short_MA']:>5} {row['Short_SK']:>5} "
              f"{row['Short_SKs']:>5} {row['Short_SD']:>5} "
              f"{row['Long_Lev']:>4}x {row['Long_MA']:>5} {row['Long_SK']:>5} "
              f"{row['Long_SKs']:>5} {row['Long_SD']:>5} "
              f"{row['Short_CAGR']:>7.1f}% {row['Combined_CAGR']:>7.1f}% "
              f"{row['Combined_MDD']:>6.1f}% {row['Combined_Sharpe']:>7.2f} "
              f"{row['Combined_Trades']:>6} {row['Short_Trades']:>4} {row['Long_Trades']:>4}")

    # ── 봇 적용용 COMBINED_TRADING_CONFIGS 출력 ──
    print(f"\n{'=' * 80}")
    print("📋 MAJOR_TRADING_CONFIGS (봇 적용용 - 숏 우선)")
    print(f"{'=' * 80}")
    print("MAJOR_TRADING_CONFIGS = [")
    for _, row in df_results.iterrows():
        print(f"    {{'symbol': '{row['Symbol']}', "
              f"'short_ma': {row['Short_MA']}, 'short_sk': {row['Short_SK']}, "
              f"'short_sks': {row['Short_SKs']}, 'short_sd': {row['Short_SD']}, "
              f"'long_ma': {row['Long_MA']}, 'long_sk': {row['Long_SK']}, "
              f"'long_sks': {row['Long_SKs']}, 'long_sd': {row['Long_SD']}, "
              f"'short_lev': {row['Short_Lev']}, 'long_lev': {row['Long_Lev']}}},")
    print("]")

    # ── 통계 요약 ──
    print(f"\n{'=' * 80}")
    print(f"📊 통계 요약")
    print(f"{'=' * 80}")
    print(f"평균 Short-only CAGR: {df_results['Short_CAGR'].mean():.1f}%")
    print(f"평균 Combined CAGR:   {df_results['Combined_CAGR'].mean():.1f}%")
    print(f"평균 Combined MDD:    {df_results['Combined_MDD'].mean():.1f}%")
    print(f"평균 Sharpe:          {df_results['Combined_Sharpe'].mean():.2f}")

    improved = len(df_results[df_results['Combined_CAGR'] > df_results['Short_CAGR']])
    print(f"\n롱 추가 효과: {improved}/{len(df_results)}개 코인 개선")

    print(f"\n📊 레버리지 분포:")
    print(f"  숏: ", end="")
    for lev in range(1, 6):
        cnt = len(df_results[df_results['Short_Lev'] == lev])
        if cnt > 0:
            print(f"{lev}x({cnt}개) ", end="")
    print()
    print(f"  롱: ", end="")
    for lev in range(1, 6):
        cnt = len(df_results[df_results['Long_Lev'] == lev])
        if cnt > 0:
            print(f"{lev}x({cnt}개) ", end="")
    print()

    print(f"\n✅ 메이저 코인 숏+롱 독립 파라미터 최적화 (숏 우선) 완료!")


if __name__ == "__main__":
    main()
