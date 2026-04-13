"""
================================================================================
Binance Futures 메이저 코인 롱우선 vs 숏우선 비교 최적화
================================================================================
대상: BTC, ETH, XRP, SOL, TRX, DOGE, BCH, ADA (8개)

각 코인에 대해:
  A) 롱 우선 전략: Stage1 롱 최적화 → Stage2 숏 최적화(롱 고정) → 결합
  B) 숏 우선 전략: Stage1 숏 최적화 → Stage2 롱 최적화(숏 고정) → 결합
  → 코인별 CAGR/MDD/Sharpe 비교, 최종 승패 집계

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

TARGET_COINS = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT',
                'TRXUSDT', 'DOGEUSDT', 'BCHUSDT', 'ADAUSDT']

PHASE1_TRIALS = 300
PHASE2_TRIALS = 300

FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0005
FUNDING_RATE_8H = 0.0001
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5

MIN_DATA_DAYS = 365

PARAM_RANGES = {
    'ma_range': (20, 350),
    'stoch_k_range': (14, 150),
    'stoch_smooth_range': (5, 80),
    'stoch_d_range': (3, 50),
    'leverage_range': (1, 5),
}

SAVE_DIR = os.path.expanduser("~/Downloads")
CACHE_DIR = os.path.expanduser("~/binance_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ==========================================
# 데이터 수집
# ==========================================

def get_binance_futures_klines(symbol: str, interval: str = '4h', silent: bool = False) -> pd.DataFrame:
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
        'symbol': symbol, 'df_4h': df_4h, 'df_daily': df_daily,
        'start_date': start_date, 'end_date': end_date, 'days': days
    }


# ==========================================
# 지표 계산
# ==========================================

def calculate_stochastic(df: pd.DataFrame, k_period: int, k_smooth: int, d_period: int) -> tuple:
    lowest = df['low'].rolling(window=k_period).min()
    highest = df['high'].rolling(window=k_period).max()
    fast_k = ((df['close'] - lowest) / (highest - lowest)) * 100
    fast_k = fast_k.replace([np.inf, -np.inf], np.nan)
    slow_k = fast_k.rolling(window=k_smooth).mean()
    slow_d = slow_k.rolling(window=d_period).mean()
    return slow_k, slow_d


def prepare_signals(df_4h: pd.DataFrame, df_daily: pd.DataFrame,
                    ma_period: int, stoch_k: int, stoch_ks: int, stoch_d: int) -> pd.DataFrame:
    df = df_4h.copy()
    df['ma'] = df['close'].rolling(window=ma_period).mean()
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


# ==========================================
# 성과 계산
# ==========================================

def calculate_performance(equity_curve: np.ndarray, total_days: int,
                          trade_count: int, long_count: int = 0, short_count: int = 0) -> dict:
    fail = {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
            'long_trades': 0, 'short_trades': 0}

    if len(equity_curve) < 2 or total_days < 30:
        return fail
    if equity_curve[-1] <= 0:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': trade_count,
                'long_trades': long_count, 'short_trades': short_count}

    equity_curve = np.maximum(equity_curve, 0.01)
    total_return = equity_curve[-1] / equity_curve[0]
    if total_return <= 0:
        return fail

    years = total_days / 365.25
    cagr = (total_return ** (1 / years) - 1) * 100

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    mdd = drawdown.min()

    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) > 10:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 6) if returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'cagr': cagr, 'mdd': mdd, 'sharpe': sharpe,
        'trades': trade_count, 'days': total_days,
        'long_trades': long_count, 'short_trades': short_count,
    }


# ==========================================
# 순수 롱/숏 단방향 백테스트
# ==========================================

def backtest_single_direction(data: dict, ma: int, sk: int, sks: int, sd: int, lev: int,
                               direction: str) -> dict:
    """
    direction='long' or 'short' 단방향 백테스트
    """
    df_4h = data['df_4h']
    df_daily = data['df_daily']

    max_stoch = sk + sks + sd + 10
    if len(df_4h) < ma + 50 or len(df_daily) < max_stoch:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    df_sig = prepare_signals(df_4h, df_daily, ma, sk, sks, sd)
    df_bt = df_sig[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                     'prev_slow_k', 'prev_slow_d']].copy()
    df_bt = df_bt.dropna(subset=['ma', 'prev_slow_k', 'prev_slow_d']).reset_index(drop=True)

    if len(df_bt) < 100:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    is_long = (direction == 'long')
    initial_capital = 10000
    equity = initial_capital
    in_pos = False
    entry_price = 0.0
    entry_margin = 0.0
    cum_funding = 0.0
    trade_count = 0
    equity_curve = [equity]

    for i in range(1, len(df_bt)):
        prev = df_bt.iloc[i - 1]
        curr = df_bt.iloc[i]
        opening_price = curr['open']

        if is_long:
            signal = (opening_price > prev['ma']) and (prev['prev_slow_k'] > prev['prev_slow_d'])
        else:
            signal = (opening_price < prev['ma']) and (prev['prev_slow_k'] < prev['prev_slow_d'])

        # 진입
        if signal and not in_pos:
            fee = equity * FEE_RATE * lev
            slippage = equity * SLIPPAGE_PCT * lev
            equity -= (fee + slippage)
            entry_margin = equity
            entry_price = opening_price
            cum_funding = 0.0
            in_pos = True
            trade_count += 1

        # 청산
        elif not signal and in_pos:
            if entry_price > 0:
                price_return = opening_price / entry_price - 1
                if not is_long:
                    price_return = -price_return
                trade_pnl = entry_margin * price_return * lev
                equity = entry_margin + trade_pnl - cum_funding

            if entry_price > 0:
                exit_notional = abs(entry_margin * lev * (opening_price / entry_price))
            else:
                exit_notional = abs(equity * lev)
            equity -= exit_notional * (FEE_RATE + SLIPPAGE_PCT)
            equity = max(equity, 0)

            in_pos = False
            entry_price = 0.0
            entry_margin = 0.0
            cum_funding = 0.0
            trade_count += 1

        # 보유 중
        if in_pos and entry_price > 0:
            price_return = curr['close'] / entry_price - 1
            if not is_long:
                price_return = -price_return
            unrealized_pnl = entry_margin * price_return * lev
            cum_funding += entry_margin * lev * FUNDING_PER_4H
            display_equity = entry_margin + unrealized_pnl - cum_funding

            # 강제청산
            if lev > 0:
                if is_long:
                    liq_price = entry_price * (1 - 1 / lev)
                    triggered = curr['low'] <= liq_price
                else:
                    liq_price = entry_price * (1 + 1 / lev)
                    triggered = curr['high'] >= liq_price

                if triggered:
                    equity = 0
                    in_pos = False
                    entry_price = 0.0
                    entry_margin = 0.0
                    cum_funding = 0.0
                    equity_curve.append(0)
                    break

            equity_curve.append(max(display_equity, 0))
        else:
            equity_curve.append(max(equity, 0))

    # 마지막 포지션 정리
    if in_pos and entry_price > 0:
        last_price = df_bt.iloc[-1]['close']
        price_return = last_price / entry_price - 1
        if not is_long:
            price_return = -price_return
        trade_pnl = entry_margin * price_return * lev
        equity = entry_margin + trade_pnl - cum_funding
        exit_notional = abs(entry_margin * lev * (last_price / entry_price))
        equity -= exit_notional * (FEE_RATE + SLIPPAGE_PCT)

    total_days = (df_bt.iloc[-1]['timestamp'] - df_bt.iloc[0]['timestamp']).days
    equity_arr = np.array(equity_curve)

    lc = trade_count // 2 if is_long else 0
    sc = 0 if is_long else trade_count // 2
    return calculate_performance(equity_arr, total_days, trade_count, long_count=lc, short_count=sc)


# ==========================================
# 결합 백테스트 (priority 파라미터로 우선순위 전환)
# ==========================================

def backtest_combined(data: dict,
                      pri_ma: int, pri_sk: int, pri_sks: int, pri_sd: int, pri_lev: int,
                      sec_ma: int, sec_sk: int, sec_sks: int, sec_sd: int, sec_lev: int,
                      priority: str = 'long') -> dict:
    """
    priority='long': 롱 우선 (숏 보조) - 롱 시그널 시 숏 청산 후 롱 진입
    priority='short': 숏 우선 (롱 보조) - 숏 시그널 시 롱 청산 후 숏 진입

    pri_* = 우선 방향 파라미터, sec_* = 보조 방향 파라미터
    """
    df_4h = data['df_4h']
    df_daily = data['df_daily']

    max_stoch_pri = pri_sk + pri_sks + pri_sd + 10
    max_stoch_sec = sec_sk + sec_sks + sec_sd + 10
    min_ma = max(pri_ma, sec_ma)

    if len(df_4h) < min_ma + 50 or len(df_daily) < max(max_stoch_pri, max_stoch_sec):
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    df_pri = prepare_signals(df_4h, df_daily, pri_ma, pri_sk, pri_sks, pri_sd)
    df_sec = prepare_signals(df_4h, df_daily, sec_ma, sec_sk, sec_sks, sec_sd)

    df_bt = df_pri[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                     'prev_slow_k', 'prev_slow_d']].copy()
    df_bt.rename(columns={'ma': 'ma_pri', 'prev_slow_k': 'pri_k', 'prev_slow_d': 'pri_d'}, inplace=True)
    df_bt['ma_sec'] = df_sec['ma'].values
    df_bt['sec_k'] = df_sec['prev_slow_k'].values
    df_bt['sec_d'] = df_sec['prev_slow_d'].values

    df_bt = df_bt.dropna(subset=['ma_pri', 'pri_k', 'pri_d',
                                  'ma_sec', 'sec_k', 'sec_d']).reset_index(drop=True)

    if len(df_bt) < 100:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

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
        if priority == 'long':
            pri_signal = (opening_price > prev['ma_pri']) and (prev['pri_k'] > prev['pri_d'])  # 롱
            sec_signal = (opening_price < prev['ma_sec']) and (prev['sec_k'] < prev['sec_d'])  # 숏
        else:
            pri_signal = (opening_price < prev['ma_pri']) and (prev['pri_k'] < prev['pri_d'])  # 숏
            sec_signal = (opening_price > prev['ma_sec']) and (prev['sec_k'] > prev['sec_d'])  # 롱

        # ── 내부 헬퍼: 포지션 청산 ──
        def close_position():
            nonlocal equity, in_long, in_short, entry_price, entry_margin, cum_funding, trade_count, current_lev
            if entry_price > 0:
                pr = opening_price / entry_price - 1
                if in_short:
                    pr = -pr
                pnl = entry_margin * pr * current_lev
                equity = entry_margin + pnl - cum_funding
                exit_notional = abs(entry_margin * current_lev * (opening_price / entry_price))
                equity -= exit_notional * (FEE_RATE + SLIPPAGE_PCT)
                equity = max(equity, 0)
            in_long = False
            in_short = False
            entry_price = 0.0
            entry_margin = 0.0
            cum_funding = 0.0
            trade_count += 1

        def enter_position(direction, lev):
            nonlocal equity, in_long, in_short, entry_price, entry_margin, cum_funding
            nonlocal trade_count, long_trade_count, short_trade_count, current_lev
            fee = equity * FEE_RATE * lev
            slippage = equity * SLIPPAGE_PCT * lev
            equity -= (fee + slippage)
            entry_margin = equity
            entry_price = opening_price
            cum_funding = 0.0
            current_lev = lev
            trade_count += 1
            if direction == 'long':
                in_long = True
                long_trade_count += 1
            else:
                in_short = True
                short_trade_count += 1

        # ── 포지션 전환 로직 ──
        if priority == 'long':
            # 롱 우선
            if pri_signal:  # 롱 시그널
                if in_short:
                    close_position()
                if not in_long and equity > 0:
                    enter_position('long', pri_lev)
            elif sec_signal and not in_long:  # 숏 시그널 & 롱 미보유
                if not in_short and equity > 0:
                    enter_position('short', sec_lev)
            else:
                if in_long:
                    close_position()
                if in_short:
                    close_position()
        else:
            # 숏 우선
            if pri_signal:  # 숏 시그널
                if in_long:
                    close_position()
                if not in_short and equity > 0:
                    enter_position('short', pri_lev)
            elif sec_signal and not in_short:  # 롱 시그널 & 숏 미보유
                if not in_long and equity > 0:
                    enter_position('long', sec_lev)
            else:
                if in_short:
                    close_position()
                if in_long:
                    close_position()

        # ── 포지션 보유 중 처리 ──
        if (in_long or in_short) and entry_price > 0:
            pr = curr['close'] / entry_price - 1
            if in_short:
                pr = -pr
            unrealized_pnl = entry_margin * pr * current_lev
            cum_funding += entry_margin * current_lev * FUNDING_PER_4H
            display_equity = entry_margin + unrealized_pnl - cum_funding

            # 강제청산
            if current_lev > 0:
                if in_long:
                    liq_price = entry_price * (1 - 1 / current_lev)
                    triggered = curr['low'] <= liq_price
                else:
                    liq_price = entry_price * (1 + 1 / current_lev)
                    triggered = curr['high'] >= liq_price

                if triggered:
                    equity = 0
                    in_long = False
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
    if (in_long or in_short) and entry_price > 0:
        last_price = df_bt.iloc[-1]['close']
        pr = last_price / entry_price - 1
        if in_short:
            pr = -pr
        pnl = entry_margin * pr * current_lev
        equity = entry_margin + pnl - cum_funding
        exit_notional = abs(entry_margin * current_lev * (last_price / entry_price))
        equity -= exit_notional * (FEE_RATE + SLIPPAGE_PCT)

    total_days = (df_bt.iloc[-1]['timestamp'] - df_bt.iloc[0]['timestamp']).days
    equity_arr = np.array(equity_curve)

    result = calculate_performance(equity_arr, total_days, trade_count,
                                   long_count=long_trade_count, short_count=short_trade_count)
    result['long_ratio'] = long_trade_count / max(trade_count, 1) * 100
    result['short_ratio'] = short_trade_count / max(trade_count, 1) * 100
    return result


# ==========================================
# 2단계 최적화 공통 함수
# ==========================================

def get_narrowed_range(center, original_range, ratio=0.3):
    min_val, max_val = original_range
    delta = int((max_val - min_val) * ratio)
    return (max(min_val, center - delta), min(max_val, center + delta))


def optimize_single_direction(data: dict, direction: str, phase: int,
                               prev_result: dict = None) -> dict:
    """단방향(롱 or 숏) 파라미터 최적화 (Phase 1 or 2)"""
    prefix = 'long' if direction == 'long' else 'short'
    n_trials = PHASE1_TRIALS if phase == 1 else PHASE2_TRIALS

    if phase == 2 and prev_result:
        ranges = {
            'ma_range': get_narrowed_range(prev_result[f'{prefix}_ma'], PARAM_RANGES['ma_range']),
            'stoch_k_range': get_narrowed_range(prev_result[f'{prefix}_sk'], PARAM_RANGES['stoch_k_range']),
            'stoch_smooth_range': get_narrowed_range(prev_result[f'{prefix}_sks'], PARAM_RANGES['stoch_smooth_range']),
            'stoch_d_range': get_narrowed_range(prev_result[f'{prefix}_sd'], PARAM_RANGES['stoch_d_range']),
            'leverage_range': PARAM_RANGES['leverage_range'],
        }
    else:
        ranges = PARAM_RANGES

    def objective(trial):
        ma = trial.suggest_int(f'{prefix}_ma', *ranges['ma_range'])
        sk = trial.suggest_int(f'{prefix}_sk', *ranges['stoch_k_range'])
        sks = trial.suggest_int(f'{prefix}_sks', *ranges['stoch_smooth_range'])
        sd = trial.suggest_int(f'{prefix}_sd', *ranges['stoch_d_range'])
        lev = trial.suggest_int(f'{prefix}_lev', *ranges['leverage_range'])

        result = backtest_single_direction(data, ma, sk, sks, sd, lev, direction)
        cagr, mdd, sharpe = result['cagr'], result['mdd'], result['sharpe']
        if cagr < -50 or mdd < -90:
            return -9999

        if phase == 1:
            return cagr + (mdd * 0.5) + (sharpe * 15)
        else:
            return cagr + (mdd * 0.7) + (sharpe * 20)

    startup = min(20, n_trials // 2) if phase == 1 else min(30, n_trials // 3)
    sampler = TPESampler(seed=42, n_startup_trials=startup)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    if phase == 2 and prev_result:
        study.enqueue_trial({
            f'{prefix}_ma': prev_result[f'{prefix}_ma'],
            f'{prefix}_sk': prev_result[f'{prefix}_sk'],
            f'{prefix}_sks': prev_result[f'{prefix}_sks'],
            f'{prefix}_sd': prev_result[f'{prefix}_sd'],
            f'{prefix}_lev': prev_result[f'{prefix}_lev'],
        })

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_single_direction(
        data, best[f'{prefix}_ma'], best[f'{prefix}_sk'],
        best[f'{prefix}_sks'], best[f'{prefix}_sd'], best[f'{prefix}_lev'], direction)

    out = {
        f'{prefix}_ma': best[f'{prefix}_ma'],
        f'{prefix}_sk': best[f'{prefix}_sk'],
        f'{prefix}_sks': best[f'{prefix}_sks'],
        f'{prefix}_sd': best[f'{prefix}_sd'],
        f'{prefix}_lev': best[f'{prefix}_lev'],
        'cagr': best_result['cagr'],
        'mdd': best_result['mdd'],
        'sharpe': best_result['sharpe'],
        'trades': best_result['trades'],
    }
    if phase == 2 and prev_result:
        out['improvement'] = best_result['cagr'] - prev_result['cagr']
    return out


def optimize_secondary(data: dict, primary_params: dict, primary_dir: str,
                        phase: int, prev_result: dict = None) -> dict:
    """보조 방향 파라미터 최적화 (주 방향 고정, 결합 백테스트)"""
    sec_dir = 'short' if primary_dir == 'long' else 'long'
    sec_prefix = sec_dir
    pri_prefix = primary_dir
    n_trials = PHASE1_TRIALS if phase == 1 else PHASE2_TRIALS

    if phase == 2 and prev_result:
        ranges = {
            'ma_range': get_narrowed_range(prev_result[f'{sec_prefix}_ma'], PARAM_RANGES['ma_range']),
            'stoch_k_range': get_narrowed_range(prev_result[f'{sec_prefix}_sk'], PARAM_RANGES['stoch_k_range']),
            'stoch_smooth_range': get_narrowed_range(prev_result[f'{sec_prefix}_sks'], PARAM_RANGES['stoch_smooth_range']),
            'stoch_d_range': get_narrowed_range(prev_result[f'{sec_prefix}_sd'], PARAM_RANGES['stoch_d_range']),
            'leverage_range': PARAM_RANGES['leverage_range'],
        }
    else:
        ranges = PARAM_RANGES

    def objective(trial):
        ma = trial.suggest_int(f'{sec_prefix}_ma', *ranges['ma_range'])
        sk = trial.suggest_int(f'{sec_prefix}_sk', *ranges['stoch_k_range'])
        sks = trial.suggest_int(f'{sec_prefix}_sks', *ranges['stoch_smooth_range'])
        sd = trial.suggest_int(f'{sec_prefix}_sd', *ranges['stoch_d_range'])
        lev = trial.suggest_int(f'{sec_prefix}_lev', *ranges['leverage_range'])

        result = backtest_combined(
            data,
            primary_params[f'{pri_prefix}_ma'], primary_params[f'{pri_prefix}_sk'],
            primary_params[f'{pri_prefix}_sks'], primary_params[f'{pri_prefix}_sd'],
            primary_params[f'{pri_prefix}_lev'],
            ma, sk, sks, sd, lev,
            priority=primary_dir
        )
        cagr, mdd, sharpe = result['cagr'], result['mdd'], result['sharpe']
        if cagr < -50 or mdd < -90:
            return -9999
        if phase == 1:
            return cagr + (mdd * 0.5) + (sharpe * 15)
        else:
            return cagr + (mdd * 0.7) + (sharpe * 20)

    startup = min(20, n_trials // 2) if phase == 1 else min(30, n_trials // 3)
    sampler = TPESampler(seed=42, n_startup_trials=startup)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    if phase == 2 and prev_result:
        study.enqueue_trial({
            f'{sec_prefix}_ma': prev_result[f'{sec_prefix}_ma'],
            f'{sec_prefix}_sk': prev_result[f'{sec_prefix}_sk'],
            f'{sec_prefix}_sks': prev_result[f'{sec_prefix}_sks'],
            f'{sec_prefix}_sd': prev_result[f'{sec_prefix}_sd'],
            f'{sec_prefix}_lev': prev_result[f'{sec_prefix}_lev'],
        })

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_combined(
        data,
        primary_params[f'{pri_prefix}_ma'], primary_params[f'{pri_prefix}_sk'],
        primary_params[f'{pri_prefix}_sks'], primary_params[f'{pri_prefix}_sd'],
        primary_params[f'{pri_prefix}_lev'],
        best[f'{sec_prefix}_ma'], best[f'{sec_prefix}_sk'],
        best[f'{sec_prefix}_sks'], best[f'{sec_prefix}_sd'], best[f'{sec_prefix}_lev'],
        priority=primary_dir
    )

    out = {
        f'{sec_prefix}_ma': best[f'{sec_prefix}_ma'],
        f'{sec_prefix}_sk': best[f'{sec_prefix}_sk'],
        f'{sec_prefix}_sks': best[f'{sec_prefix}_sks'],
        f'{sec_prefix}_sd': best[f'{sec_prefix}_sd'],
        f'{sec_prefix}_lev': best[f'{sec_prefix}_lev'],
        'cagr': best_result['cagr'],
        'mdd': best_result['mdd'],
        'sharpe': best_result['sharpe'],
        'trades': best_result['trades'],
        'long_trades': best_result.get('long_trades', 0),
        'short_trades': best_result.get('short_trades', 0),
    }
    if phase == 2 and prev_result:
        out['improvement'] = best_result['cagr'] - prev_result['cagr']
    return out


# ==========================================
# 한 방향 우선 전체 파이프라인
# ==========================================

def run_pipeline(symbol: str, data: dict, primary_dir: str) -> dict:
    """
    primary_dir='long' → 롱 우선 파이프라인
    primary_dir='short' → 숏 우선 파이프라인
    """
    sec_dir = 'short' if primary_dir == 'long' else 'long'
    pri_label = '롱' if primary_dir == 'long' else '숏'
    sec_label = '숏' if primary_dir == 'long' else '롱'

    print(f"\n  ━━ [{pri_label} 우선] Stage 1: {pri_label} 파라미터 최적화 ━━")

    # Stage 1 Phase 1
    print(f"    Phase 1 ({PHASE1_TRIALS} trials)...", end=" ", flush=True)
    p1 = optimize_single_direction(data, primary_dir, phase=1)
    print(f"✅ CAGR:{p1['cagr']:.1f}% Lev:{p1[f'{primary_dir}_lev']}x "
          f"MA:{p1[f'{primary_dir}_ma']} "
          f"SK:{p1[f'{primary_dir}_sk']}/{p1[f'{primary_dir}_sks']}/{p1[f'{primary_dir}_sd']}")

    # Stage 1 Phase 2
    print(f"    Phase 2 ({PHASE2_TRIALS} trials)...", end=" ", flush=True)
    p2 = optimize_single_direction(data, primary_dir, phase=2, prev_result=p1)
    print(f"✅ CAGR:{p2['cagr']:.1f}% (개선:{p2['improvement']:+.1f}%) "
          f"MDD:{p2['mdd']:.1f}% Sharpe:{p2['sharpe']:.2f}")

    primary_only_cagr = p2['cagr']
    primary_params = {
        f'{primary_dir}_ma': p2[f'{primary_dir}_ma'],
        f'{primary_dir}_sk': p2[f'{primary_dir}_sk'],
        f'{primary_dir}_sks': p2[f'{primary_dir}_sks'],
        f'{primary_dir}_sd': p2[f'{primary_dir}_sd'],
        f'{primary_dir}_lev': p2[f'{primary_dir}_lev'],
    }

    # Stage 2: 보조 방향 최적화
    print(f"\n  ━━ [{pri_label} 우선] Stage 2: {sec_label} 파라미터 최적화 ({pri_label} 고정) ━━")

    print(f"    Phase 1 ({PHASE1_TRIALS} trials)...", end=" ", flush=True)
    s1 = optimize_secondary(data, primary_params, primary_dir, phase=1)
    print(f"✅ Combined CAGR:{s1['cagr']:.1f}% "
          f"{sec_label}:{s1.get(f'{sec_dir}_trades', s1.get('short_trades', 0))}회")

    print(f"    Phase 2 ({PHASE2_TRIALS} trials)...", end=" ", flush=True)
    s2 = optimize_secondary(data, primary_params, primary_dir, phase=2, prev_result=s1)
    print(f"✅ Combined CAGR:{s2['cagr']:.1f}% (개선:{s2['improvement']:+.1f}%) "
          f"MDD:{s2['mdd']:.1f}% Sharpe:{s2['sharpe']:.2f}")

    # Stage 3: 결합 검증
    combined_result = backtest_combined(
        data,
        primary_params[f'{primary_dir}_ma'], primary_params[f'{primary_dir}_sk'],
        primary_params[f'{primary_dir}_sks'], primary_params[f'{primary_dir}_sd'],
        primary_params[f'{primary_dir}_lev'],
        s2[f'{sec_dir}_ma'], s2[f'{sec_dir}_sk'],
        s2[f'{sec_dir}_sks'], s2[f'{sec_dir}_sd'], s2[f'{sec_dir}_lev'],
        priority=primary_dir
    )

    diff = combined_result['cagr'] - primary_only_cagr
    better = f"✅ {sec_label} 추가 효과" if diff > 0 else f"⚠️ {sec_label} 추가 시 하락"

    print(f"\n  ━━ [{pri_label} 우선] Stage 3: 결합 검증 ━━")
    print(f"    {pri_label}만 CAGR:  {primary_only_cagr:.1f}%")
    print(f"    결합 CAGR:   {combined_result['cagr']:.1f}% ({diff:+.1f}%) → {better}")
    print(f"    결합 MDD:    {combined_result['mdd']:.1f}%  Sharpe: {combined_result['sharpe']:.2f}")
    print(f"    거래수: 총 {combined_result['trades']}회 "
          f"(롱 {combined_result.get('long_trades', 0)}회, "
          f"숏 {combined_result.get('short_trades', 0)}회)")

    # 결과 취합
    if primary_dir == 'long':
        return {
            'primary_only_cagr': primary_only_cagr,
            'combined_cagr': combined_result['cagr'],
            'combined_mdd': combined_result['mdd'],
            'combined_sharpe': combined_result['sharpe'],
            'combined_trades': combined_result['trades'],
            'long_trades': combined_result.get('long_trades', 0),
            'short_trades': combined_result.get('short_trades', 0),
            'long_ma': primary_params['long_ma'],
            'long_sk': primary_params['long_sk'],
            'long_sks': primary_params['long_sks'],
            'long_sd': primary_params['long_sd'],
            'long_lev': primary_params['long_lev'],
            'short_ma': s2['short_ma'],
            'short_sk': s2['short_sk'],
            'short_sks': s2['short_sks'],
            'short_sd': s2['short_sd'],
            'short_lev': s2['short_lev'],
        }
    else:
        return {
            'primary_only_cagr': primary_only_cagr,
            'combined_cagr': combined_result['cagr'],
            'combined_mdd': combined_result['mdd'],
            'combined_sharpe': combined_result['sharpe'],
            'combined_trades': combined_result['trades'],
            'long_trades': combined_result.get('long_trades', 0),
            'short_trades': combined_result.get('short_trades', 0),
            'short_ma': primary_params['short_ma'],
            'short_sk': primary_params['short_sk'],
            'short_sks': primary_params['short_sks'],
            'short_sd': primary_params['short_sd'],
            'short_lev': primary_params['short_lev'],
            'long_ma': s2['long_ma'],
            'long_sk': s2['long_sk'],
            'long_sks': s2['long_sks'],
            'long_sd': s2['long_sd'],
            'long_lev': s2['long_lev'],
        }


# ==========================================
# 메인
# ==========================================

def main():
    print("=" * 90)
    print("📊 Binance Futures 메이저 코인: 롱 우선 vs 숏 우선 비교 최적화")
    print("=" * 90)
    print(f"📌 대상: {', '.join(TARGET_COINS)} ({len(TARGET_COINS)}개)")
    print(f"📌 수수료: {FEE_RATE*100:.2f}% | 슬리피지: {SLIPPAGE_PCT*100:.2f}% | 펀딩: {FUNDING_RATE_8H*100:.3f}%/8h")
    print(f"📌 각 방향: Phase1 {PHASE1_TRIALS} + Phase2 {PHASE2_TRIALS} trials × 2단계")
    print(f"📌 코인당 총 trials: {(PHASE1_TRIALS + PHASE2_TRIALS) * 2 * 2} (롱우선 + 숏우선)")
    print(f"📌 레버리지: {PARAM_RANGES['leverage_range'][0]}~{PARAM_RANGES['leverage_range'][1]}배")
    print("=" * 90)

    comparison_rows = []

    for idx, symbol in enumerate(TARGET_COINS, 1):
        print(f"\n{'#' * 90}")
        print(f"# [{idx}/{len(TARGET_COINS)}] {symbol}")
        print(f"{'#' * 90}")

        data = prepare_coin_data(symbol)
        if not data:
            print(f"  ❌ {symbol}: 데이터 준비 실패")
            continue

        print(f"  📅 기간: {data['start_date'].date()} ~ {data['end_date'].date()} ({data['days']}일)")

        # ── A) 롱 우선 ──
        print(f"\n{'─' * 90}")
        print(f"  🔵 [A] 롱 우선 전략")
        print(f"{'─' * 90}")
        long_first = run_pipeline(symbol, data, 'long')

        # ── B) 숏 우선 ──
        print(f"\n{'─' * 90}")
        print(f"  🔴 [B] 숏 우선 전략")
        print(f"{'─' * 90}")
        short_first = run_pipeline(symbol, data, 'short')

        # ── 비교 ──
        print(f"\n  {'=' * 70}")
        print(f"  📊 {symbol} 비교 결과")
        print(f"  {'=' * 70}")
        print(f"  {'항목':<16} {'롱 우선':>12} {'숏 우선':>12} {'차이':>10}")
        print(f"  {'─' * 50}")

        lf_cagr = long_first['combined_cagr']
        sf_cagr = short_first['combined_cagr']
        lf_mdd = long_first['combined_mdd']
        sf_mdd = short_first['combined_mdd']
        lf_sharpe = long_first['combined_sharpe']
        sf_sharpe = short_first['combined_sharpe']

        print(f"  {'주방향 CAGR':<16} {long_first['primary_only_cagr']:>11.1f}% {short_first['primary_only_cagr']:>11.1f}%")
        print(f"  {'결합 CAGR':<16} {lf_cagr:>11.1f}% {sf_cagr:>11.1f}% {sf_cagr - lf_cagr:>+9.1f}%")
        print(f"  {'결합 MDD':<16} {lf_mdd:>11.1f}% {sf_mdd:>11.1f}% {sf_mdd - lf_mdd:>+9.1f}%")
        print(f"  {'Sharpe':<16} {lf_sharpe:>12.2f} {sf_sharpe:>12.2f} {sf_sharpe - lf_sharpe:>+10.2f}")
        print(f"  {'거래수':<16} {long_first['combined_trades']:>12} {short_first['combined_trades']:>12}")

        # 승자 결정 (CAGR 기준, 동일하면 Sharpe)
        if lf_cagr > sf_cagr:
            winner = '롱 우선'
        elif sf_cagr > lf_cagr:
            winner = '숏 우선'
        else:
            winner = '롱 우선' if lf_sharpe >= sf_sharpe else '숏 우선'

        winner_emoji = '🔵' if winner == '롱 우선' else '🔴'
        print(f"\n  → {winner_emoji} 승자: {winner}")

        comparison_rows.append({
            'Symbol': symbol,
            'Days': data['days'],
            # 롱 우선
            'LF_Primary_CAGR': long_first['primary_only_cagr'],
            'LF_Combined_CAGR': lf_cagr,
            'LF_Combined_MDD': lf_mdd,
            'LF_Combined_Sharpe': lf_sharpe,
            'LF_Trades': long_first['combined_trades'],
            'LF_Long_Trades': long_first['long_trades'],
            'LF_Short_Trades': long_first['short_trades'],
            'LF_Long_MA': long_first['long_ma'],
            'LF_Long_SK': long_first['long_sk'],
            'LF_Long_SKs': long_first['long_sks'],
            'LF_Long_SD': long_first['long_sd'],
            'LF_Long_Lev': long_first['long_lev'],
            'LF_Short_MA': long_first['short_ma'],
            'LF_Short_SK': long_first['short_sk'],
            'LF_Short_SKs': long_first['short_sks'],
            'LF_Short_SD': long_first['short_sd'],
            'LF_Short_Lev': long_first['short_lev'],
            # 숏 우선
            'SF_Primary_CAGR': short_first['primary_only_cagr'],
            'SF_Combined_CAGR': sf_cagr,
            'SF_Combined_MDD': sf_mdd,
            'SF_Combined_Sharpe': sf_sharpe,
            'SF_Trades': short_first['combined_trades'],
            'SF_Long_Trades': short_first['long_trades'],
            'SF_Short_Trades': short_first['short_trades'],
            'SF_Short_MA': short_first['short_ma'],
            'SF_Short_SK': short_first['short_sk'],
            'SF_Short_SKs': short_first['short_sks'],
            'SF_Short_SD': short_first['short_sd'],
            'SF_Short_Lev': short_first['short_lev'],
            'SF_Long_MA': short_first['long_ma'],
            'SF_Long_SK': short_first['long_sk'],
            'SF_Long_SKs': short_first['long_sks'],
            'SF_Long_SD': short_first['long_sd'],
            'SF_Long_Lev': short_first['long_lev'],
            # 비교
            'Winner': winner,
            'CAGR_Diff': sf_cagr - lf_cagr,
        })

    # ==========================================
    # 최종 비교 테이블
    # ==========================================
    if not comparison_rows:
        print("\n❌ 결과가 없습니다.")
        return

    df = pd.DataFrame(comparison_rows)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(SAVE_DIR, f"binance_long_vs_short_first_compare_{ts}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 전체 결과 CSV: {csv_path}")

    # ── 요약 테이블 ──
    print(f"\n{'=' * 120}")
    print(f"📊 최종 비교 요약: 롱 우선 vs 숏 우선")
    print(f"{'=' * 120}")
    print(f"{'Symbol':<12} {'롱우선 CAGR':>11} {'롱우선 MDD':>10} {'롱우선 Sharpe':>12} "
          f"{'숏우선 CAGR':>11} {'숏우선 MDD':>10} {'숏우선 Sharpe':>12} {'승자':>8}")
    print(f"{'─' * 120}")

    for _, row in df.iterrows():
        w_mark = '🔵' if row['Winner'] == '롱 우선' else '🔴'
        print(f"{row['Symbol']:<12} "
              f"{row['LF_Combined_CAGR']:>10.1f}% {row['LF_Combined_MDD']:>9.1f}% {row['LF_Combined_Sharpe']:>12.2f} "
              f"{row['SF_Combined_CAGR']:>10.1f}% {row['SF_Combined_MDD']:>9.1f}% {row['SF_Combined_Sharpe']:>12.2f} "
              f"{w_mark} {row['Winner']}")

    # ── 승패 집계 ──
    long_wins = len(df[df['Winner'] == '롱 우선'])
    short_wins = len(df[df['Winner'] == '숏 우선'])
    total = len(df)

    print(f"\n{'=' * 80}")
    print(f"📊 종합 승패")
    print(f"{'=' * 80}")
    print(f"  🔵 롱 우선 승: {long_wins}/{total}개 ({long_wins/total*100:.0f}%)")
    print(f"  🔴 숏 우선 승: {short_wins}/{total}개 ({short_wins/total*100:.0f}%)")

    print(f"\n📊 평균 지표 비교:")
    print(f"  {'항목':<20} {'롱 우선':>12} {'숏 우선':>12}")
    print(f"  {'─' * 44}")
    print(f"  {'평균 결합 CAGR':<20} {df['LF_Combined_CAGR'].mean():>11.1f}% {df['SF_Combined_CAGR'].mean():>11.1f}%")
    print(f"  {'평균 결합 MDD':<20} {df['LF_Combined_MDD'].mean():>11.1f}% {df['SF_Combined_MDD'].mean():>11.1f}%")
    print(f"  {'평균 Sharpe':<20} {df['LF_Combined_Sharpe'].mean():>12.2f} {df['SF_Combined_Sharpe'].mean():>12.2f}")
    print(f"  {'평균 거래수':<20} {df['LF_Trades'].mean():>12.0f} {df['SF_Trades'].mean():>12.0f}")

    # ── 코인별 최적 전략 CONFIGS 출력 ──
    print(f"\n{'=' * 80}")
    print("📋 코인별 최적 전략 CONFIGS (각 코인 승자 기준)")
    print(f"{'=' * 80}")
    print("BEST_TRADING_CONFIGS = [")
    for _, row in df.iterrows():
        if row['Winner'] == '롱 우선':
            prefix_note = "long_first"
            print(f"    # {row['Symbol']} → 롱 우선 (CAGR:{row['LF_Combined_CAGR']:.1f}%)")
            print(f"    {{'symbol': '{row['Symbol']}', 'priority': 'long', "
                  f"'long_ma': {row['LF_Long_MA']}, 'long_sk': {row['LF_Long_SK']}, "
                  f"'long_sks': {row['LF_Long_SKs']}, 'long_sd': {row['LF_Long_SD']}, "
                  f"'long_lev': {row['LF_Long_Lev']}, "
                  f"'short_ma': {row['LF_Short_MA']}, 'short_sk': {row['LF_Short_SK']}, "
                  f"'short_sks': {row['LF_Short_SKs']}, 'short_sd': {row['LF_Short_SD']}, "
                  f"'short_lev': {row['LF_Short_Lev']}}},")
        else:
            print(f"    # {row['Symbol']} → 숏 우선 (CAGR:{row['SF_Combined_CAGR']:.1f}%)")
            print(f"    {{'symbol': '{row['Symbol']}', 'priority': 'short', "
                  f"'short_ma': {row['SF_Short_MA']}, 'short_sk': {row['SF_Short_SK']}, "
                  f"'short_sks': {row['SF_Short_SKs']}, 'short_sd': {row['SF_Short_SD']}, "
                  f"'short_lev': {row['SF_Short_Lev']}, "
                  f"'long_ma': {row['SF_Long_MA']}, 'long_sk': {row['SF_Long_SK']}, "
                  f"'long_sks': {row['SF_Long_SKs']}, 'long_sd': {row['SF_Long_SD']}, "
                  f"'long_lev': {row['SF_Long_Lev']}}},")
    print("]")

    overall = '롱 우선' if long_wins > short_wins else ('숏 우선' if short_wins > long_wins else '동률')
    print(f"\n✅ 비교 최적화 완료! 종합 우세: {overall}")


if __name__ == "__main__":
    main()
