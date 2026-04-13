"""
================================================================================
288코인 롱우선 vs 숏우선 비교 최적화 + 봉우리 검증
================================================================================
대상: binance_bot-5.py의 SHORT_TRADING_CONFIGS 288개 코인

각 코인에 대해:
  A) 롱 우선 전략: Stage1 롱 최적화 → Stage2 숏 최적화(롱 고정) → 결합
  B) 숏 우선 전략: Stage1 숏 최적화 → Stage2 롱 최적화(숏 고정) → 결합
  → 코인별 CAGR/MDD/Sharpe 비교, 최종 승패 집계

특징:
  - 중단/재개: 코인별 결과를 즉시 CSV에 저장, 이전 결과 자동 스킵
  - 진행률 + ETA 표시
  - 봉우리 검증: 최적화 후 CAGR순 Top10 후보를 순서대로 ±10% 이웃 검증
    → 첫 번째 강건/양호 후보 채택 (바늘/주의 자동 회피)
  - 강건성 재최적화: param_robustness_results.csv의 바늘/주의 코인은 다음 실행 시 자동 재최적화

비용 모델:
  - 수수료: 0.04% (노셔널 기준)
  - 슬리피지: 0.05% (노셔널 기준)
  - 펀딩비: 바이낸스 실제 펀딩비 사용 (USE_REAL_FUNDING=True)
    → 롱: 양수 펀딩비 시 비용, 음수 시 수익
    → 숏: 양수 펀딩비 시 수익, 음수 시 비용
    → 데이터 없는 구간은 0.01%/8h 폴백
  - 포지션 내 비복리, 거래 간 복리
  - 강제청산: 롱=entry*(1-1/lev), 숏=entry*(1+1/lev)
================================================================================
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import os
import sys
import time
import json
import warnings
import requests
from datetime import datetime, timedelta
from scipy.ndimage import uniform_filter

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 설정
# ==========================================

# 최적화 trials 수
PHASE1_TRIALS = 300
PHASE2_TRIALS = 300

# 비용 모델
FEE_RATE = 0.0004
SLIPPAGE_PCT = 0.0005
FUNDING_RATE_8H = 0.0001       # 실제 데이터 없을 때 폴백용
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5
USE_REAL_FUNDING = True        # True: 바이낸스 실제 펀딩비 사용

# 필터링
MIN_DATA_DAYS = 365          # 최소 데이터 기간 (일)
MIN_QUOTE_VOLUME = 0         # 최소 24시간 거래대금 (USDT), 0=필터 없음
EXCLUDE_STABLECOINS = True   # 스테이블코인 제외 (USDCUSDT 등)
SORT_BY_VOLUME = True        # 거래대금 순 정렬 (큰것부터)

# 파라미터 범위
PARAM_RANGES = {
    'ma_range': (20, 350),
    'stoch_k_range': (14, 150),
    'stoch_smooth_range': (5, 80),
    'stoch_d_range': (3, 50),
    'leverage_range': (1, 5),
}

# 경로
SAVE_DIR = os.path.expanduser("~/Downloads")
CACHE_DIR = os.path.expanduser("~/binance_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Resume용 결과 파일 (코인별 즉시 저장)
RESULT_CSV = os.path.join(SAVE_DIR, "all_coins_long_vs_short_results.csv")

# 강건성 검증 결과 (바늘/주의 코인 재최적화용)
ROBUSTNESS_CSV = os.path.join(SAVE_DIR, "param_robustness_results.csv")

# 강건성 검증 설정
ROBUSTNESS_MAX_RETRY = 2   # 바늘/주의 시 재시도 횟수
ROBUSTNESS_MAX_POINTS = 7  # 검증 그리드 포인트 (7^4 = 2401 조합)
ROBUSTNESS_PCT = 0.10      # ±10% 이웃 탐색
ROBUSTNESS_PASS_THRESHOLD = 0.65  # 이 이상이면 통과 (강건/양호)

# 288코인 소스 (binance_bot-5.py에서 심볼 추출)
BOT_FILE = os.path.join(SAVE_DIR, "binance_bot-5.py")

# 스테이블코인 심볼 패턴
STABLECOIN_PATTERNS = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'DAIUSDT',
                       'FDUSDUSDT', 'USDPUSDT', 'EURUSDT', 'GBPUSDT']


# ==========================================
# 288코인 심볼 수집 (binance_bot-5.py 기준)
# ==========================================

def load_288_symbols_from_bot() -> list:
    """
    binance_bot-5.py의 SHORT_TRADING_CONFIGS에서 288개 심볼 추출.
    Binance API로 24h 거래대금도 가져와 정렬.
    Returns: [{'symbol': 'BTCUSDT', 'quote_volume_24h': 12345678.0}, ...]
    """
    print(f"📡 288코인 심볼 로드: {BOT_FILE}")

    if not os.path.exists(BOT_FILE):
        print(f"  ❌ {BOT_FILE} 없음!")
        return []

    # binance_bot-5.py에서 SHORT_TRADING_CONFIGS 파싱 → 심볼 추출
    with open(BOT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        s_start = content.index('SHORT_TRADING_CONFIGS = [')
        s_end = content.index(']', s_start) + 1
        ns = {}
        exec(content[s_start:s_end], {}, ns)
        symbols_288 = [c['symbol'] for c in ns['SHORT_TRADING_CONFIGS']]
    except Exception as e:
        print(f"  ❌ SHORT_TRADING_CONFIGS 파싱 오류: {e}")
        return []

    symbols_set = set(symbols_288)
    print(f"  📋 봇 파일 심볼: {len(symbols_set)}개")

    # 24h Ticker - 거래대금 (정렬용)
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=30)
        tickers = resp.json()
        vol_map = {t['symbol']: float(t.get('quoteVolume', 0)) for t in tickers}
    except Exception as e:
        print(f"  ⚠️ ticker API 오류 (볼륨 정렬 불가): {e}")
        vol_map = {}

    result = []
    for sym in symbols_set:
        qv = vol_map.get(sym, 0)
        result.append({'symbol': sym, 'quote_volume_24h': qv})

    if SORT_BY_VOLUME:
        result.sort(key=lambda x: x['quote_volume_24h'], reverse=True)
    else:
        result.sort(key=lambda x: x['symbol'])

    print(f"  ✅ 대상: {len(result)}개 코인")
    if result:
        top5 = [f"{r['symbol']}(${r['quote_volume_24h']/1e6:.0f}M)" for r in result[:5]]
        print(f"  📊 거래대금 Top 5: {', '.join(top5)}")

    return result


# ==========================================
# 데이터 수집
# ==========================================

def get_funding_rates(symbol: str, silent: bool = False) -> pd.DataFrame:
    """바이낸스 실제 펀딩비 히스토리 다운로드 (캐시 지원)"""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_funding.csv")

    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mod_time).days < 1:
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            return df

    if not silent:
        print(f"  📥 펀딩비: {symbol}...", end=" ", flush=True)

    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    start_time = int(pd.Timestamp('2019-01-01').timestamp() * 1000)

    for _ in range(50):
        params = {'symbol': symbol, 'startTime': start_time, 'limit': 1000}
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(2)
                continue
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
        except Exception as e:
            if not silent:
                print(f"⚠️ {e}", end=" ")
            time.sleep(1)
            continue

    if not all_data:
        if not silent:
            print("❌")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['funding_rate'] = df['fundingRate'].astype(float)
    df = df[['timestamp', 'funding_rate']].sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    if not silent:
        print(f"✅ {len(df)}개")
    return df


def map_funding_to_4h(df_4h: pd.DataFrame, df_funding: pd.DataFrame) -> pd.DataFrame:
    """펀딩비를 4H 캔들에 매핑.
    8H 펀딩비(00:00, 08:00, 16:00 UTC)를 4H봉에 배분.
    각 4H봉에 해당 구간의 펀딩비 절반(4h/8h)을 할당."""
    if df_funding.empty:
        df_4h = df_4h.copy()
        df_4h['funding_rate_4h'] = FUNDING_PER_4H  # 폴백
        return df_4h

    df_4h = df_4h.copy()
    # merge_asof: 각 4H 타임스탬프에 가장 가까운 이전 펀딩비 매핑
    df_4h_sorted = df_4h.sort_values('timestamp')
    df_funding_sorted = df_funding.sort_values('timestamp')

    merged = pd.merge_asof(
        df_4h_sorted,
        df_funding_sorted.rename(columns={'funding_rate': 'funding_rate_8h'}),
        on='timestamp',
        direction='backward'
    )
    # 8H → 4H 비례 배분 (4H는 8H의 절반)
    merged['funding_rate_4h'] = merged['funding_rate_8h'] * 0.5
    # NaN인 경우 (펀딩 데이터보다 이전) 폴백
    merged['funding_rate_4h'] = merged['funding_rate_4h'].fillna(FUNDING_PER_4H)

    return merged


def get_binance_futures_klines(symbol: str, interval: str = '4h', silent: bool = False) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}_full.csv")

    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mod_time).days < 1:
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            if not silent:
                print(f"  📂 캐시: {symbol} ({interval}) - {len(df)}개")
            return df

    if not silent:
        print(f"  📥 다운로드: {symbol} ({interval})...", end=" ", flush=True)

    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)

    for _ in range(100):
        params = {'symbol': symbol, 'interval': interval, 'limit': 1500, 'endTime': end_time}
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(2)
                continue
            if response.status_code != 200:
                break
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
                print(f"⚠️ {e}", end=" ")
            time.sleep(1)
            continue

    if not all_data:
        if not silent:
            print("❌ 데이터 없음")
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
        print(f"✅ {len(df)}개")
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
            print(f"    ⚠️ 기간 부족: {days}일 < {MIN_DATA_DAYS}일")
        return None

    # 실제 펀딩비 로드
    df_funding = pd.DataFrame()
    if USE_REAL_FUNDING:
        df_funding = get_funding_rates(symbol, silent)
        if not df_funding.empty:
            df_4h = map_funding_to_4h(df_4h, df_funding)
        else:
            df_4h = df_4h.copy()
            df_4h['funding_rate_4h'] = FUNDING_PER_4H
    else:
        df_4h = df_4h.copy()
        df_4h['funding_rate_4h'] = FUNDING_PER_4H

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
# 단방향 백테스트
# ==========================================

def backtest_single_direction(data: dict, ma: int, sk: int, sks: int, sd: int, lev: int,
                               direction: str) -> dict:
    df_4h = data['df_4h']
    df_daily = data['df_daily']

    max_stoch = sk + sks + sd + 10
    if len(df_4h) < ma + 50 or len(df_daily) < max_stoch:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0,
                'long_trades': 0, 'short_trades': 0}

    df_sig = prepare_signals(df_4h, df_daily, ma, sk, sks, sd)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'ma',
            'prev_slow_k', 'prev_slow_d']
    if 'funding_rate_4h' in df_sig.columns:
        cols.append('funding_rate_4h')
    df_bt = df_sig[cols].copy()
    df_bt = df_bt.dropna(subset=['ma', 'prev_slow_k', 'prev_slow_d']).reset_index(drop=True)
    has_funding_col = 'funding_rate_4h' in df_bt.columns

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

        if signal and not in_pos:
            fee = equity * FEE_RATE * lev
            slippage = equity * SLIPPAGE_PCT * lev
            equity -= (fee + slippage)
            entry_margin = equity
            entry_price = opening_price
            cum_funding = 0.0
            in_pos = True
            trade_count += 1

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

        if in_pos and entry_price > 0:
            price_return = curr['close'] / entry_price - 1
            if not is_long:
                price_return = -price_return
            unrealized_pnl = entry_margin * price_return * lev
            # 실제 펀딩비 적용: 롱은 양수 펀딩비 시 지불, 숏은 양수 펀딩비 시 수취
            fr_4h = curr['funding_rate_4h'] if has_funding_col else FUNDING_PER_4H
            if is_long:
                cum_funding += entry_margin * lev * fr_4h       # 양수→비용, 음수→수익
            else:
                cum_funding += entry_margin * lev * (-fr_4h)    # 양수→수익, 음수→비용
            display_equity = entry_margin + unrealized_pnl - cum_funding

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
# 결합 백테스트
# ==========================================

def backtest_combined(data: dict,
                      pri_ma: int, pri_sk: int, pri_sks: int, pri_sd: int, pri_lev: int,
                      sec_ma: int, sec_sk: int, sec_sks: int, sec_sd: int, sec_lev: int,
                      priority: str = 'long') -> dict:
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

    cols_pri = ['timestamp', 'open', 'high', 'low', 'close', 'ma',
                'prev_slow_k', 'prev_slow_d']
    if 'funding_rate_4h' in df_pri.columns:
        cols_pri.append('funding_rate_4h')
    df_bt = df_pri[cols_pri].copy()
    df_bt.rename(columns={'ma': 'ma_pri', 'prev_slow_k': 'pri_k', 'prev_slow_d': 'pri_d'}, inplace=True)
    df_bt['ma_sec'] = df_sec['ma'].values
    df_bt['sec_k'] = df_sec['prev_slow_k'].values
    df_bt['sec_d'] = df_sec['prev_slow_d'].values
    has_funding_col = 'funding_rate_4h' in df_bt.columns

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

        if priority == 'long':
            pri_signal = (opening_price > prev['ma_pri']) and (prev['pri_k'] > prev['pri_d'])
            sec_signal = (opening_price < prev['ma_sec']) and (prev['sec_k'] < prev['sec_d'])
        else:
            pri_signal = (opening_price < prev['ma_pri']) and (prev['pri_k'] < prev['pri_d'])
            sec_signal = (opening_price > prev['ma_sec']) and (prev['sec_k'] > prev['sec_d'])

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

        if priority == 'long':
            if pri_signal:
                if in_short:
                    close_position()
                if not in_long and equity > 0:
                    enter_position('long', pri_lev)
            elif sec_signal and not in_long:
                if not in_short and equity > 0:
                    enter_position('short', sec_lev)
            else:
                if in_long:
                    close_position()
                if in_short:
                    close_position()
        else:
            if pri_signal:
                if in_long:
                    close_position()
                if not in_short and equity > 0:
                    enter_position('short', pri_lev)
            elif sec_signal and not in_short:
                if not in_long and equity > 0:
                    enter_position('long', sec_lev)
            else:
                if in_short:
                    close_position()
                if in_long:
                    close_position()

        if (in_long or in_short) and entry_price > 0:
            pr = curr['close'] / entry_price - 1
            if in_short:
                pr = -pr
            unrealized_pnl = entry_margin * pr * current_lev
            # 실제 펀딩비 적용
            fr_4h = curr['funding_rate_4h'] if has_funding_col else FUNDING_PER_4H
            if in_long:
                cum_funding += entry_margin * current_lev * fr_4h
            else:
                cum_funding += entry_margin * current_lev * (-fr_4h)
            display_equity = entry_margin + unrealized_pnl - cum_funding

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
# 최적화 함수
# ==========================================

def get_narrowed_range(center, original_range, ratio=0.3):
    min_val, max_val = original_range
    delta = int((max_val - min_val) * ratio)
    return (max(min_val, center - delta), min(max_val, center + delta))


def optimize_single_direction(data: dict, direction: str, phase: int,
                               prev_result: dict = None) -> dict:
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
        '_study': study,  # 상위 후보 접근용
    }
    if phase == 2 and prev_result:
        out['improvement'] = best_result['cagr'] - prev_result['cagr']
    return out


def optimize_secondary(data: dict, primary_params: dict, primary_dir: str,
                        phase: int, prev_result: dict = None) -> dict:
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
        '_study': study,  # 상위 후보 접근용
    }
    if phase == 2 and prev_result:
        out['improvement'] = best_result['cagr'] - prev_result['cagr']
    return out


def get_top_candidates(study, prefix, top_n=10):
    """study에서 상위 N개 후보 추출 (objective값 순)"""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    candidates = []
    seen = set()
    for t in completed[:top_n * 2]:  # 중복 제거 위해 여유있게
        p = t.params
        key = (p[f'{prefix}_ma'], p[f'{prefix}_sk'], p[f'{prefix}_sks'], p[f'{prefix}_sd'])
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            f'{prefix}_ma': p[f'{prefix}_ma'],
            f'{prefix}_sk': p[f'{prefix}_sk'],
            f'{prefix}_sks': p[f'{prefix}_sks'],
            f'{prefix}_sd': p[f'{prefix}_sd'],
            f'{prefix}_lev': p[f'{prefix}_lev'],
            '_score': t.value,
        })
        if len(candidates) >= top_n:
            break
    return candidates


# ==========================================
# 파이프라인
# ==========================================

def _build_result(primary_dir, primary_params, sec_result, combined_result, primary_only_cagr, robustness_info=None):
    """파이프라인 결과 dict 생성"""
    sec_dir = 'short' if primary_dir == 'long' else 'long'
    out = {
        'primary_only_cagr': primary_only_cagr,
        'combined_cagr': combined_result['cagr'],
        'combined_mdd': combined_result['mdd'],
        'combined_sharpe': combined_result['sharpe'],
        'combined_trades': combined_result['trades'],
        'long_trades': combined_result.get('long_trades', 0),
        'short_trades': combined_result.get('short_trades', 0),
        f'{primary_dir}_ma': primary_params[f'{primary_dir}_ma'],
        f'{primary_dir}_sk': primary_params[f'{primary_dir}_sk'],
        f'{primary_dir}_sks': primary_params[f'{primary_dir}_sks'],
        f'{primary_dir}_sd': primary_params[f'{primary_dir}_sd'],
        f'{primary_dir}_lev': primary_params[f'{primary_dir}_lev'],
        f'{sec_dir}_ma': sec_result[f'{sec_dir}_ma'],
        f'{sec_dir}_sk': sec_result[f'{sec_dir}_sk'],
        f'{sec_dir}_sks': sec_result[f'{sec_dir}_sks'],
        f'{sec_dir}_sd': sec_result[f'{sec_dir}_sd'],
        f'{sec_dir}_lev': sec_result[f'{sec_dir}_lev'],
    }
    if robustness_info:
        out.update(robustness_info)
    return out


def _run_secondary_and_combine(data, primary_params, primary_dir):
    """Stage2(보조방향) 최적화 + 결합 백테스트 실행"""
    sec_dir = 'short' if primary_dir == 'long' else 'long'
    s1 = optimize_secondary(data, primary_params, primary_dir, phase=1)
    s2 = optimize_secondary(data, primary_params, primary_dir, phase=2, prev_result=s1)

    combined_result = backtest_combined(
        data,
        primary_params[f'{primary_dir}_ma'], primary_params[f'{primary_dir}_sk'],
        primary_params[f'{primary_dir}_sks'], primary_params[f'{primary_dir}_sd'],
        primary_params[f'{primary_dir}_lev'],
        s2[f'{sec_dir}_ma'], s2[f'{sec_dir}_sk'],
        s2[f'{sec_dir}_sks'], s2[f'{sec_dir}_sd'], s2[f'{sec_dir}_lev'],
        priority=primary_dir
    )
    return s2, combined_result


def run_pipeline(symbol: str, data: dict, primary_dir: str, quiet: bool = False) -> dict:
    sec_dir = 'short' if primary_dir == 'long' else 'long'
    pri_label = '롱' if primary_dir == 'long' else '숏'
    sec_label = '숏' if primary_dir == 'long' else '롱'
    TOP_N = 10  # 강건성 검증 최대 후보 수

    if not quiet:
        print(f"    [{pri_label}우선] Stage1...", end=" ", flush=True)

    # ── Stage1: Primary 방향 최적화 ──
    p1 = optimize_single_direction(data, primary_dir, phase=1)
    p2 = optimize_single_direction(data, primary_dir, phase=2, prev_result=p1)
    primary_study = p2.pop('_study', None)

    if not quiet:
        print(f"CAGR:{p2['cagr']:.1f}%", end=" → ", flush=True)

    # 상위 후보 추출 (1순위 = Phase2 best)
    primary_candidates = []
    if primary_study:
        primary_candidates = get_top_candidates(primary_study, primary_dir, top_n=TOP_N)

    # 1순위 후보 = Phase2 best
    best_primary = {
        f'{primary_dir}_ma': p2[f'{primary_dir}_ma'],
        f'{primary_dir}_sk': p2[f'{primary_dir}_sk'],
        f'{primary_dir}_sks': p2[f'{primary_dir}_sks'],
        f'{primary_dir}_sd': p2[f'{primary_dir}_sd'],
        f'{primary_dir}_lev': p2[f'{primary_dir}_lev'],
    }

    primary_only_cagr = p2['cagr']

    # ── Stage2: Secondary 방향 최적화 ──
    if not quiet:
        print(f"Stage2...", end=" ", flush=True)

    s2, combined_result = _run_secondary_and_combine(data, best_primary, primary_dir)
    sec_study = s2.pop('_study', None)

    if not quiet:
        diff = combined_result['cagr'] - primary_only_cagr
        print(f"Combined:{combined_result['cagr']:.1f}%({diff:+.1f}%) "
              f"MDD:{combined_result['mdd']:.1f}% Sharpe:{combined_result['sharpe']:.2f}")

    # ── 강건성 검증: Primary 방향 (상위 후보 순) ──
    if not quiet:
        print(f"    [{pri_label}우선] 강건성 검증...", end=" ", flush=True)

    # 현재 best의 풀 config 구성
    full_config = {
        'priority': primary_dir,
        f'{primary_dir}_ma': best_primary[f'{primary_dir}_ma'],
        f'{primary_dir}_sk': best_primary[f'{primary_dir}_sk'],
        f'{primary_dir}_sks': best_primary[f'{primary_dir}_sks'],
        f'{primary_dir}_sd': best_primary[f'{primary_dir}_sd'],
        f'{primary_dir}_lev': best_primary[f'{primary_dir}_lev'],
        f'{sec_dir}_ma': s2[f'{sec_dir}_ma'],
        f'{sec_dir}_sk': s2[f'{sec_dir}_sk'],
        f'{sec_dir}_sks': s2[f'{sec_dir}_sks'],
        f'{sec_dir}_sd': s2[f'{sec_dir}_sd'],
        f'{sec_dir}_lev': s2[f'{sec_dir}_lev'],
    }

    # Primary 방향 검증
    pri_check = robustness_check_direction(data, full_config, primary_dir,
                                            max_points=ROBUSTNESS_MAX_POINTS,
                                            pct=ROBUSTNESS_PCT)
    pri_verdict = pri_check['verdict']
    pri_ratio = pri_check['smooth_ratio']

    adopted_rank = 1
    if pri_ratio < ROBUSTNESS_PASS_THRESHOLD and primary_candidates:
        # 1순위 실패 → 2순위부터 순서대로 검증
        if not quiet:
            print(f"{pri_label} {pri_verdict}(R={pri_ratio:.2f})→후보탐색", end=" ", flush=True)
        for rank, cand in enumerate(primary_candidates[1:], 2):  # [1:]은 1순위 건너뛰기
            cand_config = dict(full_config)
            cand_config[f'{primary_dir}_ma'] = cand[f'{primary_dir}_ma']
            cand_config[f'{primary_dir}_sk'] = cand[f'{primary_dir}_sk']
            cand_config[f'{primary_dir}_sks'] = cand[f'{primary_dir}_sks']
            cand_config[f'{primary_dir}_sd'] = cand[f'{primary_dir}_sd']
            cand_config[f'{primary_dir}_lev'] = cand[f'{primary_dir}_lev']

            c_check = robustness_check_direction(data, cand_config, primary_dir,
                                                  max_points=ROBUSTNESS_MAX_POINTS,
                                                  pct=ROBUSTNESS_PCT)
            if c_check['smooth_ratio'] >= ROBUSTNESS_PASS_THRESHOLD:
                # 통과! → 이 후보로 교체, Stage2 재실행
                new_primary = {
                    f'{primary_dir}_ma': cand[f'{primary_dir}_ma'],
                    f'{primary_dir}_sk': cand[f'{primary_dir}_sk'],
                    f'{primary_dir}_sks': cand[f'{primary_dir}_sks'],
                    f'{primary_dir}_sd': cand[f'{primary_dir}_sd'],
                    f'{primary_dir}_lev': cand[f'{primary_dir}_lev'],
                }
                if not quiet:
                    print(f"#{rank}통과(R={c_check['smooth_ratio']:.2f})→Stage2재실행", end=" ", flush=True)
                s2, combined_result = _run_secondary_and_combine(data, new_primary, primary_dir)
                sec_study = s2.pop('_study', None)
                best_primary = new_primary
                pri_verdict = c_check['verdict']
                pri_ratio = c_check['smooth_ratio']
                adopted_rank = rank
                # full_config 갱신
                full_config.update(new_primary)
                full_config[f'{sec_dir}_ma'] = s2[f'{sec_dir}_ma']
                full_config[f'{sec_dir}_sk'] = s2[f'{sec_dir}_sk']
                full_config[f'{sec_dir}_sks'] = s2[f'{sec_dir}_sks']
                full_config[f'{sec_dir}_sd'] = s2[f'{sec_dir}_sd']
                full_config[f'{sec_dir}_lev'] = s2[f'{sec_dir}_lev']
                break
        else:
            # 모든 후보 실패 → 1순위 유지 (최선)
            if not quiet:
                print(f"후보{len(primary_candidates)}개 모두 미달→1순위유지", end=" ", flush=True)

    # ── 강건성 검증: Secondary 방향 (상위 후보 순) ──
    sec_check = robustness_check_direction(data, full_config, sec_dir,
                                            max_points=ROBUSTNESS_MAX_POINTS,
                                            pct=ROBUSTNESS_PCT)
    sec_verdict = sec_check['verdict']
    sec_ratio = sec_check['smooth_ratio']

    sec_adopted_rank = 1
    sec_candidates = []
    if sec_study:
        sec_candidates = get_top_candidates(sec_study, sec_dir, top_n=TOP_N)

    if sec_ratio < ROBUSTNESS_PASS_THRESHOLD and sec_candidates:
        if not quiet:
            print(f"{sec_label} {sec_verdict}(R={sec_ratio:.2f})→후보탐색", end=" ", flush=True)
        for rank, cand in enumerate(sec_candidates[1:], 2):
            cand_config = dict(full_config)
            cand_config[f'{sec_dir}_ma'] = cand[f'{sec_dir}_ma']
            cand_config[f'{sec_dir}_sk'] = cand[f'{sec_dir}_sk']
            cand_config[f'{sec_dir}_sks'] = cand[f'{sec_dir}_sks']
            cand_config[f'{sec_dir}_sd'] = cand[f'{sec_dir}_sd']
            cand_config[f'{sec_dir}_lev'] = cand[f'{sec_dir}_lev']

            c_check = robustness_check_direction(data, cand_config, sec_dir,
                                                  max_points=ROBUSTNESS_MAX_POINTS,
                                                  pct=ROBUSTNESS_PCT)
            if c_check['smooth_ratio'] >= ROBUSTNESS_PASS_THRESHOLD:
                if not quiet:
                    print(f"#{rank}통과(R={c_check['smooth_ratio']:.2f})", end=" ", flush=True)
                # secondary 파라미터 교체
                s2[f'{sec_dir}_ma'] = cand[f'{sec_dir}_ma']
                s2[f'{sec_dir}_sk'] = cand[f'{sec_dir}_sk']
                s2[f'{sec_dir}_sks'] = cand[f'{sec_dir}_sks']
                s2[f'{sec_dir}_sd'] = cand[f'{sec_dir}_sd']
                s2[f'{sec_dir}_lev'] = cand[f'{sec_dir}_lev']
                # 결합 백테스트 재실행
                combined_result = backtest_combined(
                    data,
                    best_primary[f'{primary_dir}_ma'], best_primary[f'{primary_dir}_sk'],
                    best_primary[f'{primary_dir}_sks'], best_primary[f'{primary_dir}_sd'],
                    best_primary[f'{primary_dir}_lev'],
                    s2[f'{sec_dir}_ma'], s2[f'{sec_dir}_sk'],
                    s2[f'{sec_dir}_sks'], s2[f'{sec_dir}_sd'], s2[f'{sec_dir}_lev'],
                    priority=primary_dir
                )
                sec_verdict = c_check['verdict']
                sec_ratio = c_check['smooth_ratio']
                sec_adopted_rank = rank
                break
        else:
            if not quiet:
                print(f"후보{len(sec_candidates)}개 모두 미달→1순위유지", end=" ", flush=True)

    # ── 결과 출력 ──
    robustness_info = {
        'pri_verdict': pri_verdict,
        'pri_ratio': pri_ratio,
        'pri_rank': adopted_rank,
        'sec_verdict': sec_verdict,
        'sec_ratio': sec_ratio,
        'sec_rank': sec_adopted_rank,
    }

    if not quiet:
        pri_tag = f"{pri_label}:{pri_verdict}(R={pri_ratio:.2f},#{adopted_rank})"
        sec_tag = f"{sec_label}:{sec_verdict}(R={sec_ratio:.2f},#{sec_adopted_rank})"
        print(f"\n    [{pri_label}우선] 최종: {combined_result['cagr']:.1f}%/{combined_result['mdd']:.1f}% "
              f"강건성[{pri_tag} {sec_tag}]")

    return _build_result(primary_dir, best_primary, s2, combined_result,
                          primary_only_cagr, robustness_info)


# ==========================================
# 강건성 검증 (봉우리 vs 바늘)
# ==========================================

def _make_grid(center, pct=0.10, min_radius=3, max_points=7, min_val=1):
    """중심값 기준 ±N% 범위, 최대 max_points개"""
    radius = max(min_radius, int(round(center * pct)))
    lo = max(min_val, center - radius)
    hi = center + radius
    values = list(range(lo, hi + 1))
    if len(values) <= max_points:
        return values
    step = max(1, len(values) // (max_points - 1))
    grid = list(range(lo, hi + 1, step))
    if center not in grid:
        grid.append(center)
        grid.sort()
    while len(grid) > max_points + 1:
        grid.pop()
    return grid


def _precompute_indicators(data, ma_range, sk_range, sks_range, sd_range):
    """지정 범위의 MA / 스토캐스틱 지표 사전계산"""
    df_4h = data['df_4h'].sort_values('timestamp').reset_index(drop=True)
    df_daily = data['df_daily'].sort_values('timestamp').reset_index(drop=True)
    n = len(df_4h)

    opens = df_4h['open'].values.astype(np.float64)
    highs = df_4h['high'].values.astype(np.float64)
    lows = df_4h['low'].values.astype(np.float64)
    closes = df_4h['close'].values.astype(np.float64)

    d_closes = df_daily['close'].values.astype(np.float64)
    d_dates = pd.to_datetime(df_daily['timestamp']).dt.date.values
    h_dates = pd.to_datetime(df_4h['timestamp']).dt.date.values

    # MA (일봉)
    prev_mas = {}
    for ma in ma_range:
        sma = pd.Series(d_closes).rolling(ma).mean().values
        daily_prev = np.empty(len(d_dates)); daily_prev[:] = np.nan
        daily_prev[1:] = sma[:-1]
        day_map = {}
        for i, d in enumerate(d_dates):
            if not np.isnan(daily_prev[i]):
                day_map[d] = daily_prev[i]
        arr = np.empty(n); arr[:] = np.nan
        for i, d in enumerate(h_dates):
            arr[i] = day_map.get(d, np.nan)
        prev_mas[ma] = arr

    # 스토캐스틱 (4시간봉)
    prev_stochs = {}
    for sk in sk_range:
        roll_low = pd.Series(lows).rolling(sk).min().values
        roll_high = pd.Series(highs).rolling(sk).max().values
        diff = roll_high - roll_low
        diff[diff == 0] = 1
        raw_k = (closes - roll_low) / diff * 100
        for sks in sks_range:
            smooth_k = pd.Series(raw_k).rolling(sks).mean().values
            for sd in sd_range:
                d_line = pd.Series(smooth_k).rolling(sd).mean().values
                prev_sk = np.empty(n); prev_sk[:] = np.nan
                prev_sd = np.empty(n); prev_sd[:] = np.nan
                prev_sk[1:] = smooth_k[:-1]
                prev_sd[1:] = d_line[:-1]
                prev_stochs[(sk, sks, sd)] = (prev_sk, prev_sd)

    # 펀딩비 배열
    if 'funding_rate_4h' in df_4h.columns:
        funding_rates = df_4h['funding_rate_4h'].values.astype(np.float64)
    else:
        funding_rates = np.full(n, FUNDING_PER_4H)

    return opens, highs, lows, closes, n, prev_mas, prev_stochs, funding_rates


def _fast_bt(opens, highs, lows, closes, n,
             long_prev_ma, long_prev_sk, long_prev_sd,
             short_prev_ma, short_prev_sk, short_prev_sd,
             long_lev, short_lev, priority_is_long,
             funding_rates=None):
    """단일 파라미터 세트 빠른 백테스트 → (sharpe, cagr)"""
    equity = 1.0
    pos = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    lev = 0

    equities = []
    start_idx = 100

    for i in range(start_idx, n):
        c = closes[i]; o = opens[i]; h = highs[i]; lo_ = lows[i]

        # 청산 체크
        if pos == 1:
            liq = entry_price * (1.0 - 1.0 / lev) if lev > 0 else 0
            if lo_ <= liq:
                equity *= (1.0 - FEE_RATE - SLIPPAGE_PCT)
                equity *= 0.01
                pos = 0
        elif pos == -1:
            liq = entry_price * (1.0 + 1.0 / lev) if lev > 0 else 1e18
            if h >= liq:
                equity *= (1.0 - FEE_RATE - SLIPPAGE_PCT)
                equity *= 0.01
                pos = 0

        # 시그널 체크
        l_ma = long_prev_ma[i]; l_sk = long_prev_sk[i]; l_sd = long_prev_sd[i]
        s_ma = short_prev_ma[i]; s_sk = short_prev_sk[i]; s_sd = short_prev_sd[i]

        if np.isnan(l_ma) or np.isnan(l_sk) or np.isnan(l_sd) or \
           np.isnan(s_ma) or np.isnan(s_sk) or np.isnan(s_sd):
            equities.append(equity)
            continue

        long_signal = (c > l_ma) and (l_sk > l_sd) and (l_sk < 80)
        short_signal = (c < s_ma) and (s_sk < s_sd) and (s_sk > 20)

        want_long = long_signal
        want_short = short_signal

        if priority_is_long:
            if want_long:
                want_short = False
        else:
            if want_short:
                want_long = False

        # 포지션 전환
        if pos == 0:
            if want_long:
                pos = 1; entry_price = c; lev = long_lev
                equity *= (1.0 - FEE_RATE - SLIPPAGE_PCT)
            elif want_short:
                pos = -1; entry_price = c; lev = short_lev
                equity *= (1.0 - FEE_RATE - SLIPPAGE_PCT)
        elif pos == 1:
            if not want_long:
                pnl = (c - entry_price) / entry_price * lev
                equity *= (1.0 + pnl - FEE_RATE - SLIPPAGE_PCT)
                # 실제 펀딩비: 롱은 양수 펀딩비 시 비용
                fr = funding_rates[i] if funding_rates is not None else FUNDING_PER_4H
                equity *= (1.0 - fr * lev)
                pos = 0
                if want_short:
                    pos = -1; entry_price = c; lev = short_lev
                    equity *= (1.0 - FEE_RATE - SLIPPAGE_PCT)
        elif pos == -1:
            if not want_short:
                pnl = (entry_price - c) / entry_price * lev
                equity *= (1.0 + pnl - FEE_RATE - SLIPPAGE_PCT)
                # 실제 펀딩비: 숏은 양수 펀딩비 시 수익 (부호 반전)
                fr = funding_rates[i] if funding_rates is not None else FUNDING_PER_4H
                equity *= (1.0 + fr * lev)
                pos = 0
                if want_long:
                    pos = 1; entry_price = c; lev = long_lev
                    equity *= (1.0 - FEE_RATE - SLIPPAGE_PCT)

        if equity < 0.001:
            equity = 0.001
        equities.append(equity)

    if len(equities) < 50:
        return (-999.0, -999.0)

    eq = np.array(equities)
    total_ret = eq[-1] / eq[0] if eq[0] > 0 else 0
    days = len(eq) / 6.0
    years = days / 365.0
    cagr = ((total_ret ** (1.0 / years)) - 1.0) * 100 if years > 0 and total_ret > 0 else -100.0

    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 50 or np.std(rets) == 0:
        return (-999.0, cagr)
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(365 * 6)
    return (sharpe, cagr)


def robustness_check_direction(data, config, direction, max_points=7, pct=0.10):
    """
    한 방향의 파라미터 이웃 탐색 → smooth_ratio, verdict 반환.
    config: {'priority': 'long'|'short', 'long_ma':..., 'short_ma':..., ...}
    """
    priority_is_long = (config['priority'] == 'long')

    s_ma = config[f'{direction}_ma']
    s_sk = config[f'{direction}_sk']
    s_sks = config[f'{direction}_sks']
    s_sd = config[f'{direction}_sd']
    s_lev = config[f'{direction}_lev']

    fix_dir = 'long' if direction == 'short' else 'short'
    f_ma = config[f'{fix_dir}_ma']
    f_sk = config[f'{fix_dir}_sk']
    f_sks = config[f'{fix_dir}_sks']
    f_sd = config[f'{fix_dir}_sd']
    f_lev = config[f'{fix_dir}_lev']

    ma_range = _make_grid(s_ma, pct=pct, max_points=max_points, min_val=5)
    sk_range = _make_grid(s_sk, pct=pct, max_points=max_points, min_val=2)
    sks_range = _make_grid(s_sks, pct=pct, max_points=max_points, min_val=2)
    sd_range = _make_grid(s_sd, pct=pct, max_points=max_points, min_val=2)

    all_ma = sorted(set(ma_range + [f_ma]))
    all_sk = sorted(set(sk_range + [f_sk]))
    all_sks = sorted(set(sks_range + [f_sks]))
    all_sd = sorted(set(sd_range + [f_sd]))

    opens, highs, lows, closes, n, prev_mas, prev_stochs, funding_rates = \
        _precompute_indicators(data, all_ma, all_sk, all_sks, all_sd)

    fixed_prev_ma = prev_mas[f_ma]
    fixed_stoch = prev_stochs.get((f_sk, f_sks, f_sd))
    if fixed_stoch is None:
        return {'smooth_ratio': 0.0, 'verdict': '바늘'}
    fixed_prev_sk, fixed_prev_sd = fixed_stoch

    search_is_long = (direction == 'long')

    n_ma = len(ma_range); n_sk = len(sk_range)
    n_sks = len(sks_range); n_sd = len(sd_range)
    sharpe_grid = np.full((n_ma, n_sk, n_sks, n_sd), -999.0)
    center_idx = None

    for i_ma, ma in enumerate(ma_range):
        search_ma = prev_mas.get(ma)
        if search_ma is None:
            continue
        for i_sk, sk in enumerate(sk_range):
            for i_sks, sks in enumerate(sks_range):
                for i_sd, sd in enumerate(sd_range):
                    stoch = prev_stochs.get((sk, sks, sd))
                    if stoch is None:
                        continue
                    s_skv, s_sdv = stoch
                    if search_is_long:
                        result = _fast_bt(opens, highs, lows, closes, n,
                                          search_ma, s_skv, s_sdv,
                                          fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
                                          s_lev, f_lev, priority_is_long,
                                          funding_rates)
                    else:
                        result = _fast_bt(opens, highs, lows, closes, n,
                                          fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
                                          search_ma, s_skv, s_sdv,
                                          f_lev, s_lev, priority_is_long,
                                          funding_rates)
                    sharpe_grid[i_ma, i_sk, i_sks, i_sd] = result[0]
                    if ma == s_ma and sk == s_sk and sks == s_sks and sd == s_sd:
                        center_idx = (i_ma, i_sk, i_sks, i_sd)

    valid = sharpe_grid > -900
    n_valid = valid.sum()

    if center_idx is None or n_valid < 10:
        return {'smooth_ratio': 0.0, 'verdict': '바늘'}

    center_sharpe = float(sharpe_grid[center_idx])

    sg = sharpe_grid.copy(); sg[~valid] = 0
    cnt = valid.astype(float)
    s_sum = uniform_filter(sg, size=3, mode='constant', cval=0)
    c_sum = uniform_filter(cnt, size=3, mode='constant', cval=0)
    c_sum[c_sum == 0] = 1
    smoothed = s_sum / c_sum
    smoothed[~valid] = -999
    smoothed_at_center = float(smoothed[center_idx])

    if center_sharpe > 0:
        smooth_ratio = smoothed_at_center / center_sharpe
    else:
        smooth_ratio = 1.0 if smoothed_at_center <= 0 else 0.0

    if smooth_ratio >= 0.85:
        verdict = '강건'
    elif smooth_ratio >= 0.65:
        verdict = '양호'
    elif smooth_ratio >= 0.45:
        verdict = '주의'
    else:
        verdict = '바늘'

    return {'smooth_ratio': smooth_ratio, 'verdict': verdict}


def check_robustness(data, pipeline_result, priority_dir):
    """
    최적화 결과의 양방향 강건성 검증.
    Returns: {'long_verdict':..., 'short_verdict':..., 'long_ratio':..., 'short_ratio':..., 'pass': bool}
    """
    config = {
        'priority': priority_dir,
        'long_ma': pipeline_result['long_ma'],
        'long_sk': pipeline_result['long_sk'],
        'long_sks': pipeline_result['long_sks'],
        'long_sd': pipeline_result['long_sd'],
        'long_lev': pipeline_result['long_lev'],
        'short_ma': pipeline_result['short_ma'],
        'short_sk': pipeline_result['short_sk'],
        'short_sks': pipeline_result['short_sks'],
        'short_sd': pipeline_result['short_sd'],
        'short_lev': pipeline_result['short_lev'],
    }

    long_r = robustness_check_direction(data, config, 'long',
                                         max_points=ROBUSTNESS_MAX_POINTS,
                                         pct=ROBUSTNESS_PCT)
    short_r = robustness_check_direction(data, config, 'short',
                                          max_points=ROBUSTNESS_MAX_POINTS,
                                          pct=ROBUSTNESS_PCT)

    passed = (long_r['smooth_ratio'] >= ROBUSTNESS_PASS_THRESHOLD and
              short_r['smooth_ratio'] >= ROBUSTNESS_PASS_THRESHOLD)

    return {
        'long_verdict': long_r['verdict'],
        'short_verdict': short_r['verdict'],
        'long_ratio': long_r['smooth_ratio'],
        'short_ratio': short_r['smooth_ratio'],
        'pass': passed,
    }


# ==========================================
# Resume 지원: 결과 저장/로드
# ==========================================

def get_weak_coins_from_robustness() -> set:
    """강건성 검증에서 '주의' 또는 '바늘' 판정된 코인 목록 반환"""
    if not os.path.exists(ROBUSTNESS_CSV):
        return set()
    try:
        df = pd.read_csv(ROBUSTNESS_CSV)
        # 288coin_HYBRID 그룹에서 주의/바늘 판정된 코인
        bad = df[(df['group'] == '288coin_HYBRID') & (df['verdict'].isin(['주의', '바늘']))]
        return set(bad['symbol'].unique())
    except Exception:
        return set()


def load_completed_symbols() -> tuple:
    """
    이미 완료된 심볼 목록 로드.
    강건성 검증에서 '주의'/'바늘' 판정된 코인은 완료 목록에서 제외하고,
    기존 CSV에서도 해당 행을 삭제하여 재최적화 대상으로 만듦.
    Returns: (completed_set, weak_coins_set)
    """
    weak_coins = get_weak_coins_from_robustness()

    if not os.path.exists(RESULT_CSV):
        return set(), weak_coins

    try:
        df = pd.read_csv(RESULT_CSV)
        all_completed = set(df['Symbol'].unique())

        # 바늘/주의 코인이 기존 결과에 있으면 삭제
        reopt_targets = all_completed & weak_coins
        if reopt_targets:
            df_clean = df[~df['Symbol'].isin(reopt_targets)]
            df_clean.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
            print(f"\n🔄 강건성 재최적화: {len(reopt_targets)}개 코인 기존 결과 삭제")
            for s in sorted(reopt_targets):
                print(f"   → {s}")
            all_completed -= reopt_targets

        return all_completed, weak_coins
    except Exception:
        return set(), weak_coins


def save_result_row(row: dict, first_write: bool = False):
    """코인 1개 결과를 CSV에 즉시 추가"""
    df_row = pd.DataFrame([row])
    if first_write and not os.path.exists(RESULT_CSV):
        df_row.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
    else:
        df_row.to_csv(RESULT_CSV, mode='a', header=not os.path.exists(RESULT_CSV),
                      index=False, encoding='utf-8-sig')


# ==========================================
# 메인
# ==========================================

def main():
    start_time = time.time()

    print("=" * 100)
    print("📊 288코인 롱 우선 vs 숏 우선 비교 최적화 (봉우리 검증 포함)")
    print("=" * 100)

    # 심볼 수집 (binance_bot-5.py 기준 288코인)
    all_symbols = load_288_symbols_from_bot()
    if not all_symbols:
        print("❌ 심볼 수집 실패!")
        return

    total_symbols = len(all_symbols)
    print(f"\n📌 대상: {total_symbols}개 심볼")
    if USE_REAL_FUNDING:
        print(f"📌 수수료: {FEE_RATE*100:.2f}% | 슬리피지: {SLIPPAGE_PCT*100:.2f}% | 펀딩: 바이낸스 실제 데이터 (폴백: {FUNDING_RATE_8H*100:.3f}%/8h)")
    else:
        print(f"📌 수수료: {FEE_RATE*100:.2f}% | 슬리피지: {SLIPPAGE_PCT*100:.2f}% | 펀딩: {FUNDING_RATE_8H*100:.3f}%/8h (고정)")
    print(f"📌 각 방향: Phase1 {PHASE1_TRIALS} + Phase2 {PHASE2_TRIALS} trials × 2단계")
    print(f"📌 코인당 총 trials: {(PHASE1_TRIALS + PHASE2_TRIALS) * 2 * 2} (롱우선 + 숏우선)")
    print(f"📌 최소 데이터 기간: {MIN_DATA_DAYS}일")
    print(f"📌 결과 파일: {RESULT_CSV}")

    # Resume 확인 (바늘/주의 코인은 자동 재최적화)
    completed, weak_coins = load_completed_symbols()
    if completed:
        print(f"\n🔄 이전 진행 발견: {len(completed)}개 완료, 나머지부터 재개")
    if weak_coins:
        print(f"⚠️  강건성 재최적화 대상: {len(weak_coins)}개 (주의/바늘 판정)")

    print("=" * 100)

    processed_count = len(completed)
    skipped_count = 0
    failed_count = 0
    coin_times = []

    for idx, sym_info in enumerate(all_symbols, 1):
        symbol = sym_info['symbol']
        vol_str = f"${sym_info['quote_volume_24h']/1e6:.1f}M" if sym_info['quote_volume_24h'] > 0 else "N/A"

        # 이미 완료된 심볼 스킵 (바늘/주의 코인은 이미 completed에서 제거됨)
        if symbol in completed:
            continue

        # 진행률
        pct = idx / total_symbols * 100
        eta_str = ""
        if coin_times:
            avg_time = np.mean(coin_times)
            remaining = total_symbols - idx
            eta_min = avg_time * remaining / 60
            if eta_min > 60:
                eta_str = f" ETA:{eta_min/60:.1f}h"
            else:
                eta_str = f" ETA:{eta_min:.0f}m"

        reopt_tag = " [재최적화:주의/바늘]" if symbol in weak_coins else ""
        print(f"\n{'━' * 100}")
        print(f"[{idx}/{total_symbols}] ({pct:.0f}%) {symbol} (Vol:{vol_str}){reopt_tag}{eta_str}")
        print(f"{'━' * 100}")

        coin_start = time.time()

        # 데이터 준비
        data = prepare_coin_data(symbol)
        if not data:
            print(f"  ❌ 데이터 부족 → 스킵")
            skipped_count += 1
            continue

        print(f"  📅 {data['start_date'].date()} ~ {data['end_date'].date()} ({data['days']}일)")

        try:
            # A) 롱 우선
            print(f"  🔵 롱 우선:")
            long_first = run_pipeline(symbol, data, 'long')

            # B) 숏 우선
            print(f"  🔴 숏 우선:")
            short_first = run_pipeline(symbol, data, 'short')

        except Exception as e:
            print(f"  ❌ 최적화 오류: {e}")
            failed_count += 1
            continue

        # 비교
        lf_cagr = long_first['combined_cagr']
        sf_cagr = short_first['combined_cagr']
        lf_sharpe = long_first['combined_sharpe']
        sf_sharpe = short_first['combined_sharpe']

        if lf_cagr > sf_cagr:
            winner = '롱 우선'
        elif sf_cagr > lf_cagr:
            winner = '숏 우선'
        else:
            winner = '롱 우선' if lf_sharpe >= sf_sharpe else '숏 우선'

        w_emoji = '🔵' if winner == '롱 우선' else '🔴'
        coin_elapsed = time.time() - coin_start
        coin_times.append(coin_elapsed)

        # 강건성 요약
        w_pri_v = (long_first if winner == '롱 우선' else short_first).get('pri_verdict', '?')
        w_sec_v = (long_first if winner == '롱 우선' else short_first).get('sec_verdict', '?')
        w_pri_r = (long_first if winner == '롱 우선' else short_first).get('pri_rank', 1)
        w_sec_r = (long_first if winner == '롱 우선' else short_first).get('sec_rank', 1)
        rob_tag = f"강건[P:{w_pri_v}#{w_pri_r} S:{w_sec_v}#{w_sec_r}]"

        print(f"  → {w_emoji} {winner} | "
              f"롱:{lf_cagr:.1f}%/{long_first['combined_mdd']:.1f}% "
              f"숏:{sf_cagr:.1f}%/{short_first['combined_mdd']:.1f}% "
              f"{rob_tag} ({coin_elapsed:.0f}s)")

        # 결과 저장
        row = {
            'Symbol': symbol,
            'QuoteVolume24h': sym_info['quote_volume_24h'],
            'Days': data['days'],
            'LF_Primary_CAGR': long_first['primary_only_cagr'],
            'LF_Combined_CAGR': lf_cagr,
            'LF_Combined_MDD': long_first['combined_mdd'],
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
            'SF_Primary_CAGR': short_first['primary_only_cagr'],
            'SF_Combined_CAGR': sf_cagr,
            'SF_Combined_MDD': short_first['combined_mdd'],
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
            'Winner': winner,
            'CAGR_Diff': sf_cagr - lf_cagr,
            # 강건성 검증 결과
            'LF_Pri_Verdict': long_first.get('pri_verdict', ''),
            'LF_Pri_Ratio': long_first.get('pri_ratio', ''),
            'LF_Pri_Rank': long_first.get('pri_rank', ''),
            'LF_Sec_Verdict': long_first.get('sec_verdict', ''),
            'LF_Sec_Ratio': long_first.get('sec_ratio', ''),
            'LF_Sec_Rank': long_first.get('sec_rank', ''),
            'SF_Pri_Verdict': short_first.get('pri_verdict', ''),
            'SF_Pri_Ratio': short_first.get('pri_ratio', ''),
            'SF_Pri_Rank': short_first.get('pri_rank', ''),
            'SF_Sec_Verdict': short_first.get('sec_verdict', ''),
            'SF_Sec_Ratio': short_first.get('sec_ratio', ''),
            'SF_Sec_Rank': short_first.get('sec_rank', ''),
        }

        save_result_row(row, first_write=(processed_count == 0))
        processed_count += 1

    # ==========================================
    # 최종 요약
    # ==========================================
    total_elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(total_elapsed)))

    print(f"\n\n{'=' * 100}")
    print(f"📊 최종 요약 (소요시간: {elapsed_str})")
    print(f"{'=' * 100}")
    print(f"  총 심볼: {total_symbols}개")
    print(f"  처리 완료: {processed_count}개")
    print(f"  데이터 부족 스킵: {skipped_count}개")
    print(f"  오류: {failed_count}개")

    # 전체 결과 로드 및 분석
    if not os.path.exists(RESULT_CSV):
        print("\n❌ 결과가 없습니다.")
        return

    df = pd.read_csv(RESULT_CSV)
    if df.empty:
        print("\n❌ 결과가 비어있습니다.")
        return

    # 타임스탬프 붙인 최종 CSV
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_csv = os.path.join(SAVE_DIR, f"all_coins_long_vs_short_FINAL_{ts}.csv")
    df.to_csv(final_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 최종 CSV: {final_csv}")

    # 요약 테이블
    print(f"\n{'=' * 130}")
    print(f"📊 코인별 비교 결과")
    print(f"{'=' * 130}")
    print(f"{'Symbol':<14} {'Vol(M)':>8} {'롱 CAGR':>9} {'롱 MDD':>8} {'롱 Sharpe':>10} "
          f"{'숏 CAGR':>9} {'숏 MDD':>8} {'숏 Sharpe':>10} {'승자':>8}")
    print(f"{'─' * 130}")

    for _, row in df.iterrows():
        w_mark = '🔵' if row['Winner'] == '롱 우선' else '🔴'
        vol_m = row.get('QuoteVolume24h', 0) / 1e6
        print(f"{row['Symbol']:<14} {vol_m:>7.0f}M "
              f"{row['LF_Combined_CAGR']:>8.1f}% {row['LF_Combined_MDD']:>7.1f}% {row['LF_Combined_Sharpe']:>10.2f} "
              f"{row['SF_Combined_CAGR']:>8.1f}% {row['SF_Combined_MDD']:>7.1f}% {row['SF_Combined_Sharpe']:>10.2f} "
              f"{w_mark} {row['Winner']}")

    # 승패 집계
    long_wins = len(df[df['Winner'] == '롱 우선'])
    short_wins = len(df[df['Winner'] == '숏 우선'])
    total = len(df)

    print(f"\n{'=' * 80}")
    print(f"📊 종합 승패 ({total}개 코인)")
    print(f"{'=' * 80}")
    print(f"  🔵 롱 우선 승: {long_wins}/{total}개 ({long_wins/total*100:.0f}%)")
    print(f"  🔴 숏 우선 승: {short_wins}/{total}개 ({short_wins/total*100:.0f}%)")

    # 유효한 결과만 필터 (CAGR > -500 등 비정상 제외)
    df_valid = df[(df['LF_Combined_CAGR'] > -500) & (df['SF_Combined_CAGR'] > -500)]
    if len(df_valid) > 0:
        print(f"\n📊 평균 지표 비교 (유효 {len(df_valid)}개 기준):")
        print(f"  {'항목':<20} {'롱 우선':>12} {'숏 우선':>12}")
        print(f"  {'─' * 44}")
        print(f"  {'평균 결합 CAGR':<20} {df_valid['LF_Combined_CAGR'].mean():>11.1f}% {df_valid['SF_Combined_CAGR'].mean():>11.1f}%")
        print(f"  {'중앙값 CAGR':<20} {df_valid['LF_Combined_CAGR'].median():>11.1f}% {df_valid['SF_Combined_CAGR'].median():>11.1f}%")
        print(f"  {'평균 결합 MDD':<20} {df_valid['LF_Combined_MDD'].mean():>11.1f}% {df_valid['SF_Combined_MDD'].mean():>11.1f}%")
        print(f"  {'평균 Sharpe':<20} {df_valid['LF_Combined_Sharpe'].mean():>12.2f} {df_valid['SF_Combined_Sharpe'].mean():>12.2f}")
        print(f"  {'평균 거래수':<20} {df_valid['LF_Trades'].mean():>12.0f} {df_valid['SF_Trades'].mean():>12.0f}")

    # 강건성 통계
    if 'LF_Pri_Verdict' in df.columns and df['LF_Pri_Verdict'].notna().any():
        print(f"\n{'=' * 80}")
        print(f"🛡️ 강건성 검증 요약 (승자 기준)")
        print(f"{'=' * 80}")
        verdicts = []
        for _, r in df.iterrows():
            if r['Winner'] == '롱 우선':
                verdicts.append({'pri': r.get('LF_Pri_Verdict', ''), 'sec': r.get('LF_Sec_Verdict', ''),
                                  'pri_rank': r.get('LF_Pri_Rank', 1), 'sec_rank': r.get('LF_Sec_Rank', 1)})
            else:
                verdicts.append({'pri': r.get('SF_Pri_Verdict', ''), 'sec': r.get('SF_Sec_Verdict', ''),
                                  'pri_rank': r.get('SF_Pri_Rank', 1), 'sec_rank': r.get('SF_Sec_Rank', 1)})

        for label, key in [('Primary', 'pri'), ('Secondary', 'sec')]:
            counts = {}
            for v in verdicts:
                vd = v[key]
                if vd:
                    counts[vd] = counts.get(vd, 0) + 1
            rank_gt1 = sum(1 for v in verdicts if v.get(f'{key}_rank', 1) > 1)
            total_v = sum(counts.values())
            if total_v > 0:
                parts = [f"{k}:{n}({n/total_v*100:.0f}%)" for k, n in sorted(counts.items())]
                print(f"  {label:>10}: {' | '.join(parts)}")
                if rank_gt1 > 0:
                    print(f"             → {rank_gt1}개 코인에서 하위 후보로 대체됨")

    # CAGR > 0 인 코인 분류
    df_positive = df_valid[(df_valid['LF_Combined_CAGR'] > 0) | (df_valid['SF_Combined_CAGR'] > 0)]
    if len(df_positive) > 0:
        lf_pos = len(df_valid[df_valid['LF_Combined_CAGR'] > 0])
        sf_pos = len(df_valid[df_valid['SF_Combined_CAGR'] > 0])
        print(f"\n📊 양수 CAGR 코인 수:")
        print(f"  🔵 롱 우선 CAGR>0: {lf_pos}/{len(df_valid)}개")
        print(f"  🔴 숏 우선 CAGR>0: {sf_pos}/{len(df_valid)}개")

    # 최적 코인 Top 10
    print(f"\n{'=' * 80}")
    print("🏆 롱 우선 CAGR Top 10:")
    print(f"{'=' * 80}")
    top_lf = df_valid.nlargest(10, 'LF_Combined_CAGR')
    for rank, (_, row) in enumerate(top_lf.iterrows(), 1):
        print(f"  {rank:>2}. {row['Symbol']:<14} CAGR:{row['LF_Combined_CAGR']:>7.1f}% "
              f"MDD:{row['LF_Combined_MDD']:>7.1f}% Sharpe:{row['LF_Combined_Sharpe']:>6.2f} "
              f"Lev:{row['LF_Long_Lev']}x/{row['LF_Short_Lev']}x")

    print(f"\n🏆 숏 우선 CAGR Top 10:")
    top_sf = df_valid.nlargest(10, 'SF_Combined_CAGR')
    for rank, (_, row) in enumerate(top_sf.iterrows(), 1):
        print(f"  {rank:>2}. {row['Symbol']:<14} CAGR:{row['SF_Combined_CAGR']:>7.1f}% "
              f"MDD:{row['SF_Combined_MDD']:>7.1f}% Sharpe:{row['SF_Combined_Sharpe']:>6.2f} "
              f"Lev:{row['SF_Short_Lev']}x/{row['SF_Long_Lev']}x")

    # 최종 BEST CONFIGS
    print(f"\n{'=' * 80}")
    print("📋 코인별 최적 전략 CONFIGS (각 코인 승자 기준, CAGR>0만)")
    print(f"{'=' * 80}")
    print("BEST_TRADING_CONFIGS = [")

    df_best = df_valid.copy()
    df_best['Best_CAGR'] = df_best.apply(
        lambda r: r['LF_Combined_CAGR'] if r['Winner'] == '롱 우선' else r['SF_Combined_CAGR'], axis=1)
    df_best = df_best[df_best['Best_CAGR'] > 0].sort_values('Best_CAGR', ascending=False)

    for _, row in df_best.iterrows():
        if row['Winner'] == '롱 우선':
            print(f"    # {row['Symbol']} → 롱 우선 (CAGR:{row['LF_Combined_CAGR']:.1f}% "
                  f"MDD:{row['LF_Combined_MDD']:.1f}% Sharpe:{row['LF_Combined_Sharpe']:.2f})")
            print(f"    {{'symbol': '{row['Symbol']}', 'priority': 'long', "
                  f"'long_ma': {int(row['LF_Long_MA'])}, 'long_sk': {int(row['LF_Long_SK'])}, "
                  f"'long_sks': {int(row['LF_Long_SKs'])}, 'long_sd': {int(row['LF_Long_SD'])}, "
                  f"'long_lev': {int(row['LF_Long_Lev'])}, "
                  f"'short_ma': {int(row['LF_Short_MA'])}, 'short_sk': {int(row['LF_Short_SK'])}, "
                  f"'short_sks': {int(row['LF_Short_SKs'])}, 'short_sd': {int(row['LF_Short_SD'])}, "
                  f"'short_lev': {int(row['LF_Short_Lev'])}}},")
        else:
            print(f"    # {row['Symbol']} → 숏 우선 (CAGR:{row['SF_Combined_CAGR']:.1f}% "
                  f"MDD:{row['SF_Combined_MDD']:.1f}% Sharpe:{row['SF_Combined_Sharpe']:.2f})")
            print(f"    {{'symbol': '{row['Symbol']}', 'priority': 'short', "
                  f"'short_ma': {int(row['SF_Short_MA'])}, 'short_sk': {int(row['SF_Short_SK'])}, "
                  f"'short_sks': {int(row['SF_Short_SKs'])}, 'short_sd': {int(row['SF_Short_SD'])}, "
                  f"'short_lev': {int(row['SF_Short_Lev'])}, "
                  f"'long_ma': {int(row['SF_Long_MA'])}, 'long_sk': {int(row['SF_Long_SK'])}, "
                  f"'long_sks': {int(row['SF_Long_SKs'])}, 'long_sd': {int(row['SF_Long_SD'])}, "
                  f"'long_lev': {int(row['SF_Long_Lev'])}}},")
    print("]")

    overall = '롱 우선' if long_wins > short_wins else ('숏 우선' if short_wins > long_wins else '동률')
    print(f"\n✅ 전체 최적화 완료! 총 {total}개 코인 분석, 종합 우세: {overall}")
    print(f"⏱️ 총 소요시간: {elapsed_str}")


if __name__ == "__main__":
    main()
