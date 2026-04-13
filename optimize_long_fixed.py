"""
================================================================================
Binance Futures 롱 포지션 2단계 베이지안 최적화 (수정 버전)
================================================================================
전략 로직:
  - 숏 시그널 확인 (CSV에서 로드한 숏 파라미터):
    시가 < MA_short(4H) AND K < D (1D, 전일) → 롱 진입 금지 구간
  - 숏 시그널 없을 때:
    시가 > MA_long(4H) AND K > D (1D, 전일, 새 롱 파라미터) → 롱 진입
  - 롱 청산: 숏 시그널 발생 OR 롱 조건 미충족

수정 사항 (원본 대비):
  - 수수료: 0.04% (0.06%에서 수정)
  - 슬리피지: 0.05%
  - 펀딩비: 0.01% / 8h (노셔널 = 마진 × 레버리지 기준)
  - 포지션 내 비복리, 거래 간 복리
  - 강제청산: entry_price * (1 - 1/leverage), low 가격 체크
  - 청산 수수료: 노셔널 기준
  - 숏 파라미터: CSV에서 자동 로드
  - 제외 코인: CSV의 Days < 365 자동 제외
  - 최적화: Optuna TPE 2단계
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

# 1단계 스크리닝 설정
PHASE1_TRIALS = 50
PHASE1_CAGR_THRESHOLD = 0

# 2단계 파인튜닝 설정
PHASE2_TRIALS = 300

# 수수료/비용 설정
FEE_RATE = 0.0004          # 0.04%
SLIPPAGE_PCT = 0.0005      # 0.05%
FUNDING_RATE_8H = 0.0001   # 0.01% per 8h
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5  # 4H 봉당 펀딩비 = 0.00005

MIN_DATA_DAYS = 365

# 파라미터 탐색 범위
PARAM_RANGES_PHASE1 = {
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

# 숏 최적화 결과 CSV 경로
SHORT_CSV_PATH = os.path.join(SAVE_DIR, "binance_short_fixed_final_20260208_145556.csv")

# 스테이블코인/인덱스 제외
ALWAYS_EXCLUDE = {'USDCUSDT', 'BTCDOMUSDT'}


# ==========================================
# CSV에서 숏 파라미터 로드 + 제외 코인 판단
# ==========================================

def load_short_configs_and_exclusions(csv_path: str) -> tuple:
    """
    숏 최적화 CSV에서:
    1. 유효한 숏 파라미터 로드 (숏 필터용)
    2. 제외할 코인 목록 생성
    3. 숏 CAGR < 0% 코인 (숏 필터 없이 롱만)

    Returns:
        short_configs: dict {symbol: {ma_period, stoch_k_period, ...}}
        excluded_coins: set of symbols to exclude entirely
        no_short_filter_coins: set of symbols where short filter is disabled
    """
    if not os.path.exists(csv_path):
        print(f"  ❌ 숏 CSV 파일 없음: {csv_path}")
        print(f"  → 숏 필터 없이 롱만 최적화합니다.")
        return {}, set(), set()

    df = pd.read_csv(csv_path)
    print(f"  📂 숏 CSV 로드: {len(df)}개 코인")

    excluded = set(ALWAYS_EXCLUDE)

    # 1. Days < 365 (데이터 부족 → 과적합 위험)
    short_data = df[df['Days'] < MIN_DATA_DAYS]
    excluded.update(short_data['Symbol'].tolist())
    print(f"    - Days < {MIN_DATA_DAYS}: {len(short_data)}개 제외")

    # 2. 숏 CAGR < 0% → 숏 필터 없이 롱만 (제외하지 않음)
    neg_cagr = df[(df['CAGR'] < 0) & (~df['Symbol'].isin(excluded))]
    no_short_filter_coins = set(neg_cagr['Symbol'].tolist())
    print(f"    - 숏 CAGR < 0% (숏필터 비활성): {len(no_short_filter_coins)}개")

    # 3. 스테이블코인/인덱스
    print(f"    - 상시 제외: {ALWAYS_EXCLUDE}")

    # 유효한 숏 파라미터 (제외X, CAGR >= 0)
    valid = df[(~df['Symbol'].isin(excluded)) & (df['CAGR'] >= 0)]
    short_configs = {}
    for _, row in valid.iterrows():
        short_configs[row['Symbol']] = {
            'ma_period': int(row['MA']),
            'stoch_k_period': int(row['Stoch_K']),
            'stoch_k_smooth': int(row['Stoch_Smooth']),
            'stoch_d_period': int(row['Stoch_D']),
            'leverage': int(row['Leverage']),
        }

    print(f"    - 유효 숏 파라미터: {len(short_configs)}개")
    print(f"    - 총 제외: {len(excluded)}개")

    return short_configs, excluded, no_short_filter_coins


# ==========================================
# 바이낸스 Futures 전체 심볼 가져오기
# ==========================================

def get_all_futures_symbols() -> list:
    """바이낸스 Futures 전체 USDT 영구 계약 심볼 목록"""
    print("\n📋 바이낸스 Futures 전체 심볼 가져오는 중...")

    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    try:
        response = requests.get(url, timeout=30)
        data = response.json()

        symbols = []
        for info in data['symbols']:
            if (info['contractType'] == 'PERPETUAL' and
                info['quoteAsset'] == 'USDT' and
                info['status'] == 'TRADING'):
                symbols.append(info['symbol'])

        priority = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT']

        sorted_symbols = []
        for p in priority:
            if p in symbols:
                sorted_symbols.append(p)
                symbols.remove(p)
        sorted_symbols.extend(sorted(symbols))

        print(f"  ✅ 총 {len(sorted_symbols)}개 심볼 발견")
        return sorted_symbols

    except Exception as e:
        print(f"  ❌ 심볼 목록 가져오기 실패: {e}")
        return []


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


# ==========================================
# 시그널 전처리: 4H MA + 1D 스토캐스틱 → 4H 프레임 매핑
# ==========================================

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
# 롱 백테스트 함수 (수정 버전)
# ==========================================

def backtest_long(data: dict, short_cfg: dict,
                  long_ma: int, long_sk: int, long_sks: int,
                  long_sd: int, long_lev: int) -> dict:
    """
    롱 포지션 백테스트 (포지션 내 비복리, 거래 간 복리)

    전략:
      숏 필터: open < MA_short AND K_short < D_short (전일) → 롱 금지
      롱 진입: NOT 숏필터 AND open > MA_long AND K_long > D_long (전일)
      롱 청산: 조건 미충족

    short_cfg가 None이면 숏 필터 없이 롱만 최적화
    """
    df_4h = data['df_4h']
    df_daily = data['df_daily']

    # ── 롱 시그널 준비 ──
    max_stoch_long = long_sk + long_sks + long_sd + 10
    if len(df_4h) < long_ma + 50 or len(df_daily) < max_stoch_long:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    df_long = prepare_signals(df_4h, df_daily, long_ma, long_sk, long_sks, long_sd)

    # ── 숏 시그널 준비 (필터용) ──
    has_short_filter = short_cfg is not None
    if has_short_filter:
        s_ma = short_cfg['ma_period']
        s_sk = short_cfg['stoch_k_period']
        s_sks = short_cfg['stoch_k_smooth']
        s_sd = short_cfg['stoch_d_period']

        max_stoch_short = s_sk + s_sks + s_sd + 10
        if len(df_4h) < s_ma + 50 or len(df_daily) < max_stoch_short:
            has_short_filter = False
        else:
            df_short = prepare_signals(df_4h, df_daily, s_ma, s_sk, s_sks, s_sd)

    # ── 합치기 ──
    df_bt = df_long[['timestamp', 'open', 'high', 'low', 'close', 'ma',
                      'prev_slow_k', 'prev_slow_d']].copy()
    df_bt.rename(columns={
        'ma': 'ma_long',
        'prev_slow_k': 'long_k',
        'prev_slow_d': 'long_d',
    }, inplace=True)

    if has_short_filter:
        df_bt['ma_short'] = df_short['ma'].values
        df_bt['short_k'] = df_short['prev_slow_k'].values
        df_bt['short_d'] = df_short['prev_slow_d'].values
        df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d',
                                      'ma_short', 'short_k', 'short_d']).reset_index(drop=True)
    else:
        df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d']).reset_index(drop=True)

    if len(df_bt) < 100:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    # ── 시뮬레이션 ──
    initial_capital = 10000
    equity = initial_capital
    in_long = False
    entry_price = 0.0
    entry_margin = 0.0
    cum_funding = 0.0
    trade_count = 0
    equity_curve = [equity]
    leverage = long_lev

    short_filter_count = 0

    for i in range(1, len(df_bt)):
        prev = df_bt.iloc[i - 1]
        curr = df_bt.iloc[i]

        opening_price = curr['open']

        # 숏 필터: open < MA_short AND K_short < D_short
        if has_short_filter:
            short_filter = (opening_price < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
        else:
            short_filter = False

        if short_filter:
            short_filter_count += 1

        # 롱 시그널: open > MA_long AND K_long > D_long
        long_signal = (opening_price > prev['ma_long']) and (prev['long_k'] > prev['long_d'])

        # 최종 롱 조건: 숏 필터 없음 + 롱 시그널
        long_condition = (not short_filter) and long_signal

        # ── 롱 진입 ──
        if long_condition and not in_long:
            fee = equity * FEE_RATE * leverage
            slippage = equity * SLIPPAGE_PCT * leverage
            equity -= (fee + slippage)

            entry_margin = equity
            entry_price = opening_price
            cum_funding = 0.0
            in_long = True
            trade_count += 1

        # ── 롱 청산 ──
        elif not long_condition and in_long:
            if entry_price > 0:
                price_return = opening_price / entry_price - 1
                trade_pnl = entry_margin * price_return * leverage
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

            in_long = False
            entry_price = 0.0
            entry_margin = 0.0
            cum_funding = 0.0
            trade_count += 1

        # ── 포지션 보유 중 ──
        if in_long and entry_price > 0:
            price_return = curr['close'] / entry_price - 1
            unrealized_pnl = entry_margin * price_return * leverage
            cum_funding += entry_margin * leverage * FUNDING_PER_4H  # ★ 노셔널 기준
            display_equity = entry_margin + unrealized_pnl - cum_funding

            # 강제청산: 롱은 가격 하락 시
            if leverage > 0:
                liquidation_price = entry_price * (1 - 1 / leverage)
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

    # ── 마지막 포지션 정리 ──
    if in_long and entry_price > 0:
        last_price = df_bt.iloc[-1]['close']
        price_return = last_price / entry_price - 1
        trade_pnl = entry_margin * price_return * leverage
        equity = entry_margin + trade_pnl - cum_funding

        exit_notional = abs(entry_margin * leverage * (last_price / entry_price))
        fee = exit_notional * FEE_RATE
        slippage_cost = exit_notional * SLIPPAGE_PCT
        equity -= (fee + slippage_cost)

    # ── 성과 계산 ──
    equity_curve = np.array(equity_curve)

    if len(equity_curve) < 2:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    if equity_curve[-1] <= 0:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': trade_count}

    total_days = (df_bt.iloc[-1]['timestamp'] - df_bt.iloc[0]['timestamp']).days
    if total_days < 30:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    equity_curve = np.maximum(equity_curve, 0.01)

    total_return = equity_curve[-1] / equity_curve[0]
    if total_return <= 0:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    years = total_days / 365.25
    cagr = (total_return ** (1 / years) - 1) * 100

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    mdd = drawdown.min()

    # Sharpe (일 수익률 기반)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) > 10:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365 * 6) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    # 숏 필터 비율
    short_filter_ratio = short_filter_count / max(len(df_bt) - 1, 1) * 100

    return {
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'trades': trade_count,
        'days': total_days,
        'short_filter_ratio': short_filter_ratio,
    }


# ==========================================
# 2단계 최적화
# ==========================================

def optimize_long_phase1(symbol: str, data: dict, short_cfg: dict) -> dict:
    """1단계: 빠른 스크리닝 (50 trials)"""

    def objective(trial):
        ma = trial.suggest_int('long_ma', *PARAM_RANGES_PHASE1['ma_range'])
        sk = trial.suggest_int('long_sk', *PARAM_RANGES_PHASE1['stoch_k_range'])
        sks = trial.suggest_int('long_sks', *PARAM_RANGES_PHASE1['stoch_smooth_range'])
        sd = trial.suggest_int('long_sd', *PARAM_RANGES_PHASE1['stoch_d_range'])
        lev = trial.suggest_int('long_lev', *PARAM_RANGES_PHASE1['leverage_range'])

        result = backtest_long(data, short_cfg, ma, sk, sks, sd, lev)

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
    best_result = backtest_long(data, short_cfg,
                                best['long_ma'], best['long_sk'],
                                best['long_sks'], best['long_sd'],
                                best['long_lev'])

    return {
        'Symbol': symbol,
        'CAGR': best_result['cagr'],
        'MDD': best_result['mdd'],
        'Sharpe': best_result['sharpe'],
        'Trades': best_result['trades'],
        'Days': data['days'],
        'Short_Filter_Ratio': best_result.get('short_filter_ratio', 0),
        'Long_MA': best['long_ma'],
        'Long_SK': best['long_sk'],
        'Long_SKs': best['long_sks'],
        'Long_SD': best['long_sd'],
        'Long_Lev': best['long_lev'],
        'Phase': 1,
    }


def optimize_long_phase2(symbol: str, data: dict, short_cfg: dict, phase1_result: dict) -> dict:
    """2단계: 정밀 파인튜닝 (300 trials)"""

    def get_range(center, original_range, ratio=0.3):
        min_val, max_val = original_range
        delta = int((max_val - min_val) * ratio)
        new_min = max(min_val, center - delta)
        new_max = min(max_val, center + delta)
        return (new_min, new_max)

    param_ranges_phase2 = {
        'ma_range': get_range(phase1_result['Long_MA'], PARAM_RANGES_PHASE1['ma_range']),
        'stoch_k_range': get_range(phase1_result['Long_SK'], PARAM_RANGES_PHASE1['stoch_k_range']),
        'stoch_smooth_range': get_range(phase1_result['Long_SKs'], PARAM_RANGES_PHASE1['stoch_smooth_range']),
        'stoch_d_range': get_range(phase1_result['Long_SD'], PARAM_RANGES_PHASE1['stoch_d_range']),
        'leverage_range': PARAM_RANGES_PHASE1['leverage_range'],
    }

    print(f"    📊 2단계 탐색 범위:")
    print(f"       MA: {param_ranges_phase2['ma_range']} (1단계: {phase1_result['Long_MA']})")
    print(f"       SK: {param_ranges_phase2['stoch_k_range']} (1단계: {phase1_result['Long_SK']})")
    print(f"       SKs: {param_ranges_phase2['stoch_smooth_range']} (1단계: {phase1_result['Long_SKs']})")
    print(f"       SD: {param_ranges_phase2['stoch_d_range']} (1단계: {phase1_result['Long_SD']})")

    def objective(trial):
        ma = trial.suggest_int('long_ma', *param_ranges_phase2['ma_range'])
        sk = trial.suggest_int('long_sk', *param_ranges_phase2['stoch_k_range'])
        sks = trial.suggest_int('long_sks', *param_ranges_phase2['stoch_smooth_range'])
        sd = trial.suggest_int('long_sd', *param_ranges_phase2['stoch_d_range'])
        lev = trial.suggest_int('long_lev', *param_ranges_phase2['leverage_range'])

        result = backtest_long(data, short_cfg, ma, sk, sks, sd, lev)

        cagr = result['cagr']
        mdd = result['mdd']
        sharpe = result['sharpe']

        if cagr < -50 or mdd < -90:
            return -9999

        score = cagr + (mdd * 0.7) + (sharpe * 20)
        return score

    sampler = TPESampler(seed=42, n_startup_trials=min(30, PHASE2_TRIALS // 3))
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Phase1 최적값을 시드로 삽입
    study.enqueue_trial({
        'long_ma': phase1_result['Long_MA'],
        'long_sk': phase1_result['Long_SK'],
        'long_sks': phase1_result['Long_SKs'],
        'long_sd': phase1_result['Long_SD'],
        'long_lev': phase1_result['Long_Lev'],
    })

    study.optimize(objective, n_trials=PHASE2_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_long(data, short_cfg,
                                best['long_ma'], best['long_sk'],
                                best['long_sks'], best['long_sd'],
                                best['long_lev'])

    return {
        'Symbol': symbol,
        'CAGR': best_result['cagr'],
        'MDD': best_result['mdd'],
        'Sharpe': best_result['sharpe'],
        'Trades': best_result['trades'],
        'Days': data['days'],
        'Short_Filter_Ratio': best_result.get('short_filter_ratio', 0),
        'Long_MA': best['long_ma'],
        'Long_SK': best['long_sk'],
        'Long_SKs': best['long_sks'],
        'Long_SD': best['long_sd'],
        'Long_Lev': best['long_lev'],
        'Phase': 2,
        'Phase1_CAGR': phase1_result['CAGR'],
        'Improvement': best_result['cagr'] - phase1_result['CAGR'],
    }


# ==========================================
# 메인 실행
# ==========================================

def main():
    print("=" * 80)
    print("📈 Binance Futures 롱 포지션 2단계 베이지안 최적화 (수정 버전)")
    print("=" * 80)
    print(f"📌 수수료: {FEE_RATE*100:.2f}%")
    print(f"📌 슬리피지: {SLIPPAGE_PCT*100:.2f}%")
    print(f"📌 펀딩비: {FUNDING_RATE_8H*100:.3f}% / 8h (노셔널 기준)")
    print(f"📌 방식: 포지션 내 비복리, 거래 간 복리")
    print(f"📌 1단계: 빠른 스크리닝 ({PHASE1_TRIALS} trials)")
    print(f"📌 2단계: CAGR >= {PHASE1_CAGR_THRESHOLD}% 코인 정밀 파인튜닝 ({PHASE2_TRIALS} trials)")
    print(f"📌 전략: NOT 숏필터 AND 가격 > MA AND K > D → 롱 진입")
    print(f"📌 레버리지 범위: {PARAM_RANGES_PHASE1['leverage_range'][0]}~{PARAM_RANGES_PHASE1['leverage_range'][1]}배")
    print("=" * 80)

    # ── CSV에서 숏 파라미터 로드 ──
    print("\n📂 숏 파라미터 로드 중...")
    short_configs, excluded_coins, no_short_filter_coins = load_short_configs_and_exclusions(SHORT_CSV_PATH)

    # ── 전체 심볼 가져오기 ──
    all_symbols = get_all_futures_symbols()
    if not all_symbols:
        print("❌ 심볼 목록을 가져올 수 없습니다.")
        return

    # ── 제외 코인 필터링 ──
    active_symbols = [s for s in all_symbols if s not in excluded_coins]
    print(f"\n📊 전체 심볼: {len(all_symbols)}개")
    print(f"📊 제외: {len(excluded_coins)}개")
    print(f"📊 대상: {len(active_symbols)}개")
    print(f"📊 숏 필터 비활성 (CAGR<0%): {len(no_short_filter_coins)}개")

    # ── 1단계: 빠른 스크리닝 ──
    print("\n" + "=" * 80)
    print("📊 1단계: 빠른 스크리닝 시작")
    print("=" * 80)

    phase1_results = []
    valid_data = {}

    for idx, symbol in enumerate(active_symbols, 1):
        print(f"\n[{idx}/{len(active_symbols)}] {symbol}", end=" ")

        # 숏 필터 결정
        if symbol in no_short_filter_coins:
            short_cfg = None
            print("(숏필터 OFF)", end=" ")
        elif symbol in short_configs:
            short_cfg = short_configs[symbol]
            print(f"(숏필터: MA{short_cfg['ma_period']})", end=" ")
        else:
            short_cfg = None
            print("(숏 파라미터 없음)", end=" ")

        data = prepare_coin_data(symbol, silent=True)
        if not data:
            print("⚠️ 데이터 부족")
            continue

        valid_data[symbol] = data
        print(f"[{data['start_date'].date()}~{data['end_date'].date()}, {data['days']}일]", end=" ")

        print("1단계...", end=" ", flush=True)
        result = optimize_long_phase1(symbol, data, short_cfg)
        result['has_short_filter'] = short_cfg is not None
        phase1_results.append(result)

        print(f"✅ Lev:{result['Long_Lev']}x "
              f"MA:{result['Long_MA']} SK:{result['Long_SK']}/{result['Long_SKs']}/{result['Long_SD']} "
              f"CAGR:{result['CAGR']:.1f}% MDD:{result['MDD']:.1f}% Sharpe:{result['Sharpe']:.2f} "
              f"거래:{result['Trades']}회 숏필터:{result['Short_Filter_Ratio']:.0f}%")

    # 1단계 결과 저장
    if phase1_results:
        df_phase1 = pd.DataFrame(phase1_results)
        df_phase1 = df_phase1.sort_values('CAGR', ascending=False)
        phase1_path = os.path.join(SAVE_DIR,
                                    f"binance_long_fixed_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_phase1.to_csv(phase1_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 1단계 결과: {phase1_path}")

    # ── 2단계 대상 선별 ──
    phase2_candidates = [r for r in phase1_results if r['CAGR'] >= PHASE1_CAGR_THRESHOLD]
    print(f"\n2단계 대상: {len(phase2_candidates)}개 (CAGR >= {PHASE1_CAGR_THRESHOLD}%)")

    if not phase2_candidates:
        print("❌ 2단계 대상이 없습니다.")
        return

    # ── 2단계: 정밀 파인튜닝 ──
    print("\n" + "=" * 80)
    print("📊 2단계: 정밀 파인튜닝 시작")
    print("=" * 80)

    phase2_results = []
    for idx, p1 in enumerate(phase2_candidates, 1):
        symbol = p1['Symbol']
        print(f"\n[{idx}/{len(phase2_candidates)}] {symbol} (1단계 CAGR: {p1['CAGR']:.1f}%)")

        if symbol not in valid_data:
            print("  ⚠️ 데이터 없음, 스킵")
            continue

        data = valid_data[symbol]

        if symbol in no_short_filter_coins:
            short_cfg = None
        elif symbol in short_configs:
            short_cfg = short_configs[symbol]
        else:
            short_cfg = None

        result = optimize_long_phase2(symbol, data, short_cfg, p1)
        result['has_short_filter'] = short_cfg is not None
        phase2_results.append(result)

        print(f"    ✅ Lev:{result['Long_Lev']}x "
              f"MA:{result['Long_MA']} SK:{result['Long_SK']}/{result['Long_SKs']}/{result['Long_SD']} "
              f"CAGR:{result['CAGR']:.1f}% MDD:{result['MDD']:.1f}% Sharpe:{result['Sharpe']:.2f} "
              f"거래:{result['Trades']}회 개선:{result['Improvement']:+.1f}%")

    # ── 결과 정리 ──
    if not phase2_results:
        print("❌ 2단계 결과가 없습니다.")
        return

    # Phase1만 있는 결과 + Phase2 결과 합치기
    phase2_symbols = {r['Symbol'] for r in phase2_results}
    final_results = phase2_results.copy()
    for r in phase1_results:
        if r['Symbol'] not in phase2_symbols:
            final_results.append(r)

    df_final = pd.DataFrame(final_results)
    df_final = df_final.sort_values('CAGR', ascending=False)

    # CSV 저장
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = os.path.join(SAVE_DIR, f"binance_long_fixed_final_{ts}.csv")
    df_final.to_csv(final_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 최종 결과: {final_path}")

    # ── 적합 코인 필터링 ──
    df_suitable = df_final[
        (df_final['CAGR'] > 0) &
        (df_final['MDD'] > -85) &
        (df_final['Sharpe'] > 0.3) &
        (df_final['Trades'] > 3)
    ].copy()

    df_suitable = df_suitable.sort_values('Sharpe', ascending=False)

    print(f"\n{'=' * 80}")
    print(f"📊 결과 요약")
    print(f"{'=' * 80}")
    print(f"전체: {len(df_final)}개  |  적합: {len(df_suitable)}개  |  제외: {len(df_final) - len(df_suitable)}개")

    if len(df_suitable) > 0:
        print(f"\n평균 CAGR: {df_suitable['CAGR'].mean():.1f}%  |  "
              f"중간값: {df_suitable['CAGR'].median():.1f}%")
        print(f"평균 MDD: {df_suitable['MDD'].mean():.1f}%  |  "
              f"평균 Sharpe: {df_suitable['Sharpe'].mean():.2f}")
        print(f"평균 거래수: {df_suitable['Trades'].mean():.1f}회  |  "
              f"평균 숏필터: {df_suitable['Short_Filter_Ratio'].mean():.1f}%")

    # ── 테이블 출력 ──
    print(f"\n{'─' * 130}")
    print(f"{'Symbol':<18} {'Lev':>4} {'L_MA':>5} {'L_SK':>5} {'L_SKs':>5} {'L_SD':>5} "
          f"{'CAGR%':>8} {'MDD%':>7} {'Sharpe':>7} {'Trades':>7} "
          f"{'S_Filt%':>7} {'Days':>5} {'Filter':>6}")
    print(f"{'─' * 130}")
    for _, row in df_suitable.head(80).iterrows():
        filt = "ON" if row.get('has_short_filter', True) else "OFF"
        print(f"{row['Symbol']:<18} {row['Long_Lev']:>3}x "
              f"{row['Long_MA']:>5} {row['Long_SK']:>5} {row['Long_SKs']:>5} {row['Long_SD']:>5} "
              f"{row['CAGR']:>7.1f}% {row['MDD']:>6.1f}% {row['Sharpe']:>7.2f} "
              f"{row['Trades']:>7} "
              f"{row.get('Short_Filter_Ratio', 0):>6.1f}% {row['Days']:>5} {filt:>6}")

    # ── LONG_TRADING_CONFIGS 출력 ──
    print(f"\n{'=' * 80}")
    print("📋 LONG_TRADING_CONFIGS (봇 적용용)")
    print(f"{'=' * 80}")
    print("LONG_TRADING_CONFIGS = [")
    for _, row in df_suitable.iterrows():
        print(f"    {{'symbol': '{row['Symbol']}', "
              f"'ma_period': {row['Long_MA']}, 'stoch_k_period': {row['Long_SK']}, "
              f"'stoch_k_smooth': {row['Long_SKs']}, 'stoch_d_period': {row['Long_SD']}, "
              f"'leverage': {row['Long_Lev']}}},")
    print("]")

    # ── 레버리지 분포 ──
    if len(df_suitable) > 0:
        print(f"\n📊 레버리지 분포:")
        for lev in range(1, 6):
            count = len(df_suitable[df_suitable['Long_Lev'] == lev])
            if count > 0:
                print(f"  {lev}x: {count}개")

    print(f"\n✅ 롱 전용 최적화 완료!")


if __name__ == "__main__":
    main()
