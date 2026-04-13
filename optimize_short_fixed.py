"""
================================================================================
Binance Futures 숏 포지션 2단계 베이지안 최적화 (수정 버전)
================================================================================
수정 사항:
- 펀딩비 추가 (0.01% / 8시간)
- 슬리피지 추가 (0.05%)
- 수수료 0.04%로 통일
- 포지션 내 비복리, 거래 간 복리 방식 (backtest.py와 동일)
- 강제청산 시 자본 완전 손실
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# 수수료/비용 설정 (backtest.py와 동일)
FEE_RATE = 0.0004          # 0.04%
SLIPPAGE_PCT = 0.0005      # 0.05%
FUNDING_RATE_8H = 0.0001   # 0.01% per 8h
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5  # 4H 봉당 펀딩비

MIN_DATA_DAYS = 365

# 파라미터 탐색 범위
PARAM_RANGES_PHASE1 = {
    'ma_range': (30, 350),
    'stoch_k_range': (20, 150),
    'stoch_smooth_range': (5, 80),
    'stoch_d_range': (3, 50),
    'leverage_range': (1, 5),
}

# 저장 경로
SAVE_DIR = os.path.expanduser("~/Downloads")
CACHE_DIR = os.path.expanduser("~/binance_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ==========================================
# 바이낸스 Futures 전체 심볼 가져오기
# ==========================================

def get_all_futures_symbols() -> list:
    """바이낸스 Futures 전체 USDT 영구 계약 심볼 목록 가져오기"""
    print("\n📋 바이낸스 Futures 전체 심볼 가져오는 중...")

    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    try:
        response = requests.get(url, timeout=30)
        data = response.json()

        symbols = []
        for symbol_info in data['symbols']:
            if (symbol_info['contractType'] == 'PERPETUAL' and
                symbol_info['quoteAsset'] == 'USDT' and
                symbol_info['status'] == 'TRADING'):
                symbols.append(symbol_info['symbol'])

        priority = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT']

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
    """바이낸스 선물 캔들 데이터 전체 다운로드"""
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

    for i in range(100):
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
    """스토캐스틱 계산"""
    lowest = df['low'].rolling(window=k_period).min()
    highest = df['high'].rolling(window=k_period).max()

    fast_k = ((df['close'] - lowest) / (highest - lowest)) * 100
    fast_k = fast_k.replace([np.inf, -np.inf], np.nan)

    slow_k = fast_k.rolling(window=k_smooth).mean()
    slow_d = slow_k.rolling(window=d_period).mean()

    return slow_k, slow_d


# ==========================================
# 백테스트 함수 (숏 포지션) - 수정 버전
# ==========================================

def backtest_short(data: dict, ma_period: int, stoch_k: int, stoch_smooth: int,
                   stoch_d: int, leverage: int) -> dict:
    """숏 포지션 백테스트 (포지션 내 비복리, 거래 간 복리, 펀딩비+슬리피지 포함)

    진입 조건: 가격 < MA AND K < D
    청산 조건: 위 조건 불충족 시

    backtest.py와 동일한 방식:
    - 진입 시 entry_margin 고정 (포지션 내 비복리)
    - 청산 시 PnL 확정 → 다음 진입 시 전체 equity 사용 (거래 간 복리)
    - 펀딩비: 4H 봉당 누적
    - 슬리피지: 진입/청산 시 적용
    """
    df_4h = data['df_4h'].copy()
    df_daily = data['df_daily'].copy()

    if len(df_4h) < ma_period + 100:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    if len(df_daily) < stoch_k + stoch_smooth + 50:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    # MA 계산 (4H봉)
    df_4h['ma'] = df_4h['close'].rolling(window=ma_period).mean()

    # 스토캐스틱 계산 (일봉)
    slow_k, slow_d = calculate_stochastic(df_daily, stoch_k, stoch_smooth, stoch_d)
    df_daily['slow_k'] = slow_k
    df_daily['slow_d'] = slow_d

    # 전일 스토캐스틱 사용
    df_daily['date'] = df_daily['timestamp'].dt.date
    df_daily['prev_slow_k'] = df_daily['slow_k'].shift(1)
    df_daily['prev_slow_d'] = df_daily['slow_d'].shift(1)

    stoch_map = df_daily.set_index('date')[['prev_slow_k', 'prev_slow_d']].to_dict('index')

    df_4h['date'] = df_4h['timestamp'].dt.date
    df_4h['slow_k'] = df_4h['date'].map(lambda x: stoch_map.get(x, {}).get('prev_slow_k'))
    df_4h['slow_d'] = df_4h['date'].map(lambda x: stoch_map.get(x, {}).get('prev_slow_d'))

    df_4h = df_4h.dropna(subset=['ma', 'slow_k', 'slow_d']).reset_index(drop=True)

    if len(df_4h) < 100:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    # 시뮬레이션 (포지션 내 비복리, 거래 간 복리)
    initial_capital = 10000
    equity = initial_capital
    in_short = False
    entry_price = 0
    entry_margin = 0
    cum_funding = 0
    trade_count = 0
    equity_curve = [equity]

    for i in range(1, len(df_4h)):
        prev = df_4h.iloc[i - 1]
        curr = df_4h.iloc[i]

        opening_price = curr['open']
        ma_price = prev['ma']

        # 숏 조건: 가격 < MA AND K < D
        ma_condition = opening_price < ma_price
        stoch_condition = curr['slow_k'] < curr['slow_d']
        short_condition = ma_condition and stoch_condition

        if short_condition and not in_short:
            # ── 숏 진입 ──
            # 수수료 + 슬리피지 차감 (레버리지 적용)
            fee = equity * FEE_RATE * leverage
            slippage = equity * SLIPPAGE_PCT * leverage
            equity -= (fee + slippage)

            entry_margin = equity    # 진입 시 마진 고정
            entry_price = opening_price
            cum_funding = 0
            in_short = True
            trade_count += 1

        elif not short_condition and in_short:
            # ── 숏 청산 ──
            if entry_price > 0:
                price_return = opening_price / entry_price - 1
                trade_pnl = entry_margin * (-price_return) * leverage
                equity = entry_margin + trade_pnl - cum_funding

            # 청산 수수료 + 슬리피지
            fee = max(equity, 0) * FEE_RATE * leverage
            slippage = max(equity, 0) * SLIPPAGE_PCT * leverage
            equity -= (fee + slippage)
            equity = max(equity, 0)

            in_short = False
            cum_funding = 0
            trade_count += 1

        # 포지션 보유 중: 미실현 손익 + 펀딩비 (표시용)
        if in_short and entry_price > 0:
            price_return = curr['close'] / entry_price - 1
            unrealized_pnl = entry_margin * (-price_return) * leverage
            cum_funding += entry_margin * FUNDING_PER_4H
            display_equity = entry_margin + unrealized_pnl - cum_funding

            # 강제청산 체크: 숏은 가격 상승 시
            # 청산가 = entry_price * (1 + 1/leverage)
            if leverage > 0:
                liquidation_price = entry_price * (1 + 1 / leverage)
                if curr['high'] >= liquidation_price:
                    equity = 0
                    in_short = False
                    entry_price = 0
                    entry_margin = 0
                    cum_funding = 0
                    equity_curve.append(0)
                    break
            equity_curve.append(max(display_equity, 0))
        else:
            equity_curve.append(max(equity, 0))

    # 마지막 포지션 정리
    if in_short and entry_price > 0:
        last_price = df_4h.iloc[-1]['close']
        price_return = last_price / entry_price - 1
        trade_pnl = entry_margin * (-price_return) * leverage
        equity = entry_margin + trade_pnl - cum_funding
        fee = max(equity, 0) * FEE_RATE * leverage
        slippage = max(equity, 0) * SLIPPAGE_PCT * leverage
        equity -= (fee + slippage)

    # 성과 계산
    equity_curve = np.array(equity_curve)

    if len(equity_curve) < 2:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    # 강제청산으로 자본 0이면
    if equity_curve[-1] <= 0:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': trade_count}

    total_days = (df_4h.iloc[-1]['timestamp'] - df_4h.iloc[0]['timestamp']).days
    if total_days < 30:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    # equity_curve에서 0 값 방지 (로그 계산용)
    equity_curve = np.maximum(equity_curve, 0.01)

    total_return = equity_curve[-1] / equity_curve[0]
    if total_return <= 0:
        return {'cagr': -999, 'mdd': -100, 'sharpe': -999, 'trades': 0}

    years = total_days / 365.25
    cagr = (total_return ** (1 / years) - 1) * 100

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    mdd = drawdown.min()

    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 6)

    return {
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'trades': trade_count
    }


# ==========================================
# 베이지안 최적화 (2단계)
# ==========================================

def optimize_short_phase1(symbol: str, data: dict) -> dict:
    """1단계: 빠른 스크리닝 최적화"""

    def objective(trial):
        ma = trial.suggest_int('ma', *PARAM_RANGES_PHASE1['ma_range'])
        stoch_k = trial.suggest_int('stoch_k', *PARAM_RANGES_PHASE1['stoch_k_range'])
        stoch_smooth = trial.suggest_int('stoch_smooth', *PARAM_RANGES_PHASE1['stoch_smooth_range'])
        stoch_d = trial.suggest_int('stoch_d', *PARAM_RANGES_PHASE1['stoch_d_range'])
        lev = trial.suggest_int('leverage', *PARAM_RANGES_PHASE1['leverage_range'])

        result = backtest_short(data, ma, stoch_k, stoch_smooth, stoch_d, lev)

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
    best_result = backtest_short(data, best['ma'], best['stoch_k'], best['stoch_smooth'],
                                 best['stoch_d'], best['leverage'])

    return {
        'Symbol': symbol,
        'CAGR': best_result['cagr'],
        'MDD': best_result['mdd'],
        'Sharpe': best_result['sharpe'],
        'Trades': best_result['trades'],
        'Days': data['days'],
        'MA': best['ma'],
        'Stoch_K': best['stoch_k'],
        'Stoch_Smooth': best['stoch_smooth'],
        'Stoch_D': best['stoch_d'],
        'Leverage': best['leverage'],
        'Phase': 1
    }


def optimize_short_phase2(symbol: str, data: dict, phase1_result: dict) -> dict:
    """2단계: 정밀 파인튜닝 최적화"""

    def get_range(center, original_range, ratio=0.3):
        min_val, max_val = original_range
        delta = int((max_val - min_val) * ratio)
        new_min = max(min_val, center - delta)
        new_max = min(max_val, center + delta)
        return (new_min, new_max)

    param_ranges_phase2 = {
        'ma_range': get_range(phase1_result['MA'], PARAM_RANGES_PHASE1['ma_range']),
        'stoch_k_range': get_range(phase1_result['Stoch_K'], PARAM_RANGES_PHASE1['stoch_k_range']),
        'stoch_smooth_range': get_range(phase1_result['Stoch_Smooth'], PARAM_RANGES_PHASE1['stoch_smooth_range']),
        'stoch_d_range': get_range(phase1_result['Stoch_D'], PARAM_RANGES_PHASE1['stoch_d_range']),
        'leverage_range': PARAM_RANGES_PHASE1['leverage_range'],
    }

    print(f"    📊 2단계 탐색 범위:")
    print(f"       MA: {param_ranges_phase2['ma_range']} (1단계: {phase1_result['MA']})")
    print(f"       Stoch K: {param_ranges_phase2['stoch_k_range']} (1단계: {phase1_result['Stoch_K']})")
    print(f"       Stoch Smooth: {param_ranges_phase2['stoch_smooth_range']} (1단계: {phase1_result['Stoch_Smooth']})")
    print(f"       Stoch D: {param_ranges_phase2['stoch_d_range']} (1단계: {phase1_result['Stoch_D']})")

    def objective(trial):
        ma = trial.suggest_int('ma', *param_ranges_phase2['ma_range'])
        stoch_k = trial.suggest_int('stoch_k', *param_ranges_phase2['stoch_k_range'])
        stoch_smooth = trial.suggest_int('stoch_smooth', *param_ranges_phase2['stoch_smooth_range'])
        stoch_d = trial.suggest_int('stoch_d', *param_ranges_phase2['stoch_d_range'])
        lev = trial.suggest_int('leverage', *param_ranges_phase2['leverage_range'])

        result = backtest_short(data, ma, stoch_k, stoch_smooth, stoch_d, lev)

        cagr = result['cagr']
        mdd = result['mdd']
        sharpe = result['sharpe']

        if cagr < -50 or mdd < -90:
            return -9999

        score = cagr + (mdd * 0.7) + (sharpe * 20)
        return score

    sampler = TPESampler(seed=42, n_startup_trials=min(30, PHASE2_TRIALS // 3))
    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.enqueue_trial({
        'ma': phase1_result['MA'],
        'stoch_k': phase1_result['Stoch_K'],
        'stoch_smooth': phase1_result['Stoch_Smooth'],
        'stoch_d': phase1_result['Stoch_D'],
        'leverage': phase1_result['Leverage']
    })

    study.optimize(objective, n_trials=PHASE2_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_result = backtest_short(data, best['ma'], best['stoch_k'], best['stoch_smooth'],
                                 best['stoch_d'], best['leverage'])

    return {
        'Symbol': symbol,
        'CAGR': best_result['cagr'],
        'MDD': best_result['mdd'],
        'Sharpe': best_result['sharpe'],
        'Trades': best_result['trades'],
        'Days': data['days'],
        'MA': best['ma'],
        'Stoch_K': best['stoch_k'],
        'Stoch_Smooth': best['stoch_smooth'],
        'Stoch_D': best['stoch_d'],
        'Leverage': best['leverage'],
        'Phase': 2,
        'Phase1_CAGR': phase1_result['CAGR'],
        'Improvement': best_result['cagr'] - phase1_result['CAGR']
    }


# ==========================================
# 메인 실행
# ==========================================

def main():
    print("=" * 70)
    print("📉 Binance Futures 숏 포지션 2단계 베이지안 최적화 (수정 버전)")
    print("=" * 70)
    print(f"📌 수수료: {FEE_RATE*100:.2f}%")
    print(f"📌 슬리피지: {SLIPPAGE_PCT*100:.2f}%")
    print(f"📌 펀딩비: {FUNDING_RATE_8H*100:.3f}% / 8h")
    print(f"📌 방식: 포지션 내 비복리, 거래 간 복리")
    print(f"📌 1단계: 빠른 스크리닝 ({PHASE1_TRIALS} trials)")
    print(f"📌 2단계: CAGR >= {PHASE1_CAGR_THRESHOLD}% 코인만 정밀 파인튜닝 ({PHASE2_TRIALS} trials)")
    print(f"📌 전략: 가격 < MA & 스토캐스틱 K < D → 숏 진입")
    print(f"📌 레버리지 범위: {PARAM_RANGES_PHASE1['leverage_range'][0]}~{PARAM_RANGES_PHASE1['leverage_range'][1]}배")
    print("=" * 70)

    # 전체 심볼 가져오기
    all_symbols = get_all_futures_symbols()
    if not all_symbols:
        print("❌ 심볼 목록을 가져올 수 없습니다.")
        return

    print(f"\n총 {len(all_symbols)}개 심볼 대상 백테스트 시작")

    # ==========================================
    # 1단계: 빠른 스크리닝
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 1단계: 빠른 스크리닝 시작")
    print("=" * 70)

    phase1_results = []
    valid_data = {}

    for idx, symbol in enumerate(all_symbols, 1):
        print(f"\n[{idx}/{len(all_symbols)}] {symbol} - 1단계 스크리닝")

        data = prepare_coin_data(symbol, silent=True)
        if not data:
            print(f"  ⚠️ 데이터 부족, 스킵")
            continue

        valid_data[symbol] = data
        print(f"  📅 기간: {data['start_date'].date()} ~ {data['end_date'].date()} ({data['days']}일)")

        result = optimize_short_phase1(symbol, data)
        phase1_results.append(result)

        print(f"  ✅ 1단계 결과: CAGR={result['CAGR']:.1f}%, MDD={result['MDD']:.1f}%, Sharpe={result['Sharpe']:.2f}")

    # 1단계 결과 저장
    if phase1_results:
        df_phase1 = pd.DataFrame(phase1_results)
        df_phase1 = df_phase1.sort_values('CAGR', ascending=False)

        phase1_save_path = os.path.join(SAVE_DIR, f"binance_short_fixed_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_phase1.to_csv(phase1_save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 1단계 결과 저장: {phase1_save_path}")

    # ==========================================
    # 2단계 대상 선별
    # ==========================================
    phase2_candidates = [r for r in phase1_results if r['CAGR'] >= PHASE1_CAGR_THRESHOLD]

    print("\n" + "=" * 70)
    print(f"📊 1단계 완료 - 2단계 대상 선별")
    print("=" * 70)
    print(f"총 스크리닝: {len(phase1_results)}개")
    print(f"CAGR >= {PHASE1_CAGR_THRESHOLD}%: {len(phase2_candidates)}개")

    if phase2_candidates:
        print("\n2단계 대상 코인:")
        for r in sorted(phase2_candidates, key=lambda x: x['CAGR'], reverse=True)[:20]:
            print(f"  - {r['Symbol']}: CAGR={r['CAGR']:.1f}%, MDD={r['MDD']:.1f}%")

    # ==========================================
    # 2단계: 정밀 파인튜닝
    # ==========================================
    if not phase2_candidates:
        print("\n⚠️ 2단계 대상 코인이 없습니다.")
        final_results = phase1_results
    else:
        print("\n" + "=" * 70)
        print(f"📊 2단계: 정밀 파인튜닝 시작 ({len(phase2_candidates)}개 코인)")
        print("=" * 70)

        phase2_results = []

        for idx, phase1_result in enumerate(sorted(phase2_candidates, key=lambda x: x['CAGR'], reverse=True), 1):
            symbol = phase1_result['Symbol']
            data = valid_data[symbol]

            print(f"\n[{idx}/{len(phase2_candidates)}] {symbol} - 2단계 파인튜닝")
            print(f"  📊 1단계 CAGR: {phase1_result['CAGR']:.1f}%")

            result = optimize_short_phase2(symbol, data, phase1_result)
            phase2_results.append(result)

            print(f"  ✅ 2단계 결과: CAGR={result['CAGR']:.1f}%, MDD={result['MDD']:.1f}%, Sharpe={result['Sharpe']:.2f}")
            print(f"  📈 개선: {result['Improvement']:+.1f}%")

        # 2단계 결과 저장
        df_phase2 = pd.DataFrame(phase2_results)
        df_phase2 = df_phase2.sort_values('CAGR', ascending=False)

        phase2_save_path = os.path.join(SAVE_DIR, f"binance_short_fixed_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_phase2.to_csv(phase2_save_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 2단계 결과 저장: {phase2_save_path}")

        phase2_symbols = set(r['Symbol'] for r in phase2_results)
        final_results = phase2_results + [r for r in phase1_results if r['Symbol'] not in phase2_symbols]

    # ==========================================
    # 최종 결과 요약
    # ==========================================
    df_final = pd.DataFrame(final_results)
    df_final = df_final.sort_values('CAGR', ascending=False)

    final_save_path = os.path.join(SAVE_DIR, f"binance_short_fixed_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_final.to_csv(final_save_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print("📊 최종 결과 요약")
    print("=" * 70)

    print(f"\n총 분석 코인: {len(df_final)}개")
    print(f"수익 코인 (CAGR > 0%): {len(df_final[df_final['CAGR'] > 0])}개")
    print(f"평균 CAGR: {df_final['CAGR'].mean():.1f}%")

    # 상위 20개만 표시
    print("\n📈 상위 20개 코인:")
    display_cols = ['Symbol', 'CAGR', 'MDD', 'Sharpe', 'Trades', 'Leverage', 'Phase']
    available_cols = [c for c in display_cols if c in df_final.columns]
    print(df_final.head(20)[available_cols].to_string(index=False))

    print(f"\n💾 최종 결과 저장: {final_save_path}")

    # 봇 설정 형식으로 출력
    print("\n" + "=" * 70)
    print("📋 숏 봇 설정 - 상위 10개 (복사용)")
    print("=" * 70)

    print("\nSHORT_TRADING_CONFIGS = [")
    for _, row in df_final.head(10).iterrows():
        phase_str = f", Phase {row.get('Phase', 1)}"
        improvement = row.get('Improvement', 0)
        imp_str = f", 개선: {improvement:+.1f}%" if improvement != 0 else ""

        print(f"""    {{
        'symbol': '{row['Symbol']}',
        'ma_period': {row['MA']},
        'stoch_k_period': {row['Stoch_K']},
        'stoch_k_smooth': {row['Stoch_Smooth']},
        'stoch_d_period': {row['Stoch_D']},
        'leverage': {row['Leverage']},
        # CAGR: {row['CAGR']:.1f}%, MDD: {row['MDD']:.1f}%, Sharpe: {row['Sharpe']:.2f}{phase_str}{imp_str}
    }},""")
    print("]")

    # 통계 요약
    print("\n" + "=" * 70)
    print("📊 통계 요약")
    print("=" * 70)

    profitable = df_final[df_final['CAGR'] > 0]
    losing = df_final[df_final['CAGR'] <= 0]

    print(f"\n수익 코인 ({len(profitable)}개):")
    if len(profitable) > 0:
        print(f"  평균 CAGR: {profitable['CAGR'].mean():.1f}%")
        print(f"  평균 MDD: {profitable['MDD'].mean():.1f}%")
        print(f"  평균 Sharpe: {profitable['Sharpe'].mean():.2f}")

    print(f"\n손실 코인 ({len(losing)}개):")
    if len(losing) > 0:
        print(f"  평균 CAGR: {losing['CAGR'].mean():.1f}%")
        print(f"  평균 MDD: {losing['MDD'].mean():.1f}%")


if __name__ == "__main__":
    main()
