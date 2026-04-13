"""
================================================================================
6코인 NEW 파라미터 그리드 분석 + 강건한 봉우리 선택
================================================================================
- 코인별 2단계 그리드 탐색 (Primary → Secondary)
- 4D 인접 평균(Smoothing)으로 바늘 봉우리 제거
- 넓은 봉우리 꼭대기 선택 → Robust 파라미터 도출
- 기존 NEW vs Robust 포트폴리오 비교
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter
from datetime import datetime
import time

from backtest_bots import (
    prepare_coin_data, prepare_signals,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H,
    calculate_portfolio_performance,
)

SAVE_DIR = os.path.expanduser("~/Downloads")
FUNDING_PER_4H = FUNDING_RATE_8H * 0.5
CAPITAL = 10000

# ==========================================
# 그리드 정의
# ==========================================
MA_GRID = list(range(20, 380, 1))      # 15값: 20,45,...,370
SK_GRID = list(range(5, 155, 1))       # 10값: 5,20,...,140
SKS_GRID = list(range(3, 83, 1))       # 8값: 3,13,...,73
SD_GRID = list(range(3, 53, 1))        # 5값: 3,13,...,43

# 6코인 현재 NEW 파라미터
NEW_CONFIGS = {
    'BTCUSDT': {'priority': 'short', 'short_ma': 254, 'short_sk': 27, 'short_sks': 23, 'short_sd': 19, 'short_lev': 1,
                'long_ma': 350, 'long_sk': 36, 'long_sks': 32, 'long_sd': 10, 'long_lev': 5},
    'ETHUSDT': {'priority': 'long', 'long_ma': 322, 'long_sk': 54, 'long_sks': 10, 'long_sd': 36, 'long_lev': 5,
                'short_ma': 220, 'short_sk': 31, 'short_sks': 44, 'short_sd': 26, 'short_lev': 2},
    'XRPUSDT': {'priority': 'short', 'short_ma': 269, 'short_sk': 121, 'short_sks': 35, 'short_sd': 47, 'short_lev': 1,
                'long_ma': 107, 'long_sk': 14, 'long_sks': 13, 'long_sd': 23, 'long_lev': 5},
    'SOLUSDT': {'priority': 'long', 'long_ma': 73, 'long_sk': 33, 'long_sks': 16, 'long_sd': 38, 'long_lev': 4,
                'short_ma': 314, 'short_sk': 37, 'short_sks': 34, 'short_sd': 44, 'short_lev': 1},
    'DOGEUSDT': {'priority': 'short', 'short_ma': 250, 'short_sk': 36, 'short_sks': 15, 'short_sd': 40, 'short_lev': 1,
                 'long_ma': 31, 'long_sk': 48, 'long_sks': 50, 'long_sd': 17, 'long_lev': 2},
    'ADAUSDT': {'priority': 'short', 'short_ma': 80, 'short_sk': 31, 'short_sks': 77, 'short_sd': 46, 'short_lev': 1,
                'long_ma': 296, 'long_sk': 19, 'long_sks': 53, 'long_sd': 15, 'long_lev': 3},
}


# ==========================================
# 지표 사전 계산
# ==========================================
def precompute_indicators(data, ma_grid, sk_grid, sks_grid, sd_grid):
    """모든 MA / 스토캐스틱 조합을 한번에 계산"""
    df_4h = data['df_4h'].sort_values('timestamp').reset_index(drop=True)
    df_daily = data['df_daily'].sort_values('timestamp').reset_index(drop=True)
    n = len(df_4h)

    opens = df_4h['open'].values.astype(np.float64)
    highs = df_4h['high'].values.astype(np.float64)
    lows = df_4h['low'].values.astype(np.float64)
    closes = df_4h['close'].values.astype(np.float64)
    dates_4h = df_4h['timestamp'].dt.date.values

    # 4H MA → prev bar's MA (shift by 1)
    prev_mas = {}
    close_series = pd.Series(closes)
    for ma in ma_grid:
        ma_arr = close_series.rolling(ma).mean().values
        shifted = np.empty(n)
        shifted[0] = np.nan
        shifted[1:] = ma_arr[:-1]
        prev_mas[ma] = shifted

    # Daily stochastic
    d_low = pd.Series(df_daily['low'].values.astype(np.float64))
    d_high = pd.Series(df_daily['high'].values.astype(np.float64))
    d_close = df_daily['close'].values.astype(np.float64)
    daily_dates = df_daily['timestamp'].dt.date.values
    daily_date_idx = {}
    for i, d in enumerate(daily_dates):
        daily_date_idx[d] = i

    fast_ks = {}
    for sk in sk_grid:
        lo = d_low.rolling(sk).min().values
        hi = d_high.rolling(sk).max().values
        denom = hi - lo
        denom[denom == 0] = np.nan
        fast_ks[sk] = (d_close - lo) / denom * 100

    # (sk, sks, sd) → (prev_stoch_k, prev_stoch_d) arrays for 4H bars
    prev_stochs = {}
    for sk in sk_grid:
        fk = fast_ks[sk]
        fk_series = pd.Series(fk)
        for sks in sks_grid:
            slow_k = fk_series.rolling(sks).mean().values
            sk_series = pd.Series(slow_k)
            for sd in sd_grid:
                slow_d = sk_series.rolling(sd).mean().values
                # Previous day's values
                prev_sk_d = np.empty(len(daily_dates))
                prev_sd_d = np.empty(len(daily_dates))
                prev_sk_d[0] = np.nan
                prev_sd_d[0] = np.nan
                prev_sk_d[1:] = slow_k[:-1]
                prev_sd_d[1:] = slow_d[:-1]
                # Map to 4H
                sk_4h = np.full(n, np.nan)
                sd_4h = np.full(n, np.nan)
                for j in range(n):
                    idx = daily_date_idx.get(dates_4h[j])
                    if idx is not None:
                        sk_4h[j] = prev_sk_d[idx]
                        sd_4h[j] = prev_sd_d[idx]
                # Shift by 1 bar (prev bar's value)
                prev_k = np.empty(n)
                prev_d = np.empty(n)
                prev_k[0] = np.nan
                prev_d[0] = np.nan
                prev_k[1:] = sk_4h[:-1]
                prev_d[1:] = sd_4h[:-1]
                prev_stochs[(sk, sks, sd)] = (prev_k, prev_d)

    return opens, highs, lows, closes, n, prev_mas, prev_stochs


# ==========================================
# 고속 개별 코인 백테스트
# ==========================================
def fast_backtest(opens, highs, lows, closes, n,
                  prev_ma_long, prev_sk_long, prev_sd_long,
                  prev_ma_short, prev_sk_short, prev_sd_short,
                  long_lev, short_lev, priority_is_long):
    """고속 개별 코인 백테스트. Returns (sharpe, cagr, mdd)"""
    FEE_SLIP = FEE_RATE + SLIPPAGE_PCT
    FUND = FUNDING_PER_4H

    equity = 10000.0
    side = 0  # 0=flat, 1=long, -1=short
    ep = 0.0  # entry price
    em = 0.0  # entry margin
    cf = 0.0  # cumulative funding
    lev = 0

    peak = equity
    max_dd = 0.0
    prev_eq = equity
    sr = 0.0   # sum returns
    sr2 = 0.0  # sum returns squared
    nr = 0
    start_i = -1

    for i in range(1, n):
        pml = prev_ma_long[i]
        pms = prev_ma_short[i]
        pkl = prev_sk_long[i]
        pdl = prev_sd_long[i]
        pks = prev_sk_short[i]
        pds = prev_sd_short[i]

        # NaN check
        if pml != pml or pms != pms or pkl != pkl or pdl != pdl or pks != pks or pds != pds:
            if side != 0:
                cp = closes[i]
                pr = cp / ep - 1
                if side == -1: pr = -pr
                cf += em * lev * FUND
                ceq = em + em * pr * lev - cf
                if ceq < 0: ceq = 0
            else:
                ceq = equity
            if prev_eq > 0:
                r = ceq / prev_eq - 1
                sr += r; sr2 += r * r; nr += 1
            if ceq > peak: peak = ceq
            if peak > 0:
                dd = (ceq - peak) / peak
                if dd < max_dd: max_dd = dd
            prev_eq = ceq
            continue

        if start_i < 0:
            start_i = i

        op = opens[i]
        ls = (op > pml) and (pkl > pdl)
        ss = (op < pms) and (pks < pds)

        # --- Trading logic ---
        def close_pos():
            nonlocal equity, side, ep, em, cf, lev
            if ep > 0:
                pr = op / ep - 1
                if side == -1: pr = -pr
                pnl = em * pr * lev
                eq = em + pnl - cf
                en = abs(em * lev * (op / ep))
                eq -= en * FEE_SLIP
                equity = max(eq, 0)
            side = 0; ep = 0; em = 0; cf = 0; lev = 0

        def enter_long():
            nonlocal equity, side, ep, em, cf, lev
            fee = equity * FEE_SLIP * long_lev
            equity -= fee
            em = equity; ep = op; cf = 0; lev = long_lev; side = 1

        def enter_short():
            nonlocal equity, side, ep, em, cf, lev
            fee = equity * FEE_SLIP * short_lev
            equity -= fee
            em = equity; ep = op; cf = 0; lev = short_lev; side = -1

        if priority_is_long:
            if ls:
                if side == -1: close_pos()
                if side == 0 and equity > 0: enter_long()
            elif ss and side != 1:
                if side == 0 and equity > 0: enter_short()
            else:
                if side != 0: close_pos()
        else:
            if ss:
                if side == 1: close_pos()
                if side == 0 and equity > 0: enter_short()
            elif ls and side != -1:
                if side == 0 and equity > 0: enter_long()
            else:
                if side != 0: close_pos()

        # Equity tracking
        if side != 0 and ep > 0:
            cp = closes[i]
            pr = cp / ep - 1
            if side == -1: pr = -pr
            cf += em * lev * FUND
            ceq = em + em * pr * lev - cf

            # Liquidation check
            if lev > 0:
                if side == 1 and lows[i] <= ep * (1 - 1.0 / lev):
                    equity = 0; side = 0; ep = 0; em = 0; cf = 0; lev = 0; ceq = 0
                elif side == -1 and highs[i] >= ep * (1 + 1.0 / lev):
                    equity = 0; side = 0; ep = 0; em = 0; cf = 0; lev = 0; ceq = 0

            if ceq < 0: ceq = 0
        else:
            ceq = max(equity, 0)

        if prev_eq > 0:
            r = ceq / prev_eq - 1
            sr += r; sr2 += r * r; nr += 1
        if ceq > peak: peak = ceq
        if peak > 0:
            dd = (ceq - peak) / peak
            if dd < max_dd: max_dd = dd
        prev_eq = ceq

        if ceq <= 0:
            break

    if nr < 100 or start_i < 0:
        return (0, 0, -100)

    final_eq = prev_eq
    days = (n - start_i) * 4 / 24  # approximate days
    years = days / 365.25
    if years < 0.5 or final_eq <= 0:
        return (0, 0, -100)

    cagr = (final_eq / 10000) ** (1 / years) - 1
    cagr *= 100

    mean_r = sr / nr
    var_r = sr2 / nr - mean_r * mean_r
    if var_r > 0:
        sharpe = mean_r / (var_r ** 0.5) * (365 * 6) ** 0.5
    else:
        sharpe = 0

    return (sharpe, cagr, max_dd * 100)


# ==========================================
# 그리드 서치 (4D → Smoothing → Robust Peak)
# ==========================================
def grid_search_direction(opens, highs, lows, closes, n, prev_mas, prev_stochs,
                          fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
                          fixed_lev, search_lev, priority_is_long, search_is_long):
    """
    한 방향 파라미터 그리드 서치.
    search_is_long=True: 롱 파라미터 탐색, 숏은 고정
    search_is_long=False: 숏 파라미터 탐색, 롱은 고정
    """
    n_ma = len(MA_GRID)
    n_sk = len(SK_GRID)
    n_sks = len(SKS_GRID)
    n_sd = len(SD_GRID)
    total = n_ma * n_sk * n_sks * n_sd

    sharpe_grid = np.full((n_ma, n_sk, n_sks, n_sd), -999.0)
    cagr_grid = np.full((n_ma, n_sk, n_sks, n_sd), -999.0)

    count = 0
    for i_ma, ma in enumerate(MA_GRID):
        search_ma = prev_mas.get(ma)
        if search_ma is None:
            continue
        for i_sk, sk in enumerate(SK_GRID):
            for i_sks, sks in enumerate(SKS_GRID):
                for i_sd, sd in enumerate(SD_GRID):
                    stoch = prev_stochs.get((sk, sks, sd))
                    if stoch is None:
                        continue
                    s_sk, s_sd = stoch

                    if search_is_long:
                        result = fast_backtest(
                            opens, highs, lows, closes, n,
                            search_ma, s_sk, s_sd,
                            fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
                            search_lev, fixed_lev, priority_is_long)
                    else:
                        result = fast_backtest(
                            opens, highs, lows, closes, n,
                            fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
                            search_ma, s_sk, s_sd,
                            fixed_lev, search_lev, priority_is_long)

                    sharpe_grid[i_ma, i_sk, i_sks, i_sd] = result[0]
                    cagr_grid[i_ma, i_sk, i_sks, i_sd] = result[1]
                    count += 1

    # Smoothing (3x3x3x3 uniform filter)
    valid_mask = sharpe_grid > -900
    sharpe_smooth = sharpe_grid.copy()
    sharpe_smooth[~valid_mask] = 0
    count_smooth = valid_mask.astype(float)

    sharpe_sum = uniform_filter(sharpe_smooth, size=3, mode='constant', cval=0)
    count_sum = uniform_filter(count_smooth, size=3, mode='constant', cval=0)
    count_sum[count_sum == 0] = 1
    sharpe_neighbor_avg = sharpe_sum / count_sum
    sharpe_neighbor_avg[~valid_mask] = -999

    # Raw peak
    raw_idx = np.unravel_index(np.argmax(sharpe_grid), sharpe_grid.shape)
    raw_sharpe = sharpe_grid[raw_idx]
    raw_cagr = cagr_grid[raw_idx]

    # Smoothed peak (robust)
    robust_idx = np.unravel_index(np.argmax(sharpe_neighbor_avg), sharpe_neighbor_avg.shape)
    robust_sharpe = sharpe_neighbor_avg[robust_idx]
    robust_raw_sharpe = sharpe_grid[robust_idx]
    robust_cagr = cagr_grid[robust_idx]

    raw_params = (MA_GRID[raw_idx[0]], SK_GRID[raw_idx[1]], SKS_GRID[raw_idx[2]], SD_GRID[raw_idx[3]])
    robust_params = (MA_GRID[robust_idx[0]], SK_GRID[robust_idx[1]], SKS_GRID[robust_idx[2]], SD_GRID[robust_idx[3]])

    return {
        'sharpe_grid': sharpe_grid,
        'cagr_grid': cagr_grid,
        'sharpe_smooth': sharpe_neighbor_avg,
        'raw_params': raw_params, 'raw_sharpe': raw_sharpe, 'raw_cagr': raw_cagr,
        'robust_params': robust_params, 'robust_sharpe': robust_sharpe,
        'robust_raw_sharpe': robust_raw_sharpe, 'robust_cagr': robust_cagr,
        'total_combos': count,
    }


# ==========================================
# 코인별 그리드 분석
# ==========================================
def analyze_coin(symbol, data, config):
    print(f"\n  --- {symbol} ---")
    t0 = time.time()

    priority = config['priority']
    priority_is_long = (priority == 'long')

    # Pre-compute indicators
    opens, highs, lows, closes, n, prev_mas, prev_stochs = precompute_indicators(
        data, MA_GRID, SK_GRID, SKS_GRID, SD_GRID)
    t1 = time.time()
    print(f"    지표 계산: {t1-t0:.1f}초 (MA {len(prev_mas)}개, Stoch {len(prev_stochs)}개)")

    # Stage 1: Primary direction search
    if priority_is_long:
        # Search LONG, fix SHORT
        fixed_ma_key = min(MA_GRID, key=lambda x: abs(x - config['short_ma']))
        fixed_stoch_key = (
            min(SK_GRID, key=lambda x: abs(x - config['short_sk'])),
            min(SKS_GRID, key=lambda x: abs(x - config['short_sks'])),
            min(SD_GRID, key=lambda x: abs(x - config['short_sd'])),
        )
        fixed_prev_ma = prev_mas[fixed_ma_key]
        fixed_stoch = prev_stochs.get(fixed_stoch_key)
        if fixed_stoch is None:
            # Find closest
            all_keys = list(prev_stochs.keys())
            dists = [abs(k[0]-config['short_sk'])+abs(k[1]-config['short_sks'])+abs(k[2]-config['short_sd']) for k in all_keys]
            fixed_stoch_key = all_keys[np.argmin(dists)]
            fixed_stoch = prev_stochs[fixed_stoch_key]
        fixed_prev_sk, fixed_prev_sd = fixed_stoch
        fixed_lev = config['short_lev']
        search_lev = config['long_lev']
        search_is_long = True
        print(f"    Stage1: 롱 파라미터 그리드 탐색 (숏 고정)")
    else:
        # Search SHORT, fix LONG
        fixed_ma_key = min(MA_GRID, key=lambda x: abs(x - config['long_ma']))
        fixed_stoch_key = (
            min(SK_GRID, key=lambda x: abs(x - config['long_sk'])),
            min(SKS_GRID, key=lambda x: abs(x - config['long_sks'])),
            min(SD_GRID, key=lambda x: abs(x - config['long_sd'])),
        )
        fixed_prev_ma = prev_mas[fixed_ma_key]
        fixed_stoch = prev_stochs.get(fixed_stoch_key)
        if fixed_stoch is None:
            all_keys = list(prev_stochs.keys())
            dists = [abs(k[0]-config['long_sk'])+abs(k[1]-config['long_sks'])+abs(k[2]-config['long_sd']) for k in all_keys]
            fixed_stoch_key = all_keys[np.argmin(dists)]
            fixed_stoch = prev_stochs[fixed_stoch_key]
        fixed_prev_sk, fixed_prev_sd = fixed_stoch
        fixed_lev = config['long_lev']
        search_lev = config['short_lev']
        search_is_long = False
        print(f"    Stage1: 숏 파라미터 그리드 탐색 (롱 고정)")

    s1 = grid_search_direction(
        opens, highs, lows, closes, n, prev_mas, prev_stochs,
        fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
        fixed_lev, search_lev, priority_is_long, search_is_long)

    t2 = time.time()
    dir_label = "롱" if search_is_long else "숏"
    print(f"    Stage1 완료 ({t2-t1:.1f}초, {s1['total_combos']:,}개)")
    print(f"      Raw peak   [{dir_label}]: MA={s1['raw_params'][0]}, SK={s1['raw_params'][1]}, "
          f"SKs={s1['raw_params'][2]}, SD={s1['raw_params'][3]} → Sharpe={s1['raw_sharpe']:.3f}, CAGR={s1['raw_cagr']:.0f}%")
    print(f"      Robust peak [{dir_label}]: MA={s1['robust_params'][0]}, SK={s1['robust_params'][1]}, "
          f"SKs={s1['robust_params'][2]}, SD={s1['robust_params'][3]} → Sharpe={s1['robust_raw_sharpe']:.3f}, CAGR={s1['robust_cagr']:.0f}%"
          f" (neighbor avg={s1['robust_sharpe']:.3f})")

    # Stage 2: Secondary direction search (fix primary at robust peak)
    r_ma, r_sk, r_sks, r_sd = s1['robust_params']
    robust_ma_key = min(MA_GRID, key=lambda x: abs(x - r_ma))
    robust_stoch_key = (
        min(SK_GRID, key=lambda x: abs(x - r_sk)),
        min(SKS_GRID, key=lambda x: abs(x - r_sks)),
        min(SD_GRID, key=lambda x: abs(x - r_sd)),
    )
    robust_prev_ma = prev_mas[robust_ma_key]
    robust_stoch = prev_stochs.get(robust_stoch_key)
    if robust_stoch is None:
        all_keys = list(prev_stochs.keys())
        dists = [abs(k[0]-r_sk)+abs(k[1]-r_sks)+abs(k[2]-r_sd) for k in all_keys]
        robust_stoch_key = all_keys[np.argmin(dists)]
        robust_stoch = prev_stochs[robust_stoch_key]
    robust_prev_sk, robust_prev_sd = robust_stoch

    if search_is_long:
        # Now search SHORT (fix LONG at robust)
        s2_search_is_long = False
        s2_fixed_ma = robust_prev_ma
        s2_fixed_sk = robust_prev_sk
        s2_fixed_sd = robust_prev_sd
        s2_fixed_lev = search_lev  # long lev
        s2_search_lev = fixed_lev  # short lev
        print(f"    Stage2: 숏 파라미터 그리드 탐색 (롱=Robust 고정)")
    else:
        s2_search_is_long = True
        s2_fixed_ma = robust_prev_ma
        s2_fixed_sk = robust_prev_sk
        s2_fixed_sd = robust_prev_sd
        s2_fixed_lev = search_lev
        s2_search_lev = fixed_lev
        print(f"    Stage2: 롱 파라미터 그리드 탐색 (숏=Robust 고정)")

    s2 = grid_search_direction(
        opens, highs, lows, closes, n, prev_mas, prev_stochs,
        s2_fixed_ma, s2_fixed_sk, s2_fixed_sd,
        s2_fixed_lev, s2_search_lev, priority_is_long, s2_search_is_long)

    t3 = time.time()
    dir_label2 = "롱" if s2_search_is_long else "숏"
    print(f"    Stage2 완료 ({t3-t2:.1f}초, {s2['total_combos']:,}개)")
    print(f"      Raw peak   [{dir_label2}]: MA={s2['raw_params'][0]}, SK={s2['raw_params'][1]}, "
          f"SKs={s2['raw_params'][2]}, SD={s2['raw_params'][3]} → Sharpe={s2['raw_sharpe']:.3f}, CAGR={s2['raw_cagr']:.0f}%")
    print(f"      Robust peak [{dir_label2}]: MA={s2['robust_params'][0]}, SK={s2['robust_params'][1]}, "
          f"SKs={s2['robust_params'][2]}, SD={s2['robust_params'][3]} → Sharpe={s2['robust_raw_sharpe']:.3f}, CAGR={s2['robust_cagr']:.0f}%"
          f" (neighbor avg={s2['robust_sharpe']:.3f})")

    # Compile robust config
    if priority_is_long:
        # s1 searched long, s2 searched short
        robust_config = {
            'priority': priority,
            'long_ma': s1['robust_params'][0], 'long_sk': s1['robust_params'][1],
            'long_sks': s1['robust_params'][2], 'long_sd': s1['robust_params'][3],
            'long_lev': config['long_lev'],
            'short_ma': s2['robust_params'][0], 'short_sk': s2['robust_params'][1],
            'short_sks': s2['robust_params'][2], 'short_sd': s2['robust_params'][3],
            'short_lev': config['short_lev'],
        }
    else:
        # s1 searched short, s2 searched long
        robust_config = {
            'priority': priority,
            'short_ma': s1['robust_params'][0], 'short_sk': s1['robust_params'][1],
            'short_sks': s1['robust_params'][2], 'short_sd': s1['robust_params'][3],
            'short_lev': config['short_lev'],
            'long_ma': s2['robust_params'][0], 'long_sk': s2['robust_params'][1],
            'long_sks': s2['robust_params'][2], 'long_sd': s2['robust_params'][3],
            'long_lev': config['long_lev'],
        }

    print(f"    총 소요: {t3-t0:.1f}초")

    return {
        'symbol': symbol,
        'stage1': s1, 'stage2': s2,
        'robust_config': robust_config,
        'original_config': config,
    }


# ==========================================
# 포트폴리오 백테스트 (compare_old_vs_new_params.py 동일)
# ==========================================
def run_portfolio(configs_dict, coin_data_cache, initial_capital=CAPITAL):
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
        df_bt = df_bt.dropna(subset=['ma_long', 'long_k', 'long_d', 'ma_short', 'short_k', 'short_d'])
        df_bt = df_bt.set_index('timestamp').sort_index()
        if len(df_bt) >= 100:
            coin_signals[symbol] = {'df': df_bt, 'long_lev': config['long_lev'],
                                     'short_lev': config['short_lev'], 'priority': config.get('priority', 'long')}
    if not coin_signals:
        return None

    all_ts = set()
    for s_data in coin_signals.values():
        all_ts.update(s_data['df'].index.tolist())
    timeline = sorted(all_ts)

    cash = initial_capital
    positions = {}
    equity_curve = []; eq_timestamps = []
    total_trades = 0; long_trades = 0; short_trades = 0

    for ts in timeline:
        active_coins = [s for s, d in coin_signals.items() if ts in d['df'].index]
        n_active = len(active_coins)
        if n_active == 0:
            equity_curve.append(cash + sum(p['margin'] for p in positions.values()))
            eq_timestamps.append(ts)
            continue
        for symbol in active_coins:
            s_data = coin_signals[symbol]; df = s_data['df']
            if ts not in df.index: continue
            curr = df.loc[ts]; op = curr['open']
            ci = df.index.get_loc(ts)
            if ci == 0: continue
            prev = df.iloc[ci - 1]
            ll = s_data['long_lev']; sl = s_data['short_lev']; pri = s_data['priority']
            ls = (op > prev['ma_long']) and (prev['long_k'] > prev['long_d'])
            ss = (op < prev['ma_short']) and (prev['short_k'] < prev['short_d'])
            pos = positions.get(symbol)
            def settle(p):
                nonlocal cash, total_trades
                pr = op / p['entry_price'] - 1
                if p['side'] == 'short': pr = -pr
                pnl = p['margin'] * pr * p['lev']
                s = p['margin'] + pnl - p['cum_funding']
                s -= abs(p['margin'] * p['lev'] * (op / p['entry_price'])) * (FEE_RATE + SLIPPAGE_PCT)
                cash += max(s, 0); total_trades += 1
            def enter(side, lev):
                nonlocal cash, total_trades, long_trades, short_trades
                teq = cash + sum(p['margin'] for p in positions.values())
                alloc = min(teq / n_active, cash * 0.995)
                if alloc > 1:
                    m = alloc - alloc * (FEE_RATE + SLIPPAGE_PCT) * lev
                    if m > 0:
                        cash -= alloc
                        positions[symbol] = {'side': side, 'entry_price': op, 'margin': m, 'cum_funding': 0, 'lev': lev}
                        total_trades += 1
                        if side == 'long': long_trades += 1
                        else: short_trades += 1
            if pri == 'long':
                if ls:
                    if pos and pos['side'] == 'short': settle(pos); del positions[symbol]; pos = None
                    if pos is None: enter('long', ll)
                elif ss and (pos is None or pos['side'] != 'long'):
                    if pos is None: enter('short', sl)
                else:
                    if pos: settle(pos); del positions[symbol]
            else:
                if ss:
                    if pos and pos['side'] == 'long': settle(pos); del positions[symbol]; pos = None
                    if pos is None: enter('short', sl)
                elif ls and (pos is None or pos['side'] != 'short'):
                    if pos is None: enter('long', ll)
                else:
                    if pos: settle(pos); del positions[symbol]
            pos = positions.get(symbol)
            if pos and ts in df.index:
                pos['cum_funding'] += pos['margin'] * pos['lev'] * FUNDING_PER_4H
                if pos['side'] == 'long' and pos['lev'] > 0:
                    if curr['low'] <= pos['entry_price'] * (1 - 1/pos['lev']): del positions[symbol]; total_trades += 1
                elif pos['side'] == 'short' and pos['lev'] > 0:
                    if curr['high'] >= pos['entry_price'] * (1 + 1/pos['lev']): del positions[symbol]; total_trades += 1
        ur = 0
        for sym, pos in positions.items():
            df = coin_signals[sym]['df']
            if ts in df.index:
                cp = df.loc[ts, 'close']; pr = cp / pos['entry_price'] - 1
                if pos['side'] == 'short': pr = -pr
                v = pos['margin'] + pos['margin'] * pr * pos['lev'] - pos['cum_funding']
                ur += max(v, 0)
            else: ur += pos['margin']
        equity_curve.append(cash + ur); eq_timestamps.append(ts)

    for sym, pos in list(positions.items()):
        df = coin_signals[sym]['df']; lp = df.iloc[-1]['close']
        pr = lp / pos['entry_price'] - 1
        if pos['side'] == 'short': pr = -pr
        pnl = pos['margin'] * pr * pos['lev']
        s = pos['margin'] + pnl - pos['cum_funding']
        s -= abs(pos['margin'] * pos['lev'] * (lp / pos['entry_price'])) * (FEE_RATE + SLIPPAGE_PCT)
        cash += max(s, 0)

    perf = calculate_portfolio_performance(equity_curve, eq_timestamps)
    if perf is None: return None
    perf['trades'] = total_trades; perf['long_trades'] = long_trades
    perf['short_trades'] = short_trades
    perf['equity_curve'] = equity_curve; perf['timestamps'] = eq_timestamps
    return perf


# ==========================================
# 차트 생성
# ==========================================
def plot_grid_results(all_results, new_perf, robust_perf, save_dir):
    # 1. 코인별 2D heatmap (MA × SK, best over SKs/SD)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for idx, res in enumerate(all_results):
        ax = axes[idx // 3][idx % 3]
        symbol = res['symbol']
        # Stage1 heatmap (primary direction)
        sg = res['stage1']['sharpe_smooth']
        # Collapse to MA × SK (max over SKs, SD)
        heatmap = np.nanmax(sg, axis=(2, 3))
        heatmap[heatmap < -900] = np.nan
        im = ax.imshow(heatmap.T, aspect='auto', origin='lower', cmap='RdYlGn',
                       extent=[MA_GRID[0], MA_GRID[-1], SK_GRID[0], SK_GRID[-1]])
        pri = res['original_config']['priority']
        dir_label = "숏" if pri == 'short' else "롱"
        # Mark original
        orig = res['original_config']
        if pri == 'short':
            o_ma, o_sk = orig['short_ma'], orig['short_sk']
        else:
            o_ma, o_sk = orig['long_ma'], orig['long_sk']
        ax.plot(o_ma, o_sk, 'bx', markersize=12, markeredgewidth=3, label='Original')
        # Mark robust
        r_ma, r_sk = res['stage1']['robust_params'][0], res['stage1']['robust_params'][1]
        ax.plot(r_ma, r_sk, 'r*', markersize=15, label='Robust')
        ax.set_xlabel('MA Period')
        ax.set_ylabel('Stoch K Period')
        ax.set_title(f'{symbol} [{dir_label} primary]\nSmoothed Sharpe (MA × SK)', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Grid Analysis: Smoothed Sharpe Landscape (MA × Stoch K)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path1 = os.path.join(save_dir, '6coin_grid_heatmaps.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1] 히트맵: {path1}")

    # 2. 파라미터 비교 테이블 차트
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')
    headers = ['Coin', 'Dir', 'Param', 'Original', 'Raw Peak', 'Robust', 'Orig Sharpe', 'Robust Sharpe']
    rows = []
    for res in all_results:
        sym = res['symbol']
        pri = res['original_config']['priority']
        orig = res['original_config']
        rob = res['robust_config']

        if pri == 'short':
            # Stage1 = short (primary), Stage2 = long
            rows.append([sym, '숏(1차)', 'MA/SK/SKs/SD',
                         f"{orig['short_ma']}/{orig['short_sk']}/{orig['short_sks']}/{orig['short_sd']}",
                         f"{res['stage1']['raw_params'][0]}/{res['stage1']['raw_params'][1]}/{res['stage1']['raw_params'][2]}/{res['stage1']['raw_params'][3]}",
                         f"{rob['short_ma']}/{rob['short_sk']}/{rob['short_sks']}/{rob['short_sd']}",
                         f"{res['stage1']['raw_sharpe']:.2f}",
                         f"{res['stage1']['robust_raw_sharpe']:.2f}"])
            rows.append(['', '롱(2차)', 'MA/SK/SKs/SD',
                         f"{orig['long_ma']}/{orig['long_sk']}/{orig['long_sks']}/{orig['long_sd']}",
                         f"{res['stage2']['raw_params'][0]}/{res['stage2']['raw_params'][1]}/{res['stage2']['raw_params'][2]}/{res['stage2']['raw_params'][3]}",
                         f"{rob['long_ma']}/{rob['long_sk']}/{rob['long_sks']}/{rob['long_sd']}",
                         f"{res['stage2']['raw_sharpe']:.2f}",
                         f"{res['stage2']['robust_raw_sharpe']:.2f}"])
        else:
            rows.append([sym, '롱(1차)', 'MA/SK/SKs/SD',
                         f"{orig['long_ma']}/{orig['long_sk']}/{orig['long_sks']}/{orig['long_sd']}",
                         f"{res['stage1']['raw_params'][0]}/{res['stage1']['raw_params'][1]}/{res['stage1']['raw_params'][2]}/{res['stage1']['raw_params'][3]}",
                         f"{rob['long_ma']}/{rob['long_sk']}/{rob['long_sks']}/{rob['long_sd']}",
                         f"{res['stage1']['raw_sharpe']:.2f}",
                         f"{res['stage1']['robust_raw_sharpe']:.2f}"])
            rows.append(['', '숏(2차)', 'MA/SK/SKs/SD',
                         f"{orig['short_ma']}/{orig['short_sk']}/{orig['short_sks']}/{orig['short_sd']}",
                         f"{res['stage2']['raw_params'][0]}/{res['stage2']['raw_params'][1]}/{res['stage2']['raw_params'][2]}/{res['stage2']['raw_params'][3]}",
                         f"{rob['short_ma']}/{rob['short_sk']}/{rob['short_sks']}/{rob['short_sd']}",
                         f"{res['stage2']['raw_sharpe']:.2f}",
                         f"{res['stage2']['robust_raw_sharpe']:.2f}"])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Original vs Raw Peak vs Robust Parameters', fontweight='bold', fontsize=13, pad=20)
    plt.tight_layout()
    path2 = os.path.join(save_dir, '6coin_grid_params_table.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 파라미터 비교: {path2}")

    # 3. Portfolio comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = ['Original\n(Bayesian)', 'Robust\n(Grid)']

    for ax, metric, label in zip(axes,
                                  [(new_perf['cagr'], robust_perf['cagr']),
                                   (abs(new_perf['mdd']), abs(robust_perf['mdd'])),
                                   (new_perf['sharpe'], robust_perf['sharpe'])],
                                  ['CAGR (%)', 'Max Drawdown (%)', 'Sharpe']):
        cols = ['#FF8A65', '#4CAF50']
        bars = ax.bar(names, metric, color=cols, alpha=0.85, width=0.5)
        for bar, val in zip(bars, metric):
            fmt = f'{val:.0f}%' if 'CAGR' in label or 'Draw' in label else f'{val:.3f}'
            if 'Draw' in label: fmt = f'-{val:.1f}%'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02,
                    fmt, ha='center', va='bottom', fontsize=13, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('6-Coin Portfolio: Bayesian Original vs Grid Robust', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path3 = os.path.join(save_dir, '6coin_grid_portfolio_comparison.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3] 포트폴리오 비교: {path3}")


# ==========================================
# 메인
# ==========================================
def main():
    print("=" * 70)
    print("  6코인 그리드 분석 + 강건한 봉우리 선택")
    print("=" * 70)
    print(f"  그리드: MA {len(MA_GRID)}값 × SK {len(SK_GRID)}값 × SKs {len(SKS_GRID)}값 × SD {len(SD_GRID)}값")
    print(f"  = {len(MA_GRID)*len(SK_GRID)*len(SKS_GRID)*len(SD_GRID):,}개/방향 × 2방향 × 6코인")
    start_time = time.time()

    # 데이터 로드
    print("\n  --- 데이터 로드 ---")
    coin_data_cache = {}
    for symbol in sorted(NEW_CONFIGS.keys()):
        data = prepare_coin_data(symbol, silent=True)
        if data:
            coin_data_cache[symbol] = data
            print(f"  {symbol}: {data['days']}일")

    # 코인별 그리드 분석
    print("\n  --- 코인별 그리드 분석 ---")
    all_results = []
    for symbol in ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT', 'ADAUSDT']:
        config = NEW_CONFIGS[symbol]
        data = coin_data_cache.get(symbol)
        if data:
            result = analyze_coin(symbol, data, config)
            all_results.append(result)

    # Robust configs 정리
    print("\n" + "=" * 70)
    print("  Robust 파라미터 도출 결과")
    print("=" * 70)
    robust_configs = {}
    for res in all_results:
        sym = res['symbol']
        orig = res['original_config']
        rob = res['robust_config']
        robust_configs[sym] = rob

        print(f"\n  {sym} (priority: {orig['priority']})")
        for direction in ['long', 'short']:
            orig_params = f"MA={orig[f'{direction}_ma']}, SK={orig[f'{direction}_sk']}, SKs={orig[f'{direction}_sks']}, SD={orig[f'{direction}_sd']}"
            rob_params = f"MA={rob[f'{direction}_ma']}, SK={rob[f'{direction}_sk']}, SKs={rob[f'{direction}_sks']}, SD={rob[f'{direction}_sd']}"
            changed = orig_params != rob_params
            print(f"    {direction:>5}: {orig_params}")
            print(f"    {'→':>5}: {rob_params} {'← CHANGED' if changed else '(동일)'}")

    # 포트폴리오 비교
    print("\n  --- 포트폴리오 백테스트 ---")
    new_perf = run_portfolio(NEW_CONFIGS, coin_data_cache)
    robust_perf = run_portfolio(robust_configs, coin_data_cache)

    if new_perf and robust_perf:
        print(f"\n  {'항목':<15} {'Original':>15} {'Robust':>15} {'차이':>15}")
        print(f"  {'-'*60}")
        print(f"  {'CAGR':<15} {new_perf['cagr']:>14.1f}% {robust_perf['cagr']:>14.1f}% {robust_perf['cagr']-new_perf['cagr']:>+14.1f}%")
        print(f"  {'MDD':<15} {new_perf['mdd']:>14.1f}% {robust_perf['mdd']:>14.1f}% {robust_perf['mdd']-new_perf['mdd']:>+14.1f}%")
        print(f"  {'Sharpe':<15} {new_perf['sharpe']:>15.3f} {robust_perf['sharpe']:>15.3f} {robust_perf['sharpe']-new_perf['sharpe']:>+15.3f}")
        print(f"  {'최종 자산':<15} ${new_perf['final_equity']:>13,.0f} ${robust_perf['final_equity']:>13,.0f}")

    # 차트 생성
    print("\n  --- 차트 생성 ---")
    plot_grid_results(all_results, new_perf, robust_perf, SAVE_DIR)

    elapsed = time.time() - start_time
    print(f"\n  완료! ({elapsed/60:.1f}분)")


if __name__ == '__main__':
    main()
