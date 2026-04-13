"""
================================================================================
파라미터 강건성 검증: 봉우리(Plateau) vs 바늘(Spike) 분석
================================================================================
각 코인의 현재 파라미터 주변 ±N 범위를 그리드 탐색하여
현재 파라미터가 넓은 봉우리 위에 있는지, 바늘처럼 혼자 튀는지 확인

- 6코인 NEW: ±10% 범위, 최대 11포인트/파라미터
- 288코인 HYBRID: ±10% 범위, 최대 7포인트/파라미터
- Smoothed Sharpe 비교로 강건성 판정
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

# ============================================
# 6코인 NEW 파라미터
# ============================================
NEW_6COIN = {
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


# ============================================
# 288코인 HYBRID 설정 로드
# ============================================
def load_hybrid_configs():
    """binance_bot-5.py에서 288코인 HYBRID 설정 로드"""
    bot_path = os.path.join(SAVE_DIR, 'binance_bot-5.py')
    if not os.path.exists(bot_path):
        print(f"  ⚠ {bot_path} 없음")
        return {}
    with open(bot_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # SHORT_TRADING_CONFIGS
    s_start = content.index('SHORT_TRADING_CONFIGS = [')
    s_end = content.index(']', s_start) + 1
    ns = {}
    exec(content[s_start:s_end], {}, ns)
    short_list = ns['SHORT_TRADING_CONFIGS']

    # LONG_TRADING_CONFIGS
    l_start = content.index('LONG_TRADING_CONFIGS = [')
    l_end = content.index(']', l_start) + 1
    ns = {}
    exec(content[l_start:l_end], {}, ns)
    long_list = ns['LONG_TRADING_CONFIGS']

    # COIN_PRIORITY
    p_start = content.index('COIN_PRIORITY = {')
    p_end = content.index('}', p_start) + 1
    ns = {}
    exec(content[p_start:p_end], {}, ns)
    priorities = ns['COIN_PRIORITY']

    long_by_sym = {c['symbol']: c for c in long_list}

    configs = {}
    for sc in short_list:
        sym = sc['symbol']
        merged = {
            'priority': priorities.get(sym, 'short'),
            'short_ma': sc['ma_period'], 'short_sk': sc['stoch_k_period'],
            'short_sks': sc['stoch_k_smooth'], 'short_sd': sc['stoch_d_period'],
            'short_lev': sc['leverage'],
        }
        lc = long_by_sym.get(sym)
        if lc:
            merged['long_ma'] = lc['long_ma']
            merged['long_sk'] = lc['long_sk']
            merged['long_sks'] = lc['long_sks']
            merged['long_sd'] = lc['long_sd']
            merged['long_lev'] = lc['long_lev']
        configs[sym] = merged

    return configs


# ============================================
# 그리드 범위 생성
# ============================================
def make_grid(center, pct=0.10, min_radius=3, max_points=11, min_val=1):
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
    # 초과 시 trim
    while len(grid) > max_points + 1:
        grid.pop()
    return grid


# ============================================
# 지표 사전계산 (소규모 범위)
# ============================================
def precompute_indicators_range(data, ma_range, sk_range, sks_range, sd_range):
    """지정 범위의 MA / 스토캐스틱 지표 사전계산"""
    df_4h = data['df_4h'].sort_values('timestamp').reset_index(drop=True)
    df_daily = data['df_daily'].sort_values('timestamp').reset_index(drop=True)
    n = len(df_4h)

    opens = df_4h['open'].values.astype(np.float64)
    highs = df_4h['high'].values.astype(np.float64)
    lows = df_4h['low'].values.astype(np.float64)
    closes = df_4h['close'].values.astype(np.float64)
    dates_4h = df_4h['timestamp'].dt.date.values

    # 4H MA (shifted by 1)
    prev_mas = {}
    close_s = pd.Series(closes)
    for ma in ma_range:
        arr = close_s.rolling(int(ma)).mean().values
        shifted = np.empty(n)
        shifted[0] = np.nan
        shifted[1:] = arr[:-1]
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
    for sk in sk_range:
        lo_r = d_low.rolling(int(sk)).min().values
        hi_r = d_high.rolling(int(sk)).max().values
        denom = hi_r - lo_r
        denom[denom == 0] = np.nan
        fast_ks[sk] = (d_close - lo_r) / denom * 100

    prev_stochs = {}
    for sk in sk_range:
        fk = fast_ks[sk]
        fk_s = pd.Series(fk)
        for sks in sks_range:
            slow_k = fk_s.rolling(int(sks)).mean().values
            sk_s = pd.Series(slow_k)
            for sd in sd_range:
                slow_d = sk_s.rolling(int(sd)).mean().values
                # Shift daily by 1
                prev_sk_d = np.empty(len(daily_dates))
                prev_sd_d = np.empty(len(daily_dates))
                prev_sk_d[0] = np.nan; prev_sd_d[0] = np.nan
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
                # Shift 4H by 1
                prev_k = np.empty(n); prev_d = np.empty(n)
                prev_k[0] = np.nan; prev_d[0] = np.nan
                prev_k[1:] = sk_4h[:-1]
                prev_d[1:] = sd_4h[:-1]
                prev_stochs[(sk, sks, sd)] = (prev_k, prev_d)

    return opens, highs, lows, closes, n, prev_mas, prev_stochs


# ============================================
# 고속 백테스트 (grid_analysis_6coin.py 동일)
# ============================================
def fast_backtest(opens, highs, lows, closes, n,
                  prev_ma_long, prev_sk_long, prev_sd_long,
                  prev_ma_short, prev_sk_short, prev_sd_short,
                  long_lev, short_lev, priority_is_long):
    FEE_SLIP = FEE_RATE + SLIPPAGE_PCT
    FUND = FUNDING_PER_4H
    equity = 10000.0
    side = 0; ep = 0.0; em = 0.0; cf = 0.0; lev = 0
    peak = equity; max_dd = 0.0; prev_eq = equity
    sr = 0.0; sr2 = 0.0; nr = 0; start_i = -1

    for i in range(1, n):
        pml = prev_ma_long[i]; pms = prev_ma_short[i]
        pkl = prev_sk_long[i]; pdl = prev_sd_long[i]
        pks = prev_sk_short[i]; pds = prev_sd_short[i]

        if pml != pml or pms != pms or pkl != pkl or pdl != pdl or pks != pks or pds != pds:
            if side != 0:
                cp = closes[i]; pr = cp / ep - 1
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

        if start_i < 0: start_i = i
        op = opens[i]
        ls = (op > pml) and (pkl > pdl)
        ss = (op < pms) and (pks < pds)

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

        if side != 0 and ep > 0:
            cp = closes[i]; pr = cp / ep - 1
            if side == -1: pr = -pr
            cf += em * lev * FUND
            ceq = em + em * pr * lev - cf
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
        if ceq <= 0: break

    if nr < 100 or start_i < 0:
        return (0, 0, -100)
    final_eq = prev_eq
    days = (n - start_i) * 4 / 24
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


# ============================================
# 이웃 그리드 탐색 + 강건성 계산
# ============================================
def neighborhood_check(data, config, direction, max_points=11, pct=0.10):
    """
    한 방향의 파라미터 이웃 탐색.
    direction: 'short' or 'long' (탐색할 방향)
    반대 방향은 기존값 고정.
    """
    priority_is_long = (config['priority'] == 'long')

    # 탐색 방향 파라미터
    s_ma = config[f'{direction}_ma']
    s_sk = config[f'{direction}_sk']
    s_sks = config[f'{direction}_sks']
    s_sd = config[f'{direction}_sd']
    s_lev = config[f'{direction}_lev']

    # 고정 방향
    fix_dir = 'long' if direction == 'short' else 'short'
    f_ma = config[f'{fix_dir}_ma']
    f_sk = config[f'{fix_dir}_sk']
    f_sks = config[f'{fix_dir}_sks']
    f_sd = config[f'{fix_dir}_sd']
    f_lev = config[f'{fix_dir}_lev']

    # 탐색 범위 생성
    ma_range = make_grid(s_ma, pct=pct, max_points=max_points, min_val=5)
    sk_range = make_grid(s_sk, pct=pct, max_points=max_points, min_val=2)
    sks_range = make_grid(s_sks, pct=pct, max_points=max_points, min_val=2)
    sd_range = make_grid(s_sd, pct=pct, max_points=max_points, min_val=2)

    total_combos = len(ma_range) * len(sk_range) * len(sks_range) * len(sd_range)

    # 고정 방향 + 탐색 방향 지표 사전계산
    all_ma = sorted(set(ma_range + [f_ma]))
    all_sk = sorted(set(sk_range + [f_sk]))
    all_sks = sorted(set(sks_range + [f_sks]))
    all_sd = sorted(set(sd_range + [f_sd]))

    opens, highs, lows, closes, n, prev_mas, prev_stochs = \
        precompute_indicators_range(data, all_ma, all_sk, all_sks, all_sd)

    # 고정 방향 지표
    fixed_prev_ma = prev_mas[f_ma]
    fixed_stoch = prev_stochs.get((f_sk, f_sks, f_sd))
    if fixed_stoch is None:
        return None
    fixed_prev_sk, fixed_prev_sd = fixed_stoch

    search_is_long = (direction == 'long')

    # 4D 그리드 탐색
    n_ma = len(ma_range); n_sk = len(sk_range)
    n_sks = len(sks_range); n_sd = len(sd_range)
    sharpe_grid = np.full((n_ma, n_sk, n_sks, n_sd), -999.0)
    cagr_grid = np.full((n_ma, n_sk, n_sks, n_sd), -999.0)
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
                        result = fast_backtest(
                            opens, highs, lows, closes, n,
                            search_ma, s_skv, s_sdv,
                            fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
                            s_lev, f_lev, priority_is_long)
                    else:
                        result = fast_backtest(
                            opens, highs, lows, closes, n,
                            fixed_prev_ma, fixed_prev_sk, fixed_prev_sd,
                            search_ma, s_skv, s_sdv,
                            f_lev, s_lev, priority_is_long)

                    sharpe_grid[i_ma, i_sk, i_sks, i_sd] = result[0]
                    cagr_grid[i_ma, i_sk, i_sks, i_sd] = result[1]

                    if ma == s_ma and sk == s_sk and sks == s_sks and sd == s_sd:
                        center_idx = (i_ma, i_sk, i_sks, i_sd)

    # 통계 계산
    valid = sharpe_grid > -900
    valid_sharpes = sharpe_grid[valid]
    n_valid = len(valid_sharpes)

    if center_idx is None or n_valid < 10:
        return None

    center_sharpe = float(sharpe_grid[center_idx])
    center_cagr = float(cagr_grid[center_idx])

    # Smoothing (3×3×3×3 uniform filter)
    sg = sharpe_grid.copy()
    sg[~valid] = 0
    cnt = valid.astype(float)
    s_sum = uniform_filter(sg, size=3, mode='constant', cval=0)
    c_sum = uniform_filter(cnt, size=3, mode='constant', cval=0)
    c_sum[c_sum == 0] = 1
    smoothed = s_sum / c_sum
    smoothed[~valid] = -999

    smoothed_at_center = float(smoothed[center_idx])

    # Metrics
    neighbor_mean = float(np.mean(valid_sharpes))
    neighbor_median = float(np.median(valid_sharpes))
    neighbor_std = float(np.std(valid_sharpes))
    neighbor_max = float(np.max(valid_sharpes))
    pctile = float((valid_sharpes < center_sharpe).sum() / n_valid * 100)

    # Smoothed ratio: smoothed / raw (closer to 1.0 = more robust)
    if center_sharpe > 0:
        smooth_ratio = smoothed_at_center / center_sharpe
    else:
        smooth_ratio = 1.0 if smoothed_at_center <= 0 else 0.0

    # Mean ratio
    if neighbor_mean > 0 and center_sharpe > 0:
        mean_ratio = neighbor_mean / center_sharpe
    else:
        mean_ratio = 1.0

    # Verdict
    if smooth_ratio >= 0.85:
        verdict = '강건'
    elif smooth_ratio >= 0.65:
        verdict = '양호'
    elif smooth_ratio >= 0.45:
        verdict = '주의'
    else:
        verdict = '바늘'

    # 2D heatmap data (MA × SK, max over SKs/SD) for chart
    heatmap = np.nanmax(sharpe_grid, axis=(2, 3))
    heatmap[heatmap < -900] = np.nan

    return {
        'direction': direction,
        'center_params': (s_ma, s_sk, s_sks, s_sd),
        'center_sharpe': center_sharpe,
        'center_cagr': center_cagr,
        'smoothed_sharpe': smoothed_at_center,
        'smooth_ratio': smooth_ratio,
        'neighbor_mean': neighbor_mean,
        'neighbor_median': neighbor_median,
        'neighbor_std': neighbor_std,
        'neighbor_max': neighbor_max,
        'mean_ratio': mean_ratio,
        'percentile': pctile,
        'verdict': verdict,
        'n_combos': n_valid,
        'heatmap': heatmap,
        'ma_range': ma_range,
        'sk_range': sk_range,
        'grid_shape': (n_ma, n_sk, n_sks, n_sd),
    }


# ============================================
# 코인별 검증
# ============================================
def check_coin(symbol, data, config, max_points=11, pct=0.10):
    """코인 하나의 양방향 강건성 검증"""
    results = {}

    # Short direction
    r_short = neighborhood_check(data, config, 'short', max_points=max_points, pct=pct)
    if r_short:
        results['short'] = r_short

    # Long direction (있는 경우만)
    if 'long_ma' in config:
        r_long = neighborhood_check(data, config, 'long', max_points=max_points, pct=pct)
        if r_long:
            results['long'] = r_long

    return results


# ============================================
# 차트 생성
# ============================================
def plot_6coin_heatmaps(all_results, save_dir):
    """6코인 이웃 Sharpe 히트맵 (MA × SK)"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for idx, (symbol, res) in enumerate(all_results.items()):
        ax = axes[idx // 3][idx % 3]
        # Primary direction
        config = res.get('config', {})
        pri = config.get('priority', 'short')
        direction = pri  # primary direction

        r = res.get(direction)
        if r is None:
            ax.text(0.5, 0.5, f'{symbol}\n데이터 없음', ha='center', va='center', fontsize=14)
            ax.set_title(symbol)
            continue

        heatmap = r['heatmap']
        ma_range = r['ma_range']
        sk_range = r['sk_range']

        im = ax.imshow(heatmap.T, aspect='auto', origin='lower', cmap='RdYlGn',
                       extent=[ma_range[0], ma_range[-1], sk_range[0], sk_range[-1]],
                       interpolation='nearest')

        # Mark center (current params)
        c_ma, c_sk = r['center_params'][0], r['center_params'][1]
        ax.plot(c_ma, c_sk, 'k*', markersize=18, markeredgewidth=1.5, label=f'현재 ({r["verdict"]})')

        # Smoothed peak
        ax.set_xlabel('MA Period', fontsize=10)
        ax.set_ylabel('Stoch K Period', fontsize=10)

        verdict_color = {'강건': '#2196F3', '양호': '#4CAF50', '주의': '#FF9800', '바늘': '#F44336'}
        vc = verdict_color.get(r['verdict'], 'gray')
        ax.set_title(f"{symbol} [{direction.upper()} primary]\n"
                     f"Sharpe={r['center_sharpe']:.2f} | Smooth={r['smoothed_sharpe']:.2f} | "
                     f"Ratio={r['smooth_ratio']:.2f} → {r['verdict']}",
                     fontweight='bold', fontsize=10, color=vc)
        ax.legend(fontsize=8, loc='upper right')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('6코인 NEW: 파라미터 이웃 Sharpe 지형 (Primary Direction)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'param_robustness_6coin_heatmaps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Chart 1] {path}")


def plot_robustness_summary(all_6coin, all_288coin, save_dir):
    """강건성 요약 차트"""
    fig = plt.figure(figsize=(20, 14))

    # ===== 1. 6코인 상세 테이블 =====
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')

    headers = ['Coin', 'Dir', 'Sharpe', 'Smoothed', 'Ratio', 'Pctile', 'Mean', 'Verdict']
    rows = []
    row_colors = []
    color_map = {'강건': '#C8E6C9', '양호': '#FFF9C4', '주의': '#FFE0B2', '바늘': '#FFCDD2'}

    for symbol, res in all_6coin.items():
        for direction in ['short', 'long']:
            r = res.get(direction)
            if r is None:
                continue
            is_primary = (res['config']['priority'] == direction)
            dir_label = f"{'★' if is_primary else ' '}{direction}"
            rows.append([
                symbol, dir_label,
                f"{r['center_sharpe']:.2f}", f"{r['smoothed_sharpe']:.2f}",
                f"{r['smooth_ratio']:.2f}", f"{r['percentile']:.0f}%",
                f"{r['neighbor_mean']:.2f}", r['verdict']
            ])
            row_colors.append(color_map.get(r['verdict'], 'white'))

    if rows:
        table = ax1.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        for j in range(len(headers)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i, color in enumerate(row_colors):
            for j in range(len(headers)):
                table[i + 1, j].set_facecolor(color)
    ax1.set_title('6코인 NEW: 방향별 강건성 상세', fontweight='bold', fontsize=12, pad=15)

    # ===== 2. 6코인 Smooth Ratio 바 차트 =====
    ax2 = fig.add_subplot(2, 2, 2)
    labels_6 = []
    ratios_6 = []
    colors_6 = []
    for symbol, res in all_6coin.items():
        for direction in ['short', 'long']:
            r = res.get(direction)
            if r is None:
                continue
            is_primary = (res['config']['priority'] == direction)
            labels_6.append(f"{symbol}\n{'★' if is_primary else ''}{direction[0].upper()}")
            ratios_6.append(r['smooth_ratio'])
            vc = {'강건': '#4CAF50', '양호': '#8BC34A', '주의': '#FF9800', '바늘': '#F44336'}
            colors_6.append(vc.get(r['verdict'], 'gray'))

    if labels_6:
        bars = ax2.barh(range(len(labels_6)), ratios_6, color=colors_6, alpha=0.85, height=0.7)
        ax2.set_yticks(range(len(labels_6)))
        ax2.set_yticklabels(labels_6, fontsize=8)
        ax2.axvline(x=0.85, color='green', linestyle='--', alpha=0.7, label='강건 (≥0.85)')
        ax2.axvline(x=0.65, color='orange', linestyle='--', alpha=0.7, label='양호 (≥0.65)')
        ax2.axvline(x=0.45, color='red', linestyle='--', alpha=0.7, label='주의 (≥0.45)')
        for i, (bar, ratio) in enumerate(zip(bars, ratios_6)):
            ax2.text(ratio + 0.01, i, f'{ratio:.2f}', va='center', fontsize=8, fontweight='bold')
        ax2.set_xlabel('Smooth Ratio (Smoothed/Raw Sharpe)', fontsize=10)
        ax2.set_title('6코인 Smooth Ratio (1.0에 가까울수록 강건)', fontweight='bold', fontsize=11)
        ax2.legend(fontsize=8, loc='lower right')
        ax2.set_xlim(0, max(ratios_6) * 1.15 if ratios_6 else 1.2)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

    # ===== 3. 288코인 Smooth Ratio 히스토그램 =====
    ax3 = fig.add_subplot(2, 2, 3)
    if all_288coin:
        ratios_288_short = []
        ratios_288_long = []
        verdicts_288 = {'강건': 0, '양호': 0, '주의': 0, '바늘': 0}

        for symbol, res in all_288coin.items():
            for direction in ['short', 'long']:
                r = res.get(direction)
                if r is None:
                    continue
                if direction == 'short':
                    ratios_288_short.append(r['smooth_ratio'])
                else:
                    ratios_288_long.append(r['smooth_ratio'])
                verdicts_288[r['verdict']] = verdicts_288.get(r['verdict'], 0) + 1

        all_ratios = ratios_288_short + ratios_288_long
        if all_ratios:
            ax3.hist(all_ratios, bins=30, color='#42A5F5', alpha=0.7, edgecolor='white')
            ax3.axvline(x=0.85, color='green', linestyle='--', linewidth=2, label='강건 (≥0.85)')
            ax3.axvline(x=0.65, color='orange', linestyle='--', linewidth=2, label='양호 (≥0.65)')
            ax3.axvline(x=0.45, color='red', linestyle='--', linewidth=2, label='주의 (≥0.45)')
            ax3.set_xlabel('Smooth Ratio', fontsize=10)
            ax3.set_ylabel('Count', fontsize=10)
            ax3.set_title(f'288코인 HYBRID: Smooth Ratio 분포 (N={len(all_ratios)})',
                         fontweight='bold', fontsize=11)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, '288코인 데이터 없음', ha='center', va='center', fontsize=14)
        ax3.set_title('288코인 HYBRID: Smooth Ratio 분포', fontweight='bold', fontsize=11)

    # ===== 4. 288코인 판정 파이차트 =====
    ax4 = fig.add_subplot(2, 2, 4)
    if all_288coin and any(verdicts_288.values()):
        labels_pie = []; sizes_pie = []; colors_pie = []
        pie_colors = {'강건': '#4CAF50', '양호': '#8BC34A', '주의': '#FF9800', '바늘': '#F44336'}
        for v in ['강건', '양호', '주의', '바늘']:
            if verdicts_288.get(v, 0) > 0:
                labels_pie.append(f"{v}\n({verdicts_288[v]})")
                sizes_pie.append(verdicts_288[v])
                colors_pie.append(pie_colors[v])
        ax4.pie(sizes_pie, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        total_checks = sum(verdicts_288.values())
        ax4.set_title(f'288코인 HYBRID: 강건성 판정 분포 (총 {total_checks}건)',
                     fontweight='bold', fontsize=11)
    else:
        ax4.text(0.5, 0.5, '288코인 데이터 없음', ha='center', va='center', fontsize=14)
        ax4.set_title('288코인 HYBRID: 강건성 판정 분포', fontweight='bold', fontsize=11)

    plt.suptitle('파라미터 강건성 검증: 봉우리(Plateau) vs 바늘(Spike)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'param_robustness_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Chart 2] {path}")


def plot_288coin_worst(all_288coin, save_dir, top_n=30):
    """288코인 중 바늘/주의 코인 상세"""
    if not all_288coin:
        return

    # Collect all results
    entries = []
    for symbol, res in all_288coin.items():
        for direction in ['short', 'long']:
            r = res.get(direction)
            if r is None:
                continue
            is_primary = (res['config']['priority'] == direction)
            entries.append({
                'symbol': symbol, 'direction': direction,
                'is_primary': is_primary,
                'smooth_ratio': r['smooth_ratio'],
                'center_sharpe': r['center_sharpe'],
                'smoothed_sharpe': r['smoothed_sharpe'],
                'verdict': r['verdict'],
                'percentile': r['percentile'],
            })

    if not entries:
        return

    df = pd.DataFrame(entries)
    df_sorted = df.sort_values('smooth_ratio', ascending=True).head(top_n)

    fig, ax = plt.subplots(figsize=(16, max(8, top_n * 0.35)))
    ax.axis('off')

    headers = ['#', 'Coin', 'Dir', 'Primary', 'Sharpe', 'Smoothed', 'Ratio', 'Pctile', 'Verdict']
    rows = []
    row_colors = []
    color_map = {'강건': '#C8E6C9', '양호': '#FFF9C4', '주의': '#FFE0B2', '바늘': '#FFCDD2'}

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        rows.append([
            i + 1, row['symbol'], row['direction'].upper(),
            '★' if row['is_primary'] else '',
            f"{row['center_sharpe']:.2f}", f"{row['smoothed_sharpe']:.2f}",
            f"{row['smooth_ratio']:.2f}", f"{row['percentile']:.0f}%",
            row['verdict']
        ])
        row_colors.append(color_map.get(row['verdict'], 'white'))

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#E53935')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i, color in enumerate(row_colors):
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title(f'288코인 HYBRID: 강건성 하위 {top_n}개 (바늘 위험도 높은 순)',
                 fontweight='bold', fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, 'param_robustness_288coin_worst.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Chart 3] {path}")


# ============================================
# 메인
# ============================================
def main():
    print("=" * 70)
    print("  파라미터 강건성 검증: 봉우리(Plateau) vs 바늘(Spike)")
    print("=" * 70)
    total_start = time.time()

    # ===== Phase 1: 6코인 NEW =====
    print("\n" + "=" * 70)
    print("  Phase 1: 6코인 NEW 파라미터 검증 (±10%, max 11 points)")
    print("=" * 70)

    all_6coin = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT', 'ADAUSDT']:
        config = NEW_6COIN[symbol]
        print(f"\n  --- {symbol} (priority: {config['priority']}) ---")
        t0 = time.time()

        data = prepare_coin_data(symbol, silent=True)
        if not data:
            print(f"    데이터 없음, 건너뜀")
            continue

        results = check_coin(symbol, data, config, max_points=11, pct=0.10)
        results['config'] = config
        all_6coin[symbol] = results

        for direction in ['short', 'long']:
            r = results.get(direction)
            if r is None:
                continue
            is_primary = '★' if config['priority'] == direction else ' '
            print(f"    {is_primary}{direction:>5}: Sharpe={r['center_sharpe']:.2f} | "
                  f"Smoothed={r['smoothed_sharpe']:.2f} | Ratio={r['smooth_ratio']:.2f} | "
                  f"Pctile={r['percentile']:.0f}% | Mean={r['neighbor_mean']:.2f} | "
                  f"{r['n_combos']:,}개 → [{r['verdict']}]")

        print(f"    소요: {time.time()-t0:.1f}초")

    # 6코인 요약
    print("\n" + "-" * 50)
    print("  6코인 요약:")
    for v in ['강건', '양호', '주의', '바늘']:
        cnt = sum(1 for res in all_6coin.values()
                  for d in ['short', 'long']
                  if d in res and res[d]['verdict'] == v)
        if cnt > 0:
            print(f"    {v}: {cnt}건")

    # ===== Phase 2: 288코인 HYBRID =====
    print("\n" + "=" * 70)
    print("  Phase 2: 288코인 HYBRID 파라미터 검증 (±10%, max 7 points)")
    print("=" * 70)

    hybrid_configs = load_hybrid_configs()
    print(f"  설정 로드: {len(hybrid_configs)}개 코인")

    all_288coin = {}
    total_coins = len(hybrid_configs)
    done = 0
    failed = 0

    for symbol, config in sorted(hybrid_configs.items()):
        done += 1
        if done % 20 == 1 or done <= 3:
            print(f"\n  [{done}/{total_coins}] {symbol} (priority: {config['priority']})")

        t0 = time.time()
        data = prepare_coin_data(symbol, silent=True)
        if not data:
            failed += 1
            continue

        results = check_coin(symbol, data, config, max_points=7, pct=0.10)
        results['config'] = config
        all_288coin[symbol] = results

        elapsed = time.time() - t0
        if done % 20 == 1 or done <= 3:
            for direction in ['short', 'long']:
                r = results.get(direction)
                if r:
                    is_primary = '★' if config['priority'] == direction else ' '
                    print(f"    {is_primary}{direction:>5}: Sharpe={r['center_sharpe']:.2f} | "
                          f"Ratio={r['smooth_ratio']:.2f} → [{r['verdict']}] ({elapsed:.1f}초)")

        if done % 50 == 0:
            elapsed_total = time.time() - total_start
            eta = elapsed_total / done * (total_coins - done) / 60
            print(f"  --- 진행: {done}/{total_coins} ({done/total_coins*100:.0f}%), "
                  f"실패: {failed}, 예상 잔여: {eta:.0f}분 ---")

    # 288코인 요약
    print(f"\n  288코인 완료: 성공 {len(all_288coin)}개, 실패 {failed}개")
    verdicts = {'강건': 0, '양호': 0, '주의': 0, '바늘': 0}
    for res in all_288coin.values():
        for d in ['short', 'long']:
            r = res.get(d)
            if r:
                verdicts[r['verdict']] = verdicts.get(r['verdict'], 0) + 1
    print("  판정 분포:")
    total_v = sum(verdicts.values())
    for v in ['강건', '양호', '주의', '바늘']:
        pct = verdicts[v] / total_v * 100 if total_v > 0 else 0
        print(f"    {v}: {verdicts[v]}건 ({pct:.1f}%)")

    # ===== 차트 생성 =====
    print("\n" + "=" * 70)
    print("  차트 생성")
    print("=" * 70)

    if all_6coin:
        plot_6coin_heatmaps(all_6coin, SAVE_DIR)
    plot_robustness_summary(all_6coin, all_288coin, SAVE_DIR)
    if all_288coin:
        plot_288coin_worst(all_288coin, SAVE_DIR, top_n=30)

    # ===== CSV 저장 =====
    rows = []
    for label, results_dict in [('6coin_NEW', all_6coin), ('288coin_HYBRID', all_288coin)]:
        for symbol, res in results_dict.items():
            for direction in ['short', 'long']:
                r = res.get(direction)
                if r is None:
                    continue
                rows.append({
                    'group': label,
                    'symbol': symbol,
                    'direction': direction,
                    'is_primary': res['config']['priority'] == direction,
                    'MA': r['center_params'][0],
                    'SK': r['center_params'][1],
                    'SKs': r['center_params'][2],
                    'SD': r['center_params'][3],
                    'center_sharpe': r['center_sharpe'],
                    'smoothed_sharpe': r['smoothed_sharpe'],
                    'smooth_ratio': r['smooth_ratio'],
                    'neighbor_mean': r['neighbor_mean'],
                    'neighbor_median': r['neighbor_median'],
                    'neighbor_std': r['neighbor_std'],
                    'percentile': r['percentile'],
                    'n_combos': r['n_combos'],
                    'verdict': r['verdict'],
                })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(SAVE_DIR, 'param_robustness_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  [CSV] {csv_path}")

    elapsed = time.time() - total_start
    print(f"\n  완료! 총 소요: {elapsed/60:.1f}분")


if __name__ == '__main__':
    main()
