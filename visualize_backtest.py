"""
4H 백테스트 시각화 분석
- 에쿼티 커브 (로그 스케일)
- 드로다운 차트
- 월별 수익률 히트맵
- 활성 포지션 수 추이
- 연간 수익률 바차트
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

# 백테스트 함수 임포트
from backtest_bots import (
    prepare_coin_data, prepare_signals, calculate_stochastic,
    load_binance_configs, calculate_portfolio_performance,
    run_bitget_portfolio, run_binance_portfolio,
    BITGET_CONFIGS, BITGET_CAPITAL, BINANCE_CAPITAL,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H
)

SAVE_DIR = os.path.expanduser("~/Downloads")


def build_equity_df(perf):
    """perf dict에서 에쿼티 DataFrame 생성"""
    df = pd.DataFrame({
        'timestamp': perf['timestamps'],
        'equity': perf['equity_curve']
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df


def compute_drawdown(equity_series):
    """드로다운 시리즈 계산"""
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    return dd


def compute_monthly_returns(equity_df):
    """월별 수익률 계산"""
    monthly = equity_df['equity'].resample('ME').last().dropna()
    returns = monthly.pct_change().dropna() * 100
    return returns


def plot_equity_and_drawdown(bitget_df, binance_df, save_path):
    """에쿼티 커브 + 드로다운 차트"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                              sharex=True, gridspec_kw={'hspace': 0.05})

    # 에쿼티 커브 (로그 스케일)
    ax1 = axes[0]
    ax1.semilogy(bitget_df.index, bitget_df['equity'], color='#2196F3', linewidth=1.2,
                 label=f'Bitget 6coins ($10K)', alpha=0.9)
    ax1.semilogy(binance_df.index, binance_df['equity'], color='#FF5722', linewidth=1.2,
                 label=f'Binance Short+Long ($100K)', alpha=0.9)

    ax1.set_ylabel('Portfolio Equity (log)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.set_title('4H Backtest - Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # 드로다운
    ax2 = axes[1]
    dd_bitget = compute_drawdown(bitget_df['equity'])
    dd_binance = compute_drawdown(binance_df['equity'])
    ax2.fill_between(dd_bitget.index, dd_bitget.values, 0, color='#2196F3', alpha=0.3, label='Bitget DD')
    ax2.fill_between(dd_binance.index, dd_binance.values, 0, color='#FF5722', alpha=0.3, label='Binance DD')
    ax2.plot(dd_bitget.index, dd_bitget.values, color='#2196F3', linewidth=0.5, alpha=0.7)
    ax2.plot(dd_binance.index, dd_binance.values, color='#FF5722', linewidth=0.5, alpha=0.7)

    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [1] 에쿼티+드로다운: {save_path}")


def plot_monthly_heatmap(equity_df, title, save_path):
    """월별 수익률 히트맵"""
    monthly_ret = compute_monthly_returns(equity_df)
    if len(monthly_ret) < 2:
        return

    # 연/월 매트릭스
    ret_df = pd.DataFrame({'return': monthly_ret})
    ret_df['year'] = ret_df.index.year
    ret_df['month'] = ret_df.index.month
    pivot = ret_df.pivot_table(values='return', index='year', columns='month', aggfunc='sum')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    # 연간 합계 추가
    pivot['Year Total'] = pivot.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.7 + 1)))

    # 색상 범위 설정
    vmax = min(pivot.iloc[:, :-1].max().max(), 200)
    vmin = max(pivot.iloc[:, :-1].min().min(), -100)
    if vmin >= 0:
        vmin = -1
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax.imshow(pivot.iloc[:, :-1].values, cmap='RdYlGn', aspect='auto', norm=norm)

    ax.set_xticks(range(len(pivot.columns) - 1))
    ax.set_xticklabels(pivot.columns[:-1], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # 셀에 값 표시
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns) - 1):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8, color=color)

    # 연간 합계 텍스트
    for i, yr_total in enumerate(pivot['Year Total']):
        ax.text(len(pivot.columns) - 1.5, i, f'  Year: {yr_total:.0f}%',
                ha='left', va='center', fontsize=9, fontweight='bold',
                color='green' if yr_total > 0 else 'red')

    ax.set_title(f'{title} - Monthly Returns (%)', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Return %')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [2] 월별 히트맵: {save_path}")


def plot_yearly_comparison(bitget_df, binance_df, save_path):
    """연간 수익률 비교 바차트"""
    def get_yearly_returns(equity_df):
        yearly = equity_df['equity'].resample('YE').last().dropna()
        returns = yearly.pct_change().dropna() * 100
        # 첫해: 초기자본 대비
        first_year_end = yearly.iloc[0]
        first_equity = equity_df['equity'].iloc[0]
        first_ret = (first_year_end / first_equity - 1) * 100
        returns = pd.concat([pd.Series({yearly.index[0]: first_ret}), returns])
        return returns

    bitget_yr = get_yearly_returns(bitget_df)
    binance_yr = get_yearly_returns(binance_df)

    all_years = sorted(set(bitget_yr.index.year) | set(binance_yr.index.year))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(all_years))
    width = 0.35

    bitget_vals = [bitget_yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]
    binance_vals = [binance_yr.get(pd.Timestamp(f'{y}-12-31'), 0) for y in all_years]

    bars1 = ax.bar(x - width/2, bitget_vals, width, label='Bitget 6coins', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, binance_vals, width, label='Binance Short+Long', color='#FF5722', alpha=0.8)

    # 값 표시
    for bar in bars1:
        h = bar.get_height()
        if h != 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 5),
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h != 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + max(abs(h)*0.02, 5),
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_years, fontsize=11)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.set_title('4H Backtest - Annual Returns Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [3] 연간 수익률: {save_path}")


def plot_rolling_stats(bitget_df, binance_df, save_path):
    """롤링 샤프 비율 + 롤링 변동성"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'hspace': 0.1})

    window = 6 * 90  # 약 90일 (4H 봉 기준) - 1H에도 비슷하게 동작

    for label, eq_df, color in [
        ('Bitget', bitget_df, '#2196F3'),
        ('Binance', binance_df, '#FF5722'),
    ]:
        returns = eq_df['equity'].pct_change().dropna()

        # 롤링 샤프 (90일 윈도우, 연환산)
        rolling_mean = returns.rolling(window=window, min_periods=window//2).mean()
        rolling_std = returns.rolling(window=window, min_periods=window//2).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(365 * 6)
        rolling_sharpe = rolling_sharpe.clip(-5, 10)  # 극단값 제한

        axes[0].plot(rolling_sharpe.index, rolling_sharpe.values, color=color,
                     linewidth=1, label=label, alpha=0.8)

        # 롤링 변동성 (연환산)
        rolling_vol = rolling_std * np.sqrt(365 * 6) * 100
        axes[1].plot(rolling_vol.index, rolling_vol.values, color=color,
                     linewidth=1, label=label, alpha=0.8)

    axes[0].set_ylabel('Rolling Sharpe (90d)', fontsize=12)
    axes[0].set_title('4H Backtest - Rolling Risk Metrics', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].axhline(y=1, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[0].axhline(y=2, color='green', linewidth=0.5, linestyle='--', alpha=0.3)

    axes[1].set_ylabel('Rolling Volatility (90d, ann.)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [4] 롤링 리스크: {save_path}")


def plot_summary_dashboard(bitget_perf, binance_perf, save_path):
    """요약 대시보드"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CAGR 비교
    ax = axes[0]
    names = ['Bitget\n6coins', 'Binance\nShort+Long']
    cagrs = [bitget_perf['cagr'], binance_perf['cagr']]
    bars = ax.bar(names, cagrs, color=['#2196F3', '#FF5722'], alpha=0.8, width=0.5)
    for bar, val in zip(bars, cagrs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title('CAGR (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # MDD 비교
    ax = axes[1]
    mdds = [abs(bitget_perf['mdd']), abs(binance_perf['mdd'])]
    bars = ax.bar(names, mdds, color=['#2196F3', '#FF5722'], alpha=0.8, width=0.5)
    for bar, val in zip(bars, mdds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'-{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title('Max Drawdown (%)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Sharpe 비교
    ax = axes[2]
    sharpes = [bitget_perf['sharpe'], binance_perf['sharpe']]
    bars = ax.bar(names, sharpes, color=['#2196F3', '#FF5722'], alpha=0.8, width=0.5)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('4H Portfolio Backtest Summary', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [5] 요약 대시보드: {save_path}")


def main():
    print("=" * 70)
    print("  4H 백테스트 시각화 분석")
    print("=" * 70)
    start_time = datetime.now()

    # 데이터 로드
    print("\n  --- 데이터 로드 ---")
    bitget_cache = {}
    for config in BITGET_CONFIGS:
        symbol = config['symbol']
        data = prepare_coin_data(symbol, silent=True)
        if data:
            bitget_cache[symbol] = data
    print(f"  Bitget: {len(bitget_cache)} 코인")

    short_configs, long_configs = load_binance_configs()
    binance_cache = {}
    all_symbols = set(c['symbol'] for c in short_configs) | set(c['symbol'] for c in long_configs)
    for idx, symbol in enumerate(sorted(all_symbols), 1):
        if idx % 100 == 0:
            print(f"  Binance 진행: {idx}/{len(all_symbols)}...")
        data = prepare_coin_data(symbol, silent=True)
        if data:
            binance_cache[symbol] = data
    print(f"  Binance: {len(binance_cache)} 코인")

    # 공통 end_date
    all_4h_ends = [d['df_4h']['timestamp'].max() for d in list(bitget_cache.values()) + list(binance_cache.values())]
    end_date = min(all_4h_ends)
    print(f"  공통 종료일: {end_date}")

    # 4H 백테스트 실행
    print("\n  --- 4H 백테스트 실행 ---")
    print("  Bitget...")
    bitget_perf = run_bitget_portfolio(use_1h=False, coin_data_cache=bitget_cache, end_date=end_date)
    print("  Binance...")
    binance_perf = run_binance_portfolio(use_1h=False, short_configs=short_configs, long_configs=long_configs,
                                          coin_data_cache=binance_cache, end_date=end_date)

    if not bitget_perf or not binance_perf:
        print("  백테스트 실패")
        return

    print(f"\n  Bitget: CAGR={bitget_perf['cagr']:.1f}%, MDD={bitget_perf['mdd']:.1f}%, Sharpe={bitget_perf['sharpe']:.3f}")
    print(f"  Binance: CAGR={binance_perf['cagr']:.1f}%, MDD={binance_perf['mdd']:.1f}%, Sharpe={binance_perf['sharpe']:.3f}")

    # DataFrame 생성
    bitget_df = build_equity_df(bitget_perf)
    binance_df = build_equity_df(binance_perf)

    # 차트 생성
    print("\n  --- 차트 생성 ---")
    date_tag = datetime.now().strftime('%Y%m%d')

    plot_equity_and_drawdown(
        bitget_df, binance_df,
        os.path.join(SAVE_DIR, f'backtest_4h_equity_drawdown_{date_tag}.png')
    )

    plot_monthly_heatmap(
        bitget_df, 'Bitget 6coins (4H)',
        os.path.join(SAVE_DIR, f'backtest_4h_bitget_monthly_{date_tag}.png')
    )

    plot_monthly_heatmap(
        binance_df, 'Binance Short+Long (4H)',
        os.path.join(SAVE_DIR, f'backtest_4h_binance_monthly_{date_tag}.png')
    )

    plot_yearly_comparison(
        bitget_df, binance_df,
        os.path.join(SAVE_DIR, f'backtest_4h_yearly_comparison_{date_tag}.png')
    )

    plot_rolling_stats(
        bitget_df, binance_df,
        os.path.join(SAVE_DIR, f'backtest_4h_rolling_risk_{date_tag}.png')
    )

    plot_summary_dashboard(
        bitget_perf, binance_perf,
        os.path.join(SAVE_DIR, f'backtest_4h_summary_{date_tag}.png')
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  완료! ({elapsed/60:.1f}분)")
    print(f"  차트 저장 위치: {SAVE_DIR}")


if __name__ == '__main__':
    main()
