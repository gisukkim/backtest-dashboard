"""
백테스트 결과를 docs/ 폴더에 생성 (GitHub Pages용)

실행 방법:
    python generate_report.py

출력:
    docs/charts/*.png  - 차트 이미지 6개
    docs/data/metrics.json - 핵심 지표 (CAGR, MDD, Sharpe)
"""
import sys
import os
import json
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── backtest_bots 경로 패치 (임포트 전에 설정)  ──────────────────────────────
# backtest_bots는 SAVE_DIR에서 binance_bot-5.py를 찾음
# GitHub Actions 환경에서는 레포 루트(= SCRIPT_DIR)에 해당 파일이 있어야 함
import backtest_bots  # noqa: E402 (경로 패치 후 임포트)

backtest_bots.SAVE_DIR = SCRIPT_DIR
backtest_bots.CACHE_DIR = os.path.join(SCRIPT_DIR, "binance_cache")
os.makedirs(backtest_bots.CACHE_DIR, exist_ok=True)

from backtest_bots import (  # noqa: E402
    prepare_coin_data, load_binance_configs,
    run_bitget_portfolio, run_binance_portfolio,
    BITGET_CONFIGS, BITGET_CAPITAL, BINANCE_CAPITAL,
)
from visualize_backtest import (  # noqa: E402
    build_equity_df,
    plot_equity_and_drawdown,
    plot_monthly_heatmap,
    plot_yearly_comparison,
    plot_rolling_stats,
    plot_summary_dashboard,
)

# ── 출력 디렉토리 ─────────────────────────────────────────────────────────────
DOCS_DIR    = os.path.join(SCRIPT_DIR, "docs")
CHARTS_DIR  = os.path.join(DOCS_DIR, "charts")
DATA_DIR    = os.path.join(DOCS_DIR, "data")
for d in [DOCS_DIR, CHARTS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 차트 파일 경로 (고정 이름 — 날짜 태그 없음) ──────────────────────────────
CHART_FILES = {
    "equity_drawdown": os.path.join(CHARTS_DIR, "equity_drawdown.png"),
    "bitget_monthly":  os.path.join(CHARTS_DIR, "bitget_monthly.png"),
    "binance_monthly": os.path.join(CHARTS_DIR, "binance_monthly.png"),
    "yearly":          os.path.join(CHARTS_DIR, "yearly_comparison.png"),
    "rolling_risk":    os.path.join(CHARTS_DIR, "rolling_risk.png"),
    "summary":         os.path.join(CHARTS_DIR, "summary.png"),
}


def _check_api_connectivity():
    """Binance Futures API 접근 가능 여부 사전 확인"""
    import requests as _req
    url = "https://fapi.binance.com/fapi/v1/ping"
    try:
        r = _req.get(url, timeout=10)
        if r.status_code == 200:
            print("  [API] fapi.binance.com 접근 OK")
            return True
        else:
            print(f"  [API] fapi.binance.com 응답 이상: HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"  [API] fapi.binance.com 접근 실패: {e}")
        return False


def main():
    print("=" * 60)
    print("  백테스트 리포트 생성 (GitHub Pages용)")
    print("=" * 60)
    start_time = datetime.now()

    # ── API 연결 확인 ──────────────────────────────────────────────────────────
    if not _check_api_connectivity():
        print("\n  ❌ Binance Futures API에 접근할 수 없습니다.")
        print("     GitHub Actions 서버에서 fapi.binance.com이 차단되었을 수 있습니다.")
        sys.exit(1)

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    print("\n[1/3] 데이터 로드 중...")

    bitget_cache = {}
    for cfg in BITGET_CONFIGS:
        sym = cfg["symbol"]
        data = prepare_coin_data(sym, silent=False)
        if data:
            bitget_cache[sym] = data
        else:
            print(f"  ⚠  {sym} 로드 실패 (데이터 부족 또는 API 오류)")
    print(f"  Bitget: {len(bitget_cache)} 코인 로드")

    if len(bitget_cache) == 0:
        print("  ❌ Bitget 코인 데이터를 하나도 로드하지 못했습니다. API 접근 문제를 확인하세요.")
        sys.exit(1)

    short_configs, long_configs = load_binance_configs()
    if not short_configs and not long_configs:
        print("  ⚠  binance_bot-5.py 파일을 찾을 수 없습니다.")
        print(f"     파일 위치: {os.path.join(SCRIPT_DIR, 'binance_bot-5.py')}")
        sys.exit(1)

    binance_cache = {}
    all_symbols = (
        {c["symbol"] for c in short_configs} | {c["symbol"] for c in long_configs}
    )
    fail_count = 0
    for i, sym in enumerate(sorted(all_symbols), 1):
        if i % 20 == 0:
            print(f"  Binance 진행: {i}/{len(all_symbols)} (로드 성공: {len(binance_cache)}, 실패: {fail_count})...")
        data = prepare_coin_data(sym, silent=True)
        if data:
            binance_cache[sym] = data
        else:
            fail_count += 1
    print(f"  Binance: {len(binance_cache)} 코인 로드 (실패: {fail_count})")

    if len(binance_cache) == 0:
        print("  ❌ Binance 코인 데이터를 하나도 로드하지 못했습니다.")
        sys.exit(1)

    # ── 공통 종료일 ────────────────────────────────────────────────────────────
    all_ends = [
        d["df_4h"]["timestamp"].max()
        for d in list(bitget_cache.values()) + list(binance_cache.values())
    ]
    if not all_ends:
        print("  ❌ 데이터가 없어 종료일을 계산할 수 없습니다.")
        sys.exit(1)
    end_date = min(all_ends)
    print(f"  공통 종료일: {end_date}")

    # ── 백테스트 실행 ──────────────────────────────────────────────────────────
    print("\n[2/3] 백테스트 실행 중...")
    print("  Bitget 포트폴리오...")
    bitget_perf = run_bitget_portfolio(
        use_1h=False, coin_data_cache=bitget_cache, end_date=end_date
    )
    print("  Binance 포트폴리오...")
    binance_perf = run_binance_portfolio(
        use_1h=False,
        short_configs=short_configs,
        long_configs=long_configs,
        coin_data_cache=binance_cache,
        end_date=end_date,
    )

    if not bitget_perf or not binance_perf:
        print("  ❌ 백테스트 실패!")
        sys.exit(1)

    print(
        f"  Bitget  → CAGR={bitget_perf['cagr']:.1f}%  "
        f"MDD={bitget_perf['mdd']:.1f}%  Sharpe={bitget_perf['sharpe']:.3f}"
    )
    print(
        f"  Binance → CAGR={binance_perf['cagr']:.1f}%  "
        f"MDD={binance_perf['mdd']:.1f}%  Sharpe={binance_perf['sharpe']:.3f}"
    )

    # ── 차트 생성 ──────────────────────────────────────────────────────────────
    print("\n[3/3] 차트 생성 중...")
    bitget_df  = build_equity_df(bitget_perf)
    binance_df = build_equity_df(binance_perf)

    plot_equity_and_drawdown(bitget_df, binance_df, CHART_FILES["equity_drawdown"])
    plot_monthly_heatmap(bitget_df,  "Bitget 6coins (4H)",        CHART_FILES["bitget_monthly"])
    plot_monthly_heatmap(binance_df, "Binance Short+Long (4H)",   CHART_FILES["binance_monthly"])
    plot_yearly_comparison(bitget_df, binance_df,                  CHART_FILES["yearly"])
    plot_rolling_stats(bitget_df, binance_df,                      CHART_FILES["rolling_risk"])
    plot_summary_dashboard(bitget_perf, binance_perf,              CHART_FILES["summary"])

    # ── 메트릭 JSON 저장 ───────────────────────────────────────────────────────
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)
    metrics = {
        "updated_at_kst": now_kst.strftime("%Y-%m-%d %H:%M KST"),
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "backtest_end_date": str(end_date)[:10],
        "bitget": {
            "name": "Bitget 6coins",
            "capital": BITGET_CAPITAL,
            "cagr":   round(float(bitget_perf["cagr"]),   1),
            "mdd":    round(float(bitget_perf["mdd"]),    1),
            "sharpe": round(float(bitget_perf["sharpe"]), 3),
        },
        "binance": {
            "name": "Binance Short+Long",
            "capital": BINANCE_CAPITAL,
            "cagr":   round(float(binance_perf["cagr"]),   1),
            "mdd":    round(float(binance_perf["mdd"]),    1),
            "sharpe": round(float(binance_perf["sharpe"]), 3),
        },
    }
    metrics_path = os.path.join(DATA_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  메트릭 저장: {metrics_path}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ 완료! ({elapsed / 60:.1f}분)")
    print(f"   결과 폴더: {DOCS_DIR}")


if __name__ == "__main__":
    main()
