"""
optimize_long_fixed.py 빠른 검증 테스트
소수 코인(ARBUSDT, SUIUSDT, BTCUSDT)으로 코드 동작 확인
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from optimize_long_fixed import (
    load_short_configs_and_exclusions, prepare_coin_data,
    backtest_long, optimize_long_phase1, SHORT_CSV_PATH,
    FEE_RATE, SLIPPAGE_PCT, FUNDING_PER_4H, FUNDING_RATE_8H
)

def test():
    print("=" * 80)
    print("🧪 optimize_long_fixed.py 빠른 검증 테스트")
    print("=" * 80)
    print(f"수수료: {FEE_RATE*100:.2f}%, 슬리피지: {SLIPPAGE_PCT*100:.2f}%, 펀딩비: {FUNDING_RATE_8H*100:.3f}%/8h")
    print()

    # 1. CSV 로드 테스트
    print("── 1. CSV 로드 테스트 ──")
    short_configs, excluded, no_short_filter = load_short_configs_and_exclusions(SHORT_CSV_PATH)
    print(f"  숏 파라미터: {len(short_configs)}개")
    print(f"  제외 코인: {len(excluded)}개")
    print(f"  숏필터 OFF: {len(no_short_filter)}개")

    # BTCUSDT가 no_short_filter에 있는지 확인
    if 'BTCUSDT' in no_short_filter:
        print("  ✅ BTCUSDT: 숏필터 OFF (CAGR < 0% 정상)")
    elif 'BTCUSDT' in short_configs:
        print(f"  ℹ️ BTCUSDT: 숏필터 ON (MA={short_configs['BTCUSDT']['ma_period']})")
    elif 'BTCUSDT' in excluded:
        print("  ⚠️ BTCUSDT: 제외됨")
    print()

    # 2. 코인별 테스트
    test_coins = ['ARBUSDT', 'SUIUSDT', 'BTCUSDT']

    for symbol in test_coins:
        print(f"── 2. {symbol} 테스트 ──")

        # 숏 필터 결정
        if symbol in no_short_filter:
            short_cfg = None
            filter_status = "숏필터 OFF"
        elif symbol in short_configs:
            short_cfg = short_configs[symbol]
            filter_status = f"숏필터 ON (MA={short_cfg['ma_period']})"
        else:
            short_cfg = None
            filter_status = "숏 파라미터 없음"

        print(f"  {filter_status}")

        if symbol in excluded:
            print(f"  ⚠️ 이 코인은 제외 대상입니다 (테스트 목적으로 계속)")

        # 데이터 다운로드
        data = prepare_coin_data(symbol)
        if not data:
            print(f"  ❌ 데이터 부족, 스킵")
            print()
            continue

        print(f"  기간: {data['start_date'].date()} ~ {data['end_date'].date()} ({data['days']}일)")
        print(f"  4H 캔들: {len(data['df_4h'])}개, 1D 캔들: {len(data['df_daily'])}개")

        # 수동 백테스트 (기본 파라미터)
        print(f"\n  ▶ 수동 백테스트 (MA=100, SK=60, SKs=20, SD=14, Lev=3):")
        result = backtest_long(data, short_cfg, 100, 60, 20, 14, 3)
        print(f"    CAGR: {result['cagr']:.1f}%")
        print(f"    MDD: {result['mdd']:.1f}%")
        print(f"    Sharpe: {result['sharpe']:.2f}")
        print(f"    거래수: {result['trades']}회")
        print(f"    숏필터 비율: {result.get('short_filter_ratio', 0):.1f}%")

        # 1단계 최적화 테스트 (5 trials만)
        print(f"\n  ▶ 1단계 최적화 (5 trials):")
        import optimize_long_fixed as olf
        orig_trials = olf.PHASE1_TRIALS
        olf.PHASE1_TRIALS = 5

        p1_result = optimize_long_phase1(symbol, data, short_cfg)

        olf.PHASE1_TRIALS = orig_trials

        print(f"    Lev: {p1_result['Long_Lev']}x")
        print(f"    MA: {p1_result['Long_MA']}, SK: {p1_result['Long_SK']}, SKs: {p1_result['Long_SKs']}, SD: {p1_result['Long_SD']}")
        print(f"    CAGR: {p1_result['CAGR']:.1f}%")
        print(f"    MDD: {p1_result['MDD']:.1f}%")
        print(f"    Sharpe: {p1_result['Sharpe']:.2f}")
        print(f"    거래: {p1_result['Trades']}회")
        print(f"    숏필터: {p1_result['Short_Filter_Ratio']:.1f}%")
        print()

    print("=" * 80)
    print("✅ 검증 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test()
