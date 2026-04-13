"""
================================================================================
주의/바늘 코인 재최적화
================================================================================
param_robustness_check.py 결과에서 '주의' 또는 '바늘' 판정된 46개 코인을
다시 베이지안 최적화합니다.

- 기존 optimize_all_coins_long_vs_short.py의 파이프라인 재사용
- 롱우선 / 숏우선 모두 재최적화 → Winner 재선정
- Resume 지원: 이전 결과 자동 스킵
- 결과: reoptimize_weak_coins_results.csv
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# 기존 최적화 코드에서 함수 임포트
from optimize_all_coins_long_vs_short import (
    prepare_coin_data, run_pipeline, save_result_row,
    PHASE1_TRIALS, PHASE2_TRIALS, FEE_RATE, SLIPPAGE_PCT, FUNDING_RATE_8H,
)

SAVE_DIR = os.path.expanduser("~/Downloads")
ROBUSTNESS_CSV = os.path.join(SAVE_DIR, "param_robustness_results.csv")
RESULT_CSV = os.path.join(SAVE_DIR, "reoptimize_weak_coins_results.csv")


def get_target_coins():
    """강건성 검증에서 주의/바늘 판정된 코인 목록 추출"""
    if not os.path.exists(ROBUSTNESS_CSV):
        print(f"  ERROR: {ROBUSTNESS_CSV} 없음")
        return []

    df = pd.read_csv(ROBUSTNESS_CSV)
    bad = df[(df['group'] == '288coin_HYBRID') & (df['verdict'].isin(['주의', '바늘']))]

    targets = []
    for symbol in sorted(bad['symbol'].unique()):
        rows = bad[bad['symbol'] == symbol]
        directions = []
        for _, r in rows.iterrows():
            directions.append({
                'direction': r['direction'],
                'is_primary': r['is_primary'],
                'verdict': r['verdict'],
                'smooth_ratio': r['smooth_ratio'],
                'center_sharpe': r['center_sharpe'],
            })
        # 가장 심각한 판정
        worst = '바늘' if any(d['verdict'] == '바늘' for d in directions) else '주의'
        targets.append({
            'symbol': symbol,
            'worst_verdict': worst,
            'details': directions,
        })

    return targets


def load_completed():
    """이미 완료된 심볼"""
    if os.path.exists(RESULT_CSV):
        try:
            df = pd.read_csv(RESULT_CSV)
            return set(df['Symbol'].unique())
        except Exception:
            return set()
    return set()


def main():
    start_time = time.time()

    print("=" * 80)
    print("  주의/바늘 코인 재최적화")
    print("=" * 80)

    # 대상 코인 추출
    targets = get_target_coins()
    if not targets:
        print("  대상 코인 없음!")
        return

    needle_coins = [t for t in targets if t['worst_verdict'] == '바늘']
    caution_coins = [t for t in targets if t['worst_verdict'] == '주의']

    print(f"\n  대상: {len(targets)}개 코인")
    print(f"    바늘: {len(needle_coins)}개 - {', '.join(t['symbol'] for t in needle_coins)}")
    print(f"    주의: {len(caution_coins)}개")
    print(f"\n  설정: Phase1 {PHASE1_TRIALS} + Phase2 {PHASE2_TRIALS} trials × 2단계 × 2방향")
    print(f"  코인당 예상: ~2-3분")
    print(f"  총 예상: ~{len(targets)*2.5:.0f}분 ({len(targets)*2.5/60:.1f}시간)")
    print(f"  결과: {RESULT_CSV}")

    # Resume
    completed = load_completed()
    if completed:
        print(f"\n  이전 진행: {len(completed)}개 완료, 나머지 재개")

    print("=" * 80)

    processed = 0
    failed = 0
    coin_times = []

    for idx, target in enumerate(targets, 1):
        symbol = target['symbol']

        if symbol in completed:
            continue

        # 진행률
        pct = idx / len(targets) * 100
        eta_str = ""
        if coin_times:
            avg_t = np.mean(coin_times)
            remaining = len(targets) - idx
            eta_min = avg_t * remaining / 60
            eta_str = f" ETA:{eta_min:.0f}분" if eta_min < 60 else f" ETA:{eta_min/60:.1f}시간"

        # 문제 방향 표시
        detail_str = " | ".join(
            f"{d['direction']}({'★' if d['is_primary'] else ''}){d['verdict']}(R={d['smooth_ratio']:.2f})"
            for d in target['details']
        )

        print(f"\n{'━' * 80}")
        print(f"[{idx}/{len(targets)}] ({pct:.0f}%) {symbol} [{target['worst_verdict']}]{eta_str}")
        print(f"  문제: {detail_str}")
        print(f"{'━' * 80}")

        coin_start = time.time()

        # 데이터 로드
        data = prepare_coin_data(symbol, silent=True)
        if not data:
            print(f"  데이터 없음 → 스킵")
            failed += 1
            continue

        print(f"  데이터: {data['start_date'].date()} ~ {data['end_date'].date()} ({data['days']}일)")

        try:
            # 롱 우선 최적화
            print(f"  롱 우선:")
            long_first = run_pipeline(symbol, data, 'long')

            # 숏 우선 최적화
            print(f"  숏 우선:")
            short_first = run_pipeline(symbol, data, 'short')

        except Exception as e:
            print(f"  최적화 오류: {e}")
            failed += 1
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

        coin_elapsed = time.time() - coin_start
        coin_times.append(coin_elapsed)

        w_emoji = '🔵' if winner == '롱 우선' else '🔴'
        # 승자의 강건성 정보
        w_result = long_first if winner == '롱 우선' else short_first
        w_pri_v = w_result.get('pri_verdict', '?')
        w_sec_v = w_result.get('sec_verdict', '?')
        w_pri_r = w_result.get('pri_rank', 1)
        w_sec_r = w_result.get('sec_rank', 1)
        rob_tag = f"강건[P:{w_pri_v}#{w_pri_r} S:{w_sec_v}#{w_sec_r}]"

        print(f"  → {w_emoji} {winner} | "
              f"롱:{lf_cagr:.1f}%/{long_first['combined_mdd']:.1f}% "
              f"숏:{sf_cagr:.1f}%/{short_first['combined_mdd']:.1f}% "
              f"{rob_tag} ({coin_elapsed:.0f}초)")

        # 결과 저장
        row = {
            'Symbol': symbol,
            'Prev_Verdict': target['worst_verdict'],
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

        df_row = pd.DataFrame([row])
        if processed == 0 and not os.path.exists(RESULT_CSV):
            df_row.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
        else:
            df_row.to_csv(RESULT_CSV, mode='a', header=not os.path.exists(RESULT_CSV),
                          index=False, encoding='utf-8-sig')

        processed += 1

    # 최종 요약
    total_elapsed = time.time() - start_time
    print(f"\n\n{'=' * 80}")
    print(f"  완료! 처리: {processed}개, 실패: {failed}개, 소요: {total_elapsed/60:.1f}분")
    print(f"{'=' * 80}")

    if os.path.exists(RESULT_CSV):
        df = pd.read_csv(RESULT_CSV)
        print(f"\n  결과 요약 ({len(df)}개 코인):")

        long_wins = len(df[df['Winner'] == '롱 우선'])
        short_wins = len(df[df['Winner'] == '숏 우선'])
        print(f"    롱 우선 승: {long_wins}개")
        print(f"    숏 우선 승: {short_wins}개")

        if 'Prev_Verdict' in df.columns:
            for verdict in ['바늘', '주의']:
                sub = df[df['Prev_Verdict'] == verdict]
                if len(sub) > 0:
                    avg_cagr = sub[['LF_Combined_CAGR', 'SF_Combined_CAGR']].max(axis=1).mean()
                    print(f"    {verdict} 코인 평균 Best CAGR: {avg_cagr:.1f}%")

        # 강건성 개선 요약
        if 'LF_Pri_Verdict' in df.columns:
            print(f"\n  🛡️ 강건성 검증 결과 (재최적화 후, 승자 기준):")
            verdicts_new = []
            for _, r in df.iterrows():
                if r['Winner'] == '롱 우선':
                    verdicts_new.append({'pri': r.get('LF_Pri_Verdict', ''), 'sec': r.get('LF_Sec_Verdict', ''),
                                          'pri_rank': r.get('LF_Pri_Rank', 1), 'sec_rank': r.get('LF_Sec_Rank', 1)})
                else:
                    verdicts_new.append({'pri': r.get('SF_Pri_Verdict', ''), 'sec': r.get('SF_Sec_Verdict', ''),
                                          'pri_rank': r.get('SF_Pri_Rank', 1), 'sec_rank': r.get('SF_Sec_Rank', 1)})

            for label, key in [('Primary', 'pri'), ('Secondary', 'sec')]:
                counts = {}
                for v in verdicts_new:
                    vd = v[key]
                    if vd:
                        counts[vd] = counts.get(vd, 0) + 1
                rank_gt1 = sum(1 for v in verdicts_new if v.get(f'{key}_rank', 1) not in ('', 1))
                total_v = sum(counts.values())
                if total_v > 0:
                    parts = [f"{k}:{n}" for k, n in sorted(counts.items())]
                    print(f"    {label:>10}: {' | '.join(parts)}")
                    if rank_gt1 > 0:
                        print(f"               → {rank_gt1}개 코인에서 하위 후보로 대체됨")

            # 여전히 바늘/주의인 코인 경고
            still_bad = []
            for _, r in df.iterrows():
                w = r['Winner']
                pri_v = r.get('LF_Pri_Verdict' if w == '롱 우선' else 'SF_Pri_Verdict', '')
                sec_v = r.get('LF_Sec_Verdict' if w == '롱 우선' else 'SF_Sec_Verdict', '')
                if pri_v in ('바늘', '주의') or sec_v in ('바늘', '주의'):
                    still_bad.append(f"{r['Symbol']}(P:{pri_v},S:{sec_v})")
            if still_bad:
                print(f"\n    ⚠️ 여전히 주의/바늘: {len(still_bad)}개")
                for sb in still_bad:
                    print(f"       {sb}")
            else:
                print(f"\n    ✅ 모든 코인 강건/양호 통과!")

        # 타임스탬프 최종 CSV
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_csv = os.path.join(SAVE_DIR, f"reoptimize_weak_FINAL_{ts}.csv")
        df.to_csv(final_csv, index=False, encoding='utf-8-sig')
        print(f"\n  최종 CSV: {final_csv}")


if __name__ == '__main__':
    main()
