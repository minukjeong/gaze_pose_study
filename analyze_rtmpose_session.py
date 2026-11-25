"""Create a quality report and time-series plot for an RTMPose posture session."""
import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS = {
    'head_bow_change_proxy_deg': ('머리 숙임 변화', 'deg'),
    'upper_body_forward_change_percent': ('상체 전방 이동', '%'),
    'shoulder_tilt_deg': ('어깨 기울기', 'deg'),
    'head_shoulder_relative_change_percent': ('머리-어깨 상대 변화', '%SW'),
    'personal_change_percent_shoulder_width': ('개인 내 자세 변화량', '%SW'),
}


def number(value):
    try:
        return float(value) if value.strip() else math.nan
    except (ValueError, AttributeError):
        return math.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', nargs='?', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'rtmpose_sessions'
    session = args.session or max((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    csv_path = session / 'posture_metrics.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No records in {csv_path}')

    timestamps = [float(row['timestamp']) for row in rows]
    columns = [name for name in LABELS if name in rows[0]]
    baseline_column = 'personal_change_percent_shoulder_width'
    first = next((i for i, row in enumerate(rows) if not math.isnan(number(row.get(baseline_column, '')))), None)
    if first is None:
        raise ValueError('No baseline-dependent values. Press B during recording.')
    elapsed = [value - timestamps[first] for value in timestamps]
    post_rows = rows[first:]

    summaries = {}
    for column in columns:
        values = [number(row[column]) for row in post_rows]
        values = [value for value in values if not math.isnan(value)]
        summaries[column] = {
            'n': len(values),
            'valid': 100 * len(values) / len(post_rows),
            'mean': statistics.mean(values),
            'sd': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'median': statistics.median(values),
            'max': max(values),
        }

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    colors = ['#355C7D', '#C06C84', '#2A9D8F', '#E76F51', '#6C5B7B']
    for axis, column, color in zip(axes, columns, colors):
        values = [number(row[column]) for row in post_rows]
        axis.plot(elapsed[first:], values, color=color, linewidth=1.8)
        label, unit = LABELS[column]
        axis.set_ylabel(f'{label}\n({unit})')
        axis.grid(alpha=.25)
    axes[-1].set_xlabel('기준 자세 설정 후 경과 시간 (초)')
    fig.suptitle(f'RTMPose 상반신 자세 지표 — {session.name}', fontsize=15)
    plot_path = session / 'posture_timeseries.png'
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    duration = timestamps[-1] - timestamps[0]
    baseline_elapsed = timestamps[first] - timestamps[0]
    lines = [
        '# RTMPose 상반신 자세 데이터 분석', '',
        f'- 세션: `{session.name}`',
        f'- 전체 기록: {len(rows)}행, {duration:.3f}초',
        f'- 기준 자세 설정: 촬영 시작 {baseline_elapsed:.3f}초 후',
        f'- 분석 구간: 기준 설정 이후 {len(post_rows)}행, {timestamps[-1]-timestamps[first]:.3f}초', '',
        '## 기준 자세 이후 요약', '',
        '| 지표 | 유효률 | 평균 | 표준편차 | 중앙값 | 최솟값 | 최댓값 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for column in columns:
        label, unit = LABELS[column]
        s = summaries[column]
        lines.append(f"| {label} ({unit}) | {s['valid']:.1f}% | {s['mean']:.3f} | {s['sd']:.3f} | {s['median']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")
    lines += [
        '', '## 판정', '',
        '기준 자세 설정 이후 다섯 지표의 유효률이 모두 90% 이상이면 상반신 키포인트 검출과 기록은 정상으로 판정한다.',
        '이 값들은 개인 기준 대비 영상 대리지표이며 경추·흉추의 해부학적 각도가 아니다.',
        '조건별 자세를 비교하려면 다음 수집부터 각 자세의 시작·종료 시점 또는 조건 라벨을 함께 기록해야 한다.',
    ]
    report_path = session / 'analysis_report.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('REPORT', report_path)
    print('PLOT', plot_path)


if __name__ == '__main__':
    main()
