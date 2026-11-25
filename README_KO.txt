자세별 Ablation Study 파일 묶음 (연구용 / 실기기 미검증)

1) gaze_runtime.py, dynamic_gaze.py, gaze_mapping.py, target_recording.py를 기존 프로젝트에 복사하기 전에 반드시 원본 백업.
2) 변경된 것은 gaze_runtime.py와 target_recording.py뿐이며, 나머지 2개 파일은 버전 일치를 위해 제공.
3) 기존 프로젝트의 pose_experiment.py, rtmpose_gaze_overlay.py, gaze_visualization.py, gaze_targets.png, xgaze-js 등 의존성은 그대로 필요.
4) 기존과 동일하게 실행. 기본 옵션 --calibration-mode dynamic --pose-policy experiment.
5) 9점 캘리브레이션 후 ENTER로 LIVE 진입.
6) 12개 과녁 기록을 시작하기 전에, 기록 중이 아닐 때 숫자키로 자세 지정:
     1: neutral, 2: forward, 3: backward, 4: neutral_return
   숫자키로 레이블을 정한 뒤 S를 눌러 12개 과녁(각 3초)을 기록. 자세는 한 번의 패스 동안 유지.
   다시 S를 누르기 전에 각 패스를 여러 번 반복해도 됨. 숫자키는 한 패스 진행 중에는 무시.
   P: 일시정지/다시 시작, E: 현재 패스 중단, Q: 종료.
7) 출력 파일: SESSION/target_records/<session>/gaze_target_distances.csv
   자세 레이블, base, selected, head 후보, head_shoulders 후보, 각 status가 기록됨.
8) 분석 실행: python analyze_ablation.py "경로/가진/gaze_target_distances.csv"
   같은 폴더에 ablation_by_target.csv 생성. 정착 구간(첫 1초 제외)의 목표점별 평균과
   동일 프레임에서의 기본 매핑 대비 후보 개선량을 산출.

주의사항:
 - head/head_shoulders 후보는 둘 다 조건부 선택과 무관하게 독립 진단 계산하며 extrapolate=True를 사용.
   즉 후보의 적용 여부를 테스트하기 위한 성능 기록으로, 검증을 통과한 시스템의 출력이라고 해석하면 안 됨.
 - 실제 선택 출력(selected)은 원래 코드의 dynamic.predict() 동작을 그대로 유지함.
 - head/shoulders 후보에 필요한 입력이 없으면 빈 값. 평가에서 해당 표본 수/coverage도 함께 공개해야 함.
 - 유효 좌표가 있는 구간만 오차를 산출하므로 출력 실패율도 따로 검토할 것.
 - 목표점은 사용자가 바라보도록 안내된 위치이며 독립형 시선 측정기의 정답이 아님.
 - 한 사람의 반복 프레임으로 p-value나 사용자 일반화 효과를 주장하지 말 것.
 - 실험용으로만 검증했으며 웹캠, 모델 가중치, 종속 모듈을 실제 실행하는 통합 테스트는 미수행.
