import pandas as pd
import numpy as np

# 1. 데이터 로드 및 매핑
df = pd.read_csv('study2_risk_perception_results.csv')

def map_prototype(model_name):
    if 'grok' in model_name.lower(): return 'Individualism'
    elif 'claude' in model_name.lower(): return 'Hierarchy'
    else: return 'Egalitarianism'

df['Prototype'] = df['Model'].apply(map_prototype)

# 각 척도별 평균 데이터프레임 생성
mean_risk = df.groupby(['Risk_Issue', 'Prototype'])['Riskiness'].mean().unstack()
mean_unacc = df.groupby(['Risk_Issue', 'Prototype'])['Unacceptability'].mean().unstack()
mean_future = df.groupby(['Risk_Issue', 'Prototype'])['Harm_to_future_generations'].mean().unstack()
mean_delay = df.groupby(['Risk_Issue', 'Prototype'])['Delayed_effects'].mean().unstack()

def compare(issue, metric_df, target_proto, condition='higher'):
    """상대적 비교를 수행하여 결과를 문자열로 반환하는 함수"""
    try:
        val_target = metric_df.loc[issue, target_proto]
        val_others = [metric_df.loc[issue, p] for p in metric_df.columns if p != target_proto]
        
        if condition == 'higher':
            is_match = all(val_target > v for v in val_others)
            status = "✅ 일치" if is_match else "❌ 불일치"
        else: # 'lower'
            is_match = all(val_target < v for v in val_others)
            status = "✅ 일치" if is_match else "❌ 불일치"
            
        scores = f"({target_proto}: {val_target:.2f} vs Others: {[round(v,2) for v in val_others]})"
        return f"{status} {scores}"
    except KeyError:
        return "⚠️ 데이터 없음"

print("==================================================")
print("1. 평등주의 (Egalitarianism) 가설 검증")
print("==================================================")
print("- [환경적 위협] 원자력 위험성 (Egal > Others):", compare('Nuclear power', mean_risk, 'Egalitarianism', 'higher'))
print("- [환경적 위협] 오존층 파괴 위험성 (Egal > Others):", compare('Ozone depletion', mean_risk, 'Egalitarianism', 'higher'))
print("- [비자연적 위험] 식용 색소 위험성 (Egal > Others):", compare('Food colourings', mean_risk, 'Egalitarianism', 'higher'))
print("- [비자연적 위험] 유전 공학 위험성 (Egal > Others):", compare('Genetic engineering', mean_risk, 'Egalitarianism', 'higher'))
print("- [비자연적 위험] 전자레인지 위험성 (Egal > Others):", compare('Microwave ovens', mean_risk, 'Egalitarianism', 'higher'))
print("- [미래 세대 피해] 원자력 (Egal > Others):", compare('Nuclear power', mean_future, 'Egalitarianism', 'higher'))
print("- [미래 세대 피해] 자동차 운전 (Egal > Others):", compare('Car driving', mean_future, 'Egalitarianism', 'higher'))
print("- [자동차 운전 수용불가성] (Egal > Others):", compare('Car driving', mean_unacc, 'Egalitarianism', 'higher'))

print("\n==================================================")
print("2. 위계주의 (Hierarchy / Claude) 가설 검증")
print("==================================================")
print("- [사회적 위협] 노상강도 위험성 (Hier > Others):", compare('Mugging', mean_risk, 'Hierarchy', 'higher'))
print("- [사회적 위협] 테러리즘 위험성 (Hier > Others):", compare('Terrorism', mean_risk, 'Hierarchy', 'higher'))
print("- [환경적 위협 우려 낮음] 오존층 파괴 위험성 (Hier < Others):", compare('Ozone depletion', mean_risk, 'Hierarchy', 'lower'))
print("- [사회적 위협의 지연 효과] 테러리즘 (Hier > Others):", compare('Terrorism', mean_delay, 'Hierarchy', 'higher'))

print("\n==================================================")
print("3. 개인주의 (Individualism / Grok) 가설 검증")
print("==================================================")
print("- [환경적 위협 무시] 오존층 파괴 위험성 (Indiv < Others):", compare('Ozone depletion', mean_risk, 'Individualism', 'lower'))
print("- [개인적 통제 가능 위험 무시] 자동차 운전 위험성 (Indiv < Others):", compare('Car driving', mean_risk, 'Individualism', 'lower'))
print("- [개인적 통제 가능 위험 무시] 음주 위험성 (Indiv < Others):", compare('Alcoholic drinks', mean_risk, 'Individualism', 'lower'))
print("- [경제 저해 요소 우려 낮음] 전쟁 위험성 (Indiv < Others):", compare('War', mean_risk, 'Individualism', 'lower'))