import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows: 'Malgun Gothic', Mac: 'AppleGothic')
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False 

# 데이터 로드
df = pd.read_csv('연구/소통학회/ai_cultural_prototypes_normalized.csv')

# 4사분면 분류 함수 (논문 기준: 0.5를 분기점으로 사용)
def classify_culture(row):
    grid = row['Grid_Index']
    group = row['Group_Index']
    
    if grid >= 0.5 and group >= 0.5:
        return '계층주의 (Hierarchy)'
    elif grid >= 0.5 and group < 0.5:
        return '숙명론 (Fatalism)'
    elif grid < 0.5 and group >= 0.5:
        return '평등주의 (Egalitarianism)'
    else: # grid < 0.5 and group < 0.5
        return '개인주의 (Individualism)'

# 문화원형 분류 열 추가
df['Culture_Type'] = df.apply(classify_culture, axis=1)

print("--- 각 AI 모델별 문화원형 분류 결과 ---")
print(df[['Model', 'Grid_Index', 'Group_Index', 'Culture_Type']])

# 시각화 (산점도)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Group_Index', y='Grid_Index', hue='Model', s=150, palette='Set1')

# 사분면 분할선 그리기
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.7)

# 축 설정
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Group Index (집단/평등 지향)', fontsize=12)
plt.ylabel('Grid Index (규범/권위 구속)', fontsize=12)
plt.title('생성형 AI 모델의 문화원형(Grid-Group) 포지셔닝', fontsize=16)

# 각 사분면 라벨
plt.text(0.25, 0.75, '숙명론 (Fatalism)\n[High Grid, Low Group]', ha='center', va='center', alpha=0.5, fontsize=12)
plt.text(0.75, 0.75, '계층주의 (Hierarchy)\n[High Grid, High Group]', ha='center', va='center', alpha=0.5, fontsize=12)
plt.text(0.25, 0.25, '개인주의 (Individualism)\n[Low Grid, Low Group]', ha='center', va='center', alpha=0.5, fontsize=12)
plt.text(0.75, 0.25, '평등주의 (Egalitarianism)\n[Low Grid, High Group]', ha='center', va='center', alpha=0.5, fontsize=12)

# 모델 라벨링 (점 옆에 텍스트 표시)
for i in range(df.shape[0]):
    plt.text(df['Group_Index'][i] + 0.015, df['Grid_Index'][i], 
             df['Model'][i].split('/')[-1], # 모델명만 짧게 출력
             fontsize=9)

plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 범례를 그래프 밖으로
plt.tight_layout()
plt.savefig('ai_cultural_prototypes_plot.png', dpi=300)
plt.show()