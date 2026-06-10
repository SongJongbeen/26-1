import plotly.graph_objects as go

# 1. 데이터 설정 (표에 있는 데이터 반영)
# Y축에 위에서부터 아래로 순서대로 출력되도록 역순 구성
variables = ['자회사규모(기존)', '국책사업비'] 

# 각 변수의 회귀계수 (Coef.)
coefs = [0.05, 0.18] 

# 95% 신뢰구간의 하한값 [0.025] 과 상한값 [0.975]
lowers = [0.02, 0.01] 
uppers = [0.07, 0.35]  

# 에러바(오차선) 길이 계산
error_plus = [u - c for u, c in zip(uppers, coefs)]
error_minus = [c - l for c, l in zip(coefs, lowers)]

# 2. 그래프 생성
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=coefs,
    y=variables,
    mode='markers',
    marker=dict(size=12, color='rgba(148, 103, 189, 1)'), # 이번엔 보라색 톤 적용
    error_x=dict(
        type='data',
        symmetric=False,
        array=error_plus,
        arrayminus=error_minus,
        color='rgba(148, 103, 189, 0.8)',
        thickness=2,
        width=5
    ),
    name='Coefficient (95% CI)'
))

# 3. 기준선 (x=0) 추가 (의미 없음의 기준)
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")

# 4. 레이아웃 설정
fig.update_layout(
    title='OLS 회귀분석 결과: 신규 자회사 증가 모델',
    xaxis_title='회귀계수 (Coefficient) 및 95% 신뢰구간',
    yaxis_title='변수',
    template='plotly_white',
    height=300
)

fig.show()