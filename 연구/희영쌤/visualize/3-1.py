import plotly.graph_objects as go

# 1. 데이터 설정 (표에 있는 데이터 그대로 사용)
# Y축에 위에서부터 아래로 순서대로 출력되도록 역순 구성
variables = ['전담인력(기존)', '국책사업비'] 

# 각 변수의 회귀계수 (Coef.)
coefs = [0.85, 0.22] 

# 95% 신뢰구간의 하한값 [0.025] 과 상한값 [0.975]
lowers = [0.78, 0.08] 
uppers = [0.93, 0.36]  

# 에러바(오차선) 길이 계산
error_plus = [u - c for u, c in zip(uppers, coefs)]
error_minus = [c - l for c, l in zip(coefs, lowers)]

# 2. 그래프 생성
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=coefs,
    y=variables,
    mode='markers',
    marker=dict(size=12, color='rgba(44, 160, 44, 1)'), # 이번엔 초록색 톤으로 설정해 보았습니다
    error_x=dict(
        type='data',
        symmetric=False,
        array=error_plus,
        arrayminus=error_minus,
        color='rgba(44, 160, 44, 0.8)',
        thickness=2,
        width=5
    ),
    name='Coefficient (95% CI)'
))

# 3. 기준선 (x=0) 추가 (가장 중요: 이 선을 넘어가면 유의하지 않음)
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")

# 4. 레이아웃 설정
fig.update_layout(
    title='OLS 회귀분석 결과: 전담인력 증가 모델',
    xaxis_title='회귀계수 (Coefficient) 및 95% 신뢰구간',
    yaxis_title='변수',
    template='plotly_white',
    height=300 # 변수가 2개뿐이라 높이를 더 낮췄습니다
)

fig.show()