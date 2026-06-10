import plotly.graph_objects as go

# 1. 데이터 설정
# Y축에 위에서부터 아래로 순서대로 출력되도록 역순으로 리스트 구성
variables = ['자회사규모', '전담인력', '자본금', '지역(수도권)'] 

# 각 변수의 회귀계수 (Coef.)
coefs = [0.07, 0.16, 0.24, 0.40] 

# 95% 신뢰구간의 하한값 [0.025] 과 상한값 [0.975]
lowers = [0.04, 0.01, 0.03, -0.14] 
uppers = [0.09, 0.32, 0.45, 0.94]  

# 에러바(오차선) 길이 계산
error_plus = [u - c for u, c in zip(uppers, coefs)]
error_minus = [c - l for c, l in zip(coefs, lowers)]

# 2. 그래프 생성
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=coefs,
    y=variables,
    mode='markers',
    marker=dict(size=12, color='rgba(31, 119, 180, 1)'), # 점(계수) 색상 및 크기
    error_x=dict(
        type='data',
        symmetric=False,
        array=error_plus,
        arrayminus=error_minus,
        color='rgba(31, 119, 180, 0.8)',
        thickness=2, # 선 굵기
        width=5      # 선 끝의 캡(T자 모양) 크기
    ),
    name='Coefficient (95% CI)'
))

# 3. 기준선 (x=0) 추가 (가장 중요: 이 선을 넘어가면 유의하지 않음)
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")

# 4. 레이아웃 설정
fig.update_layout(
    title='로지스틱 회귀분석 결과: 지역 차이 모델',
    xaxis_title='회귀계수 (Coefficient) 및 95% 신뢰구간',
    yaxis_title='변수',
    template='plotly_white',
    height=400 # 표의 행이 4개이므로 높이를 아담하게 설정
)

fig.show()