import plotly.graph_objects as go

# 1. 데이터 설정 (표 데이터 역순 배치)
variables = [
    '자본금', 
    '상호작용 (국책사업비 × 초기조직)', 
    '초기조직여부', 
    '국책사업비'
]

# 각 변수의 회귀계수 (Coef.)
coefs = [0.67, 0.15, -0.14, 0.14]

# 95% 신뢰구간 하한값, 상한값
lowers = [0.42, -0.19, -0.96, -0.08]
uppers = [0.92, 0.49, 0.69, 0.36]

# 에러바 길이 계산
error_plus = [u - c for u, c in zip(uppers, coefs)]
error_minus = [c - l for c, l in zip(coefs, lowers)]

# 2. 그래프 생성
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=coefs,
    y=variables,
    mode='markers',
    marker=dict(size=12, color='rgba(23, 190, 207, 1)'), # 청록색(Cyan) 톤 적용
    error_x=dict(
        type='data',
        symmetric=False,
        array=error_plus,
        arrayminus=error_minus,
        color='rgba(23, 190, 207, 0.8)',
        thickness=2,
        width=5
    ),
    name='Coefficient (95% CI)'
))

# 3. 기준선 (x=0) 추가 (의미 없음의 기준)
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")

# 4. 레이아웃 설정
fig.update_layout(
    title='로지스틱 회귀분석: 초기 조직 상호작용 효과 검증',
    xaxis_title='회귀계수 (Coefficient) 및 95% 신뢰구간',
    yaxis_title='변수',
    template='plotly_white',
    height=400
)

fig.show()