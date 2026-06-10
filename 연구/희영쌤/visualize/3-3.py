import plotly.graph_objects as go

# 1. 공통 변수 (Y축)
variables = ['자본금', '업력', '국책사업비']

# 2. t1 (다음 해) 데이터 셋
coefs_t1 = [0.63, 0.01, 0.19]
lowers_t1 = [0.40, -0.03, 0.03]
uppers_t1 = [0.87, 0.05, 0.36]

error_plus_t1 = [u - c for u, c in zip(uppers_t1, coefs_t1)]
error_minus_t1 = [c - l for c, l in zip(coefs_t1, lowers_t1)]

# 3. t2 (2년 후) 데이터 셋
coefs_t2 = [0.86, -0.10, 0.35]
lowers_t2 = [0.44, -0.21, 0.12]
uppers_t2 = [1.28, 0.02, 0.58]

error_plus_t2 = [u - c for u, c in zip(uppers_t2, coefs_t2)]
error_minus_t2 = [c - l for c, l in zip(coefs_t2, lowers_t2)]

# 4. 그래프 생성
fig = go.Figure()

# t1 (다음 해) 점과 선 추가 (파란색 계열)
# 겹치지 않게 y축 위치를 미세하게 조정하기 위해 변수 이름 뒤에 공백 등을 추가
variables_t1 = [var + ' (t1)' for var in variables]

fig.add_trace(go.Scatter(
    x=coefs_t1,
    y=variables,
    mode='markers',
    name='t1: 다음 해',
    marker=dict(size=12, color='rgba(31, 119, 180, 1)'),
    error_x=dict(
        type='data',
        symmetric=False,
        array=error_plus_t1,
        arrayminus=error_minus_t1,
        color='rgba(31, 119, 180, 0.8)', # 이제 리스트가 아닌 단일 색상!
        thickness=2,
        width=5
    )
))

# t2 (2년 후) 점과 선 추가 (주황색 계열)
fig.add_trace(go.Scatter(
    x=coefs_t2,
    y=variables, # 같은 Y축 라인에 그립니다
    mode='markers',
    name='t2: 2년 후',
    marker=dict(size=12, color='rgba(255, 127, 14, 1)'),
    error_x=dict(
        type='data',
        symmetric=False,
        array=error_plus_t2,
        arrayminus=error_minus_t2,
        color='rgba(255, 127, 14, 0.8)', # 단일 색상
        thickness=2,
        width=5
    )
))

# 5. 기준선 (x=0) 추가
fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")

# 6. 레이아웃 설정
fig.update_layout(
    title='시간 흐름에 따른 수익 발생 요인 변화 (다음 해 vs 2년 후)',
    xaxis_title='회귀계수 (Coefficient) 및 95% 신뢰구간',
    yaxis_title='변수',
    template='plotly_white',
    height=400,
    hovermode="y unified" # 마우스를 올리면 같은 라인의 데이터가 같이 보이도록 설정
)

fig.show()