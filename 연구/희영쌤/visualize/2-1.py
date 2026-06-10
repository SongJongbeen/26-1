import plotly.graph_objects as go

# 1. 데이터 설정
regions = ['비수도권', '수도권']

# 이미지에 제공된 데이터
policy_share = [0.69, 0.64]        # 정책수익비중
commercial_share = [0.50, 0.52]    # 사업화수익비중

# 2. 그래프 생성
fig = go.Figure()

# 정책수익비중 막대 추가 (이전 그래프의 정책형 색상인 빨간색 톤 유지)
fig.add_trace(go.Bar(
    x=regions, 
    y=policy_share, 
    name='정책수익비중', 
    marker_color='rgba(214, 39, 40, 0.8)'
))

# 사업화수익비중 막대 추가 (이전 그래프의 사업형 색상인 주황색 톤 유지)
fig.add_trace(go.Bar(
    x=regions, 
    y=commercial_share, 
    name='사업화수익비중', 
    marker_color='rgba(255, 127, 14, 0.8)'
))

# 3. 레이아웃 설정
fig.update_layout(
    barmode='group', # 막대를 옆으로 나란히 배치 (Group)
    title_text='권역별 기술지주회사 수익 비중 비교',
    xaxis_title='지역 권역',
    yaxis_title='수익 비중',
    yaxis=dict(tickformat='.2f', range=[0, 0.8]), # y축을 소수점 둘째 자리까지 표시, 보기 좋게 범위 설정
    legend_title='수익 유형',
    font_size=12,
    template='plotly_white'
)

# 4. 막대 위에 수치 표시 (가시성 향상)
fig.update_traces(texttemplate='%{y}', textposition='outside')

fig.show()