import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# 0. 데이터 로드
# ─────────────────────────────────────────────
with open("with_abstracts.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# ─────────────────────────────────────────────
# 1. 불용어 설정
# ─────────────────────────────────────────────
BASE_STOP = set("""
a about above after again against all also although an and any are as at
be been being between both but by can cannot could did do does doing during
each even ever every few for from further get had has have having he her here
him his how i if in into is it its itself just like make me more most my
no nor not now of off on once only or other our out over own per rather
re same see she since so some such than that the their them then there these
they this those through thus to too under until up us very was we were what
when where which while who will with would you your
""".split())

DOMAIN_STOP = {
    "technology", "technological", "technologies", "one", "may", "also",
    "well", "two", "many", "first", "second", "work", "even", "much",
    "use", "used", "using", "based", "make", "made", "way", "ways",
    "part", "pp", "end", "page", "bio", "see", "yet", "review",
    "book", "author", "press", "new", "york", "although", "however",
    "therefore", "thus", "rather", "indeed", "simply", "largely",
    "quite", "already", "often", "still", "long", "far", "early",
}
ALL_STOP = BASE_STOP | DOMAIN_STOP

# ─────────────────────────────────────────────
# 2. 텍스트 정제 + 연도별 그루핑
# ─────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

year_docs = defaultdict(list)
for p in papers:
    yr = p.get("publication_year")
    if yr and p.get("abstract"):
        year_docs[yr].append(clean_text(p["title"] + " " + p["abstract"]))

years_sorted = sorted(year_docs.keys())
corpus = [" ".join(year_docs[yr]) for yr in years_sorted]

print(f"분석 연도: {years_sorted}")
print(f"연도별 논문 수: {[len(year_docs[y]) for y in years_sorted]}")

# ─────────────────────────────────────────────
# 3. TF-IDF 계산
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words=list(ALL_STOP),
    ngram_range=(1, 2),       # unigram + bigram
    min_df=1,
    max_features=500,
    token_pattern=r"[a-z]{3,}"  # 3글자 이상만
)
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=[str(y) for y in years_sorted],
    columns=feature_names
)

# ─────────────────────────────────────────────
# 4. ① 워드클라우드 (연도별)
# ─────────────────────────────────────────────
CMAPS = ["viridis", "plasma", "cividis", "magma", "inferno",
         "cool", "summer", "autumn", "winter", "spring"]

n_years = len(years_sorted)
cols = 3
rows = (n_years + cols - 1) // cols

fig_wc, axes = plt.subplots(rows, cols,
                             figsize=(7 * cols, 5 * rows),
                             facecolor="white")
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, yr in enumerate(years_sorted):
    word_scores = {w: s for w, s in df_tfidf.loc[str(yr)].items() if s > 0}

    wc = WordCloud(
        width=900,
        height=550,
        background_color="white",
        colormap=CMAPS[i % len(CMAPS)],
        max_words=50,
        prefer_horizontal=0.85,
        collocations=False,
        random_state=42,
    ).generate_from_frequencies(word_scores)

    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].set_title(str(yr), fontsize=18, fontweight="bold", pad=8)
    axes[i].axis("off")

# 사용하지 않는 subplot 숨기기
for j in range(n_years, len(axes)):
    axes[j].set_visible(False)

fig_wc.suptitle("TF-IDF Word Cloud by Year",
                fontsize=22, fontweight="bold", y=1.01)
fig_wc.tight_layout(pad=2.0)
fig_wc.savefig("wordcloud_by_year.png", dpi=150,
               bbox_inches="tight", facecolor="white")
plt.show()
print("✅ wordcloud_by_year.png 저장 완료")

# ─────────────────────────────────────────────
# 5. ② TF-IDF 히트맵 (연도 × 키워드)
# ─────────────────────────────────────────────
TOP_N = 25  # 히트맵에 표시할 키워드 수

# 연도별 상위 키워드 합집합 + 전체 평균 상위 키워드
top_kw = set()
for yr in years_sorted:
    top_kw.update(df_tfidf.loc[str(yr)].nlargest(TOP_N).index.tolist())
top_kw.update(df_tfidf.mean(axis=0).nlargest(TOP_N).index.tolist())

# 전체 평균 TF-IDF 기준 내림차순 정렬
top_kw = sorted(top_kw, key=lambda k: -df_tfidf[k].mean())

heat_raw  = df_tfidf[top_kw].T          # shape: (keywords × years)
heat_norm = heat_raw.div(              # 행별 정규화 (0~1)
    heat_raw.max(axis=1).replace(0, 1), axis=0
)

hover_text = [
    [f"{heat_raw.iloc[r, c]:.4f}" for c in range(len(years_sorted))]
    for r in range(len(top_kw))
]

fig_heat = go.Figure(data=go.Heatmap(
    z=heat_norm.values,
    x=[str(y) for y in years_sorted],
    y=heat_norm.index.tolist(),
    colorscale="Viridis",
    text=hover_text,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Year: %{x}<br>"
        "TF-IDF: %{text}"
        "<extra></extra>"
    ),
    colorbar=dict(
        title=dict(text="Norm.<br>TF-IDF", side="right"),
        len=0.6,
        thickness=16,
        tickfont=dict(size=11),
    ),
    xgap=2, ygap=1,
))

fig_heat.update_layout(
    title=dict(
        text=(
            "Top Keyword TF-IDF Score by Year<br>"
            "<span style='font-size:14px; font-weight:normal; color:#555;'>"
            "행별 정규화 · 마우스 호버 시 raw TF-IDF 값 표시</span>"
        ),
        x=0.5, xanchor="center",
    ),
    xaxis=dict(
        title_text="Publication Year",
        tickmode="array",
        tickvals=[str(y) for y in years_sorted],
        tickfont=dict(size=13),
        tickangle=0,
    ),
    yaxis=dict(
        title_text="Keyword",
        autorange="reversed",
        tickfont=dict(size=12),
    ),
    height=max(500, len(top_kw) * 26 + 200),
    margin=dict(l=200, r=80, t=130, b=60),
)

fig_heat.write_image("tfidf_heatmap.png", scale=2)
fig_heat.show()
print("✅ tfidf_heatmap.png 저장 완료")
