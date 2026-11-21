import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ── 한글 폰트 (Streamlit CSS) ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── CSV 읽기 (utf-8-sig → cp949) ──────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """파일 객체에서 바이트를 꺼내 인코딩을 시도하며 읽는다."""
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 (KeyError 방지) ─────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ── 추세선 보조선 (np.polyfit 직접 사용) ─────────────────────────────────
def add_trendline(fig, x, y, color="red"):
    """산점도 fig에 기울기·절편을 구해 추세선을 추가한다."""
    mask = pd.notna(x) & pd.notna(y)
    xs, ys = x[mask], y[mask]
    if len(xs) < 2:
        return
    m, b = np.polyfit(xs, ys, 1)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=m * xs + b,
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name=f"추세선 (기울기 {m:.3f})",
            hoverinfo="skip",
        ),
        row=None,
        col=None,
    )


# ── Plotly 공통 폰트 설정 ─────────────────────────────────────────────────
def set_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig


# ── 그래프 1: 환경 시계열 (온도·습도) ─────────────────────────────────────
def plot_env_timeseries(env_df):
    cols = [c for c in ["temperature", "humidity"] if c in env_df.columns]
    if "time" not in env_df.columns or not cols:
        return None
    sub = env_df[["time"] + cols].dropna()
    if len(sub) < 2:
        return None
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["온도 (°C)", "습도 (%)"],
        vertical_spacing=0.12,
    )
    colors = {"temperature": "#e74c3c", "humidity": "#3498db"}
    for col in cols:
        row = 1 if col == "temperature" else 2
        fig.add_trace(
            go.Scatter(
                x=sub["time"],
                y=sub[col],
                mode="lines",
                name=col,
                line=dict(color=colors.get(col, "#333"), width=1.2),
                hovertemplate="%{x|%m/%d %H:%M}<br>%{y:.1f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    fig.update_xaxes(title_text="시간", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(height=480, margin=dict(l=50, r=20, t=40, b=40))
    return set_font(fig)


# ── 그래프 2: 환경 산점도 (온도 vs 습도) ──────────────────────────────────
def plot_env_scatter(env_df):
    if "temperature" not in env_df.columns or "humidity" not in env_df.columns:
        return None
    sub = env_df[["temperature", "humidity"]].dropna()
    if len(sub) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["temperature"],
            y=sub["humidity"],
            mode="markers",
            marker=dict(size=4, color="#3498db", opacity=0.6),
            hovertemplate="온도 %{x:.1f}°C<br>습도 %{y:.1f}%<extra></extra>",
        )
    )
    add_trendline(fig, sub["temperature"], sub["humidity"], color="orange")
    r, _ = stats.pearsonr(sub["temperature"], sub["humidity"])
    fig.update_layout(
        title=f"온도 vs 습도  (r = {r:.3f})",
        xaxis_title="온도 (°C)",
        yaxis_title="습도 (%)",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return set_font(fig)


# ── 그래프 3: 환경 박스플롯 ────────────────────────────────────────────────
def plot_env_box(env_df):
    cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
    if not cols:
        return None
    fig = make_subplots(
        rows=1, cols=len(cols),
        subplot_titles=cols,
        horizontal_spacing=0.1,
    )
    for i, col in enumerate(cols, 1):
        data = env_df[col].dropna()
        if len(data) == 0:
            continue
        fig.add_trace(
            go.Box(
                y=data,
                name=col,
                fillcolor="#3498db",
                marker=dict(color="#2980b9", size=3),
                line=dict(color="#1a5276"),
                boxpoints="outliers",
                hovertemplate="%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text=col, row=1, col=i)
    fig.update_yaxes(title_text="값", row=1, col=1)
    fig.update_layout(
        title="환경 측정값 분포 (박스플롯)",
        height=320,
        margin=dict(l=40, r=20, t=45, b=30),
    )
    return set_font(fig)


# ── 그래프 4: 생육 막대 (생중량 기준) ─────────────────────────────────────
def plot_growth_bar(growth_df):
    if "생중량" not in growth_df.columns:
        return None
    sub = growth_df[["개체번호", "생중량"]].dropna()
    if len(sub) < 2:
        return None
    sub = sub.sort_values("생중량", ascending=True)
    colors = (
        ["#e74c3c"] * 3
        + ["#bdc3c7"] * (len(sub) - 6)
        + ["#27ae60"] * 3
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sub["개체번호"],
            y=sub["생중량"],
            marker_color=colors,
            hovertemplate="개체 %{x}<br>생중량 %{y:.2f} g<extra></extra>",
        )
    )
    fig.update_layout(
        title="개체별 생중량 (하위 3개 빨간색, 상위 3개 초록색)",
        xaxis_title="개체번호",
        yaxis_title="생중량 (g)",
        height=400,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return set_font(fig)


# ── 그래프 5: 생육 산점도 (지상부 길이 vs 생중량) ────────────────────────
def plot_growth_above_scatter(growth_df):
    if "지상부 길이" not in growth_df.columns or "생중량" not in growth_df.columns:
        return None
    sub = growth_df[["지상부 길이", "생중량"]].dropna()
    if len(sub) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["지상부 길이"],
            y=sub["생중량"],
            mode="markers",
            marker=dict(size=7, color="#2ecc71", opacity=0.75),
            hovertemplate="지상부 %{x:.1f} mm<br>생중량 %{y:.2f} g<extra></extra>",
        )
    )
    add_trendline(fig, sub["지상부 길이"], sub["생중량"], color="red")
    r, _ = stats.pearsonr(sub["지상부 길이"], sub["생중량"])
    fig.update_layout(
        title=f"지상부 길이 vs 생중량  (r = {r:.3f})",
        xaxis_title="지상부 길이 (mm)",
        yaxis_title="생중량 (g)",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return set_font(fig)


# ── 그래프 6: 생육 산점도 (지하부 길이 vs 생중량) ────────────────────────
def plot_growth_below_scatter(growth_df):
    if "지하부 길이" not in growth_df.columns or "생중량" not in growth_df.columns:
        return None
    sub = growth_df[["지하부 길이", "생중량"]].dropna()
    if len(sub) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["지하부 길이"],
            y=sub["생중량"],
            mode="markers",
            marker=dict(size=7, color="#8e44ad", opacity=0.75),
            hovertemplate="지하부 %{x:.1f} mm<br>생중량 %{y:.2f} g<extra></extra>",
        )
    )
    add_trendline(fig, sub["지하부 길이"], sub["생중량"], color="red")
    r, _ = stats.pearsonr(sub["지하부 길이"], sub["생중량"])
    fig.update_layout(
        title=f"지하부 길이 vs 생중량  (r = {r:.3f})",
        xaxis_title="지하부 길이 (mm)",
        yaxis_title="생중량 (g)",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return set_font(fig)


# ── 그래프 7: 생육 종합 (지상부 vs 지하부 vs 생중량) ─────────────────────
def plot_growth_combined(growth_df):
    needed = ["지상부 길이", "지하부 길이", "생중량"]
    if not all(c in growth_df.columns for c in needed):
        return None
    sub = growth_df[needed].dropna()
    if len(sub) < 2:
        return None
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("지상부 길이 vs 생중량", "지하부 길이 vs 생중량"),
        horizontal_spacing=0.12,
    )
    r_a = stats.pearsonr(sub["지상부 길이"], sub["생중량"])[0]
    r_b = stats.pearsonr(sub["지하부 길이"], sub["생중량"])[0]
    fig.add_trace(
        go.Scatter(
            x=sub["지상부 길이"],
            y=sub["생중량"],
            mode="markers",
            marker=dict(size=6, color="#2ecc71", opacity=0.7),
            name=f"지상부 (r={r_a:.2f})",
            hovertemplate="지상부 %{x:.1f} mm<br>생중량 %{y:.2f} g<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sub["지하부 길이"],
            y=sub["생중량"],
            mode="markers",
            marker=dict(size=6, color="#8e44ad", opacity=0.7),
            name=f"지하부 (r={r_b:.2f})",
            hovertemplate="지하부 %{x:.1f} mm<br>생중량 %{y:.2f} g<extra></extra>",
        ),
        row=1,
        col=2,
    )
    add_trendline(fig, sub["지상부 길이"], sub["생중량"], color="red")
    add_trendline(fig, sub["지하부 길이"], sub["생중량"], color="red")
    fig.update_xaxes(title_text="지상부 길이 (mm)", row=1, col=1)
    fig.update_xaxes(title_text="지하부 길이 (mm)", row=1, col=2)
    fig.update_yaxes(title_text="생중량 (g)", row=1, col=1)
    fig.update_layout(
        title=f"지상부·지하부 길이와 생중량 관계  (왼쪽: r={r_a:.2f}, 오른쪽: r={r_b:.2f})",
        height=380,
        margin=dict(l=50, r=20, t=55, b=40),
    )
    return set_font(fig)


# ── 그래프 8: 생육 상위·하위 강조 점 그래프 ───────────────────────────────
def plot_growth_highlight(growth_df):
    needed = ["지상부 길이", "지하부 길이", "생중량"]
    if not all(c in growth_df.columns for c in needed):
        return None
    if "개체번호" not in growth_df.columns:
        return None
    sub = growth_df[["개체번호"] + needed].dropna()
    if len(sub) < 2:
        return None
    sub = sub.copy()
    sub["_idx"] = sub.index
    sub = sub.sort_values("생중량")
    n = len(sub)
    if n <= 6:
        top_idx = sub.index[-3:]
        bot_idx = sub.index[:3]
    else:
        top_idx = sub.index[-3:]
        bot_idx = sub.index[:3]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["지상부 길이"],
            y=sub["지하부 길이"],
            mode="markers",
            marker=dict(
                size=sub["생중량"] * 3 + 8,
                color=sub["생중량"],
                colorscale="Viridis",
                opacity=0.75,
                colorbar=dict(title="생중량 (g)", thickness=12),
            ),
            hovertemplate=(
                "개체번호 %{customdata[0]}<br>"
                "지상부 %{x:.1f} mm<br>"
                "지하부 %{y:.1f} mm<br>"
                "생중량 %{marker.color:.2f} g<extra></extra>"
            ),
            customdata=sub[["개체번호"]].values,
        )
    )
    for idx in bot_idx:
        row = sub.loc[idx]
        fig.add_trace(
            go.Scatter(
                x=[row["지상부 길이"]],
                y=[row["지하부 길이"]],
                mode="markers",
                marker=dict(size=14, color="red", symbol="x"),
                name=f"하위: 개체 {int(row['개체번호']) if pd.notna(row['개체번호']) else row['개체번호']}",
                hovertemplate="하위 개체<br>생중량 %{marker.size:.0f}<extra></extra>",
                showlegend=True,
            )
        )
    for idx in top_idx:
        row = sub.loc[idx]
        fig.add_trace(
            go.Scatter(
                x=[row["지상부 길이"]],
                y=[row["지하부 길이"]],
                mode="markers",
                marker=dict(size=14, color="green", symbol="star"),
                name=f"상위: 개체 {int(row['개체번호']) if pd.notna(row['개체번호']) else row['개체번호']}",
                hovertemplate="상위 개체<br>생중량 %{marker.size:.0f}<extra></extra>",
                showlegend=True,
            )
        )
    fig.update_layout(
        title="개체별 생육 위치 (원 크기·색상 = 생중량, × = 하위 3, ★ = 상위 3)",
        xaxis_title="지상부 길이 (mm)",
        yaxis_title="지하부 길이 (mm)",
        height=450,
        margin=dict(l=55, r=20, t=55, b=40),
    )
    return set_font(fig)


# ── 생육 요약 표 ──────────────────────────────────────────────────────────
def growth_summary_table(growth_df):
    rows = []
    num_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                if c in growth_df.columns]
    if not num_cols:
        return None
    data = growth_df[num_cols].dropna()
    if len(data) == 0:
        return None
    for col in num_cols:
        d = data[col]
        rows.append({
            "측정 항목": col,
            "평균": round(d.mean(), 2),
            "표준편차": round(d.std(), 2),
            "최대": round(d.max(), 2),
            "최소": round(d.min(), 2),
        })
    return pd.DataFrame(rows)


# ── 지표 카드 표시 (환경) ─────────────────────────────────────────────────
def show_env_metrics(env_df):
    metrics = []
    for col, label, unit in [
        ("temperature", "평균 온도", "°C"),
        ("humidity", "평균 습도", "%"),
        ("ph", "평균 pH", ""),
        ("ec", "평균 EC", "mS/cm"),
    ]:
        if col in env_df.columns:
            d = env_df[col].dropna()
            if len(d) > 0:
                metrics.append(st.metric(label=f"{label} ({unit})", value=round(d.mean(), 2)))
            else:
                st.info(f"{label}: 데이터가 없어 표시하지 못했어요.")
        else:
            st.info(f"{label}: 이 학교에 {col} 열이 없어 건너뛰었어요.")


# ── 지표 카드 표시 (생육) ─────────────────────────────────────────────────
def show_growth_metrics(growth_df):
    metrics = []
    for col, label, unit in [
        ("잎 수", "평균 잎 수", "장"),
        ("지상부 길이", "평균 지상부 길이", "mm"),
        ("지하부 길이", "평균 지하부 길이", "mm"),
        ("생중량", "평균 생중량", "g"),
    ]:
        if col in growth_df.columns:
            d = growth_df[col].dropna()
            if len(d) > 0:
                st.metric(label=f"{label} ({unit})", value=round(d.mean(), 2))
            else:
                st.info(f"{label}: 데이터가 없어 표시하지 못했어요.")
        else:
            st.info(f"{label}: 이 학교에 {col} 열이 없어 건너뛰었어요.")


# ═══════════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지 식물 생육 차이 분석", layout="wide")
st.title("극지 식물의 지상부·지하부에 따른 개체별 생육 차이 분석")

# ── 사이드바: CSV 업로드 ──────────────────────────────────────────────────
st.sidebar.header("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

use_cols = st.sidebar.multiselect(
    "📊 생육 측정 항목 (그래프에 사용할 열)",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량"],
    default=["지상부 길이", "지하부 길이", "생중량"],
)

# ── 데이터 로딩 (한 번만 읽음) ────────────────────────────────────────────
if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중…"):
        env_raw = read_csv(env_file)
    env_df = std_cols(env_raw)
else:
    env_df = pd.DataFrame()

if growth_file is not None:
    with st.spinner("생육 데이터를 읽는 중…"):
        growth_raw = read_csv(growth_file)
    growth_df = std_cols(growth_raw)
else:
    growth_df = pd.DataFrame()

# 학교명 표시 (파일명 앞부분)
school_name = ""
if env_file is not None:
    name = env_file.name
    school_name = name.split("_")[0].strip()
elif growth_file is not None:
    name = growth_file.name
    school_name = name.split("_")[0].strip()

if school_name:
    st.sidebar.success(f"✅ 학교명: {school_name}  (파일명에서 자동 추출)")

if env_df.empty and growth_df.empty:
    st.info(
        "📎 사이드바에서 환경 CSV와 생육 CSV를 업로드하면 분석이 시작돼요.\n\n"
        "• 인코딩: utf-8-sig → cp949 순서로 자동 시도해요.\n"
        "• 열 이름: 학교마다 다른 열 이름도 자동으로 표준화해요.\n"
        "• 데이터 없이도 탭은 정상 작동해요 (있는 파일만 보여줘요)."
    )

# ── 탭 ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["실험 개요", "환경 데이터", "생육 결과"])

# ──────────────── 탭 1: 실험 개요 ─────────────────────────────────────────
with tab1:
    try:
        st.header("🧪 실험 개요")
        st.markdown("""
**극지 식물**은 영하 환경에서도 생존하는 식물로, 지상부(잎·줄기)와 지하부(뿌리)의 생장 패턴이 다릅니다.
이 대시보드에서는 **개체별로 지상부 길이와 지하부 길이가 생중량(전체 무게)에 어떤 영향을 주는지** 분석합니다.

### 분석 핵심 질문
- 지하부 길이가 길수록 생중량이 큰가?
- 지상부 길이가 길수록 생중량이 큰가?
- 지상부보다 지하부가 생중량에 더 큰 영향을 미치는가?

### 통계 용어 간단 설명
| 용어 | 설명 |
|------|------|
| **평균 (mean)** | 모든 값을 더해 개수로 나눈 값. 대표적인 '중간 정도' 값입니다. |
| **표준편차 (std)** | 값들이 평균에서 얼마나 퍼져 있는지 나타내는 지표. 클수록 개체 간 차이가 큽니다. |
| **상관계수 (r)** | 두 변수의 관계를 −1 ~ +1 사이로 나타냅니다. +1에 가까울수록 함께 증가하고, −1에 가까울수록 함께 감소합니다. |
| **추세선** | 산점도 점들의 전반적인 방향을 나타내는 직선입니다. 기울기 방향(+/−)으로 관계를 볼 수 있습니다. |
        """.strip()
        )

        # 생육 데이터 요약 표
        if not growth_df.empty:
            st.markdown("### 📊 생육 데이터 요약 통계")
            gf_summary = growth_summary_table(growth_df)
            if gf_summary is not None:
                st.dataframe(gf_summary, width="stretch", use_container_width=False)
            else:
                st.info("요약할 숫자 열이 없어요.")
        else:
            st.info("📎 생육 CSV를 업로드하면 요약 통계가 여기에 표시돼요.")

        st.markdown("---")
        st.markdown(
            "### \u2bd6\uae08\uc548\uc2dc \ubc29\ubc95\n"
            "- **상관계수 (r)** 로 지상부 길이·지하부 길이와 생중량의 관계를 각각 계산합니다.\n"
            "- **추세선** (np.polyfit) 으로 관계의 방향(양의 기울기 / 음의 기울기)을 확인합니다.\n"
            "- 지상부보다 지하부 길이의 상관계수가 더 크면, '지하부가 생중량에 더 큰 영향을 미친다'고 해석합니다.\n"
            "- 생중량이 없는 학교는 **지상부 질량 + 지하부 질량**으로 자동 계산합니다."
        )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ──────────────── 탭 2: 환경 데이터 ───────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("📎 사이드바에서 환경 CSV를 업로드하면 이 탭의 내용이 표시돼요.")
        else:
            st.header("🌡️ 환경 데이터")
            st.caption(
                f"학교: {school_name if school_name else '(파일명에서 추출 불가)'}  |  "
                f"측정 기간: {env_df['time'].min():%Y-%m-%d} ~ {env_df['time'].max():%Y-%m-%d}  |  "
                f"총 {len(env_df):,}개 측정"
            )

            # 지표 카드
            st.subheader("📈 평균 환경 지표")
            show_env_metrics(env_df)

            col1, col2 = st.columns(2)
            with col1:
                fig_ts = plot_env_timeseries(env_df)
                if fig_ts is not None:
                    st.plotly_chart(fig_ts, width="stretch")
                else:
                    st.info("시간·온도·습도 열이 없어 시계열 그래프를 건너뛰었어요.")
            with col2:
                fig_scatter = plot_env_scatter(env_df)
                if fig_scatter is not None:
                    st.plotly_chart(fig_scatter, width="stretch")
                else:
                    st.info("온도·습도 열이 없어 산점도 그래프를 건너뛰었어요.")

            st.subheader("📦 환경 측정값 분포")
            fig_box = plot_env_box(env_df)
            if fig_box is not None:
                st.plotly_chart(fig_box, width="stretch")
            else:
                st.info("박스플롯을 그릴 측정 열이 없어요 (temperature, humidity, ph, ec).")

            # 환경 데이터 원본 미리보기
            st.subheader("📋 환경 데이터 원본 (일부)")
            env_display = env_df.copy()
            env_display["time"] = env_display["time"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(env_display, width="stretch", use_container_width=False)

            # CSV 다운로드
            csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ 환경 데이터 다운로드 (utf-8-sig)",
                data=csv_bytes,
                file_name=f"{school_name}_환경_정제.csv" if school_name else "환경_정제.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ──────────────── 탭 3: 생육 결과 ─────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("📎 사이드바에서 생육 CSV를 업로드하면 이 탭의 내용이 표시돼요.")
        else:
            st.header("🌿 생육 결과")

            n_individuals = len(growth_df.dropna(subset=["개체번호"]))
            st.caption(
                f"학교: {school_name if school_name else '(파일명에서 추출 불가)'}  |  "
                f"총 {n_individuals}개체"
            )

            # 지표 카드
            st.subheader("📈 개체별 생육 평균")
            show_growth_metrics(growth_df)

            # 그래프: 생중량 막대
            st.subheader("📊 개체별 생중량 비교")
            fig_bar = plot_growth_bar(growth_df)
            if fig_bar is not None:
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("생중량 열이 없어 막대 그래프를 건너뛰었어요.")

            # 그래프: 지상부 길이 vs 생중량
            st.subheader("📈 지상부 길이와 생중량 관계")
            fig_a = plot_growth_above_scatter(growth_df)
            if fig_a is not None:
                st.plotly_chart(fig_a, width="stretch")
            else:
                st.info("지상부 길이 또는 생중량 열이 없어 이 그래프를 건너뛰었어요.")

            # 그래프: 지하부 길이 vs 생중량
            st.subheader("📈 지하부 길이와 생중량 관계")
            fig_b = plot_growth_below_scatter(growth_df)
            if fig_b is not None:
                st.plotly_chart(fig_b, width="stretch")
            else:
                st.info("지하부 길이 또는 생중량 열이 없어 이 그래프를 건너뛰었어요.")

            # 종합 비교 (왼쪽 지상부, 오른쪽 지하부)
            st.subheader("🔬 종합: 지상부 vs 지하부, 생중량에 더 큰 영향은?")
            fig_comb = plot_growth_combined(growth_df)
            if fig_comb is not None:
                st.plotly_chart(fig_comb, width="stretch")
            else:
                st.info("지상부 길이·지하부 길이·생중량 열이 모두 필요해 이 그래프를 건너뛰었어요.")

            # 상위·하위 강조 그래프
            st.subheader("⭐ 생중량 상위·하위 개체 강조")
            fig_hl = plot_growth_highlight(growth_df)
            if fig_hl is not None:
                st.plotly_chart(fig_hl, width="stretch")
            else:
                st.info("지상부 길이·지하부 길이·생중량 열이 모두 필요해 이 그래프를 건너뛰었어요.")

            # 산점도 행렬 (선택된 생육 항목)
            if use_cols:
                st.subheader("📐 생육 측정 항목 간 산점도")
                avail = [c for c in use_cols if c in growth_df.columns]
                if len(avail) >= 2:
                    fig_matrix = px.scatter_matrix(
                        growth_df[avail].dropna(),
                        dimensions=avail,
                        color="생중량" if "생중량" in growth_df.columns else None,
                    )
                    fig_matrix.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=500,
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    st.plotly_chart(fig_matrix, width="stretch")
                else:
                    st.info(f"선택한 항목({', '.join(use_cols)}) 중 2개 이상이 필요해요.")

            # 생육 데이터 원본
            st.subheader("📋 생육 데이터 원본 (일부)")
            st.dataframe(growth_df, width="stretch", use_container_width=False)

            # CSV 다운로드
            csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ 생육 데이터 다운로드 (utf-8-sig)",
                data=csv_bytes,
                file_name=f"{school_name}_생육_정제.csv" if school_name else "생육_정제.csv",
                mime="text/csv",
            )

            # 결론 안내
            st.markdown("---")
            st.subheader("📌 분석 결과 해석 가이드")
            if "지상부 길이" in growth_df.columns and "지하부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                d = growth_df[["지상부 길이", "지하부 길이", "생중량"]].dropna()
                if len(d) >= 2:
                    r_a = stats.pearsonr(d["지상부 길이"], d["생중량"])[0]
                    r_b = stats.pearsonr(d["지하부 길이"], d["생중량"])[0]
                    st.markdown(
                        f"- **지상부 길이 ↔ 생중량 상관계수 r = {r_a:.3f}**\n"
                        f"- **지하부 길이 ↔ 생중량 상관계수 r = {r_b:.3f}**\n\n"
                        + (
                            f"👉 지하부 길이의 |r|({abs(r_b):.3f})이 지상부 길이의 |r|({abs(r_a):.3f})보다 크므로, "
                            "이 학교에서는 **지하부 길이가 생중량에 더 큰 영향**을 미친다고 볼 수 있어요."
                            if abs(r_b) > abs(r_a)
                            else f"👉 지상부 길이의 |r|({abs(r_a):.3f})이 지하부 길이의 |r|({abs(r_b):.3f})보다 크므로, "
                            "이 학교에서는 **지상부 길이가 생중량에 더 큰 영향**을 미친다고 볼 수 있어요."
                        )
                    )
                else:
                    st.info("상관계수를 계산하기에 데이터가 부족해요 (최소 2개체 필요).")
            else:
                st.info("지상부 길이·지하부 길이·생중량 열이 모두 있어야 상관계수를 계산할 수 있어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
