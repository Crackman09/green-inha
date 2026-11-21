import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


@st.cache_data
def read_csv(file):
    """utf-8-sig로 읽고 실패하면 cp949로 다시 읽기"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    """열 이름을 표준 이름으로 통일"""
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


def safe_numeric(series):
    """NaN 제거 후 충분한 값이 있으면 Series 반환, 없으면 None"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) >= 2:
        return s
    return None


def plotly_font(fig):
    """Plotly 한글 폰트 적용"""
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig


def bar_chart(df, x_col, y_col, title, color_col=None):
    """막대 그래프 (개체번호는 문자열로 그대로 표시)"""
    d = df[[x_col, y_col]].dropna()
    if len(d) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d[x_col],
            y=d[y_col],
            marker_color="#4CAF50",
            name=y_col,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return plotly_font(fig)


def line_chart(df, time_col, value_col, title, color="#2196F3"):
    """시계열 꺾은선 그래프"""
    d = df[[time_col, value_col]].dropna()
    if len(d) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d[time_col],
            y=d[value_col],
            mode="lines",
            name=value_col,
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title=value_col,
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return plotly_font(fig)


def scatter_with_trend(x, y, title, xlabel, ylabel):
    """산점도 + 추세선 (np.polyfit)"""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["x"],
            y=d["y"],
            mode="markers",
            marker=dict(color="#FF9800", size=8),
            name="데이터",
        )
    )
    slope, intercept = np.polyfit(d["x"], d["y"], 1)
    x_line = np.linspace(d["x"].min(), d["x"].max(), 100)
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"추세선 (r={np.corrcoef(d['x'], d['y'])[0,1]:.3f})",
            line=dict(color="#E91E63", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return plotly_font(fig)


def boxplot_fig(df, y_col, title):
    """박스플롯"""
    d = df[[y_col]].dropna()
    if len(d) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=d[y_col],
            fillcolor="#4CAF50",
            line=dict(color="#2E7D32"),
            name=y_col,
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title=y_col,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return plotly_font(fig)


def download_csv(df, label):
    """CSV 다운로드 버튼 (utf-8-sig)"""
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"{label} 다운로드 (CSV)",
        data=csv,
        file_name=f"{label}.csv",
        mime="text/csv",
    )


def get_school_name(filename):
    """파일명에서 학교명 추출"""
    if filename:
        base = os.path.basename(filename)
        name = base.replace(".csv", "").strip()
        return name
    return "업로드된 파일"


# ─── 사이드바 ───────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

growth_options = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_measure = st.sidebar.selectbox(
    "생육 측정 항목 선택 (강조 표시용)",
    growth_options,
    index=growth_options.index("생중량") if "생중량" in growth_options else 0,
)

school_name = get_school_name(env_file.name if env_file else growth_file.name if growth_file else "")

# ─── 데이터 로드 ────────────────────────────────────────────────────
env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중..."):
        env_raw = read_csv(env_file)
        env_df = std_cols(env_raw)

if growth_file is not None:
    with st.spinner("생육 결과 데이터를 읽는 중..."):
        growth_raw = read_csv(growth_file)
        growth_df = std_cols(growth_raw)

st.sidebar.markdown(f"### 🏫 {school_name}")
st.sidebar.caption("CSV 파일을 업로드하면 분석이 시작됩니다.")


# ─── Tab 1: 실험 개요 ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown(
            """
**연구 배경**
극지식물은 낮은 온도와 강한 바람, 적은 수분 등 혹독한 환경에서 자라는 식물입니다.
우리 학교에서 재배 중인 극지식물 개체들을 관찰하여, 개체별로 생육 상태가 어떻게 다른지 비교·분석합니다.

**연구 목적**
- 개체별 생중량 분포를 확인하고 평균·표준편차·변동계수(CV)를 이해한다.
- 잎 수, 지상부 길이와 생중량 사이의 상관관계를 알아본다.
- 생중량이 큰 개체와 작은 개체를 비교해 생육 차이의 원인을 탐색한다.

> 💡 **통계 용어 간단 설명**
> - **평균**: 모든 값을 더해 개수로 나눈 대표값
> - **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 값 (클수록 편차가 큼)
> - **CV(변동계수)**: 표준편차를 평균으로 나눈 값. 단위가 다른 데이터끼리도 비교 가능
> - **상관계수(r)**: 두 변수가 함께 움직이는 정도 (-1 ~ 1). 1에 가까울수록 함께 증가
"""
        )

        if len(env_df) == 0 and len(growth_df) == 0:
            st.info("아직 CSV 파일이 업로드되지 않았습니다. 사이드바에서 파일을 업로드해 주세요.")
        else:
            # 데이터 요약
            st.header("데이터 요약")

            # 개체 수
            n_growth = len(growth_df)
            # 환경 측정 기간
            env_period = ""
            n_env = len(env_df)
            if "time" in env_df.columns and n_env > 0:
                t = env_df["time"].dropna()
                if len(t) > 0:
                    env_period = f"{t.min().strftime('%Y-%m-%d')} ~ {t.max().strftime('%Y-%m-%d')}"

            summary_rows = []
            summary_rows.append(
                {
                    "항목": "총 개체 수",
                    "값": f"{n_growth}개체",
                }
            )
            if env_period:
                summary_rows.append(
                    {"항목": "환경 측정 기간", "값": env_period}
                )
            if "temperature" in env_df.columns:
                summary_rows.append(
                    {"항목": "측정 항목 (환경)", "값": "온도, 습도, pH, EC"}
                )
            else:
                summary_rows.append(
                    {"항목": "측정 항목 (환경)", "값": "일부 항목만 측정됨"}
                )
            summary_rows.append(
                {"항목": "측정 항목 (생육)", "값": "잎 수, 지상부 길이, 지하부 길이, 생중량"}
            )
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, width="stretch", use_container_width=False)

            # 주요 지표 카드 4개
            st.header("주요 지표")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "총 개체수",
                f"{n_growth}개체",
            )

            if "temperature" in env_df.columns and n_env > 0:
                temp_vals = safe_numeric(env_df["temperature"])
                avg_temp = temp_vals.mean() if temp_vals is not None else None
                col2.metric(
                    "평균 온도",
                    f"{avg_temp:.1f}°C" if avg_temp is not None else "데이터 없음",
                )
            else:
                col2.metric("평균 온도", "데이터 없음")

            if "humidity" in env_df.columns and n_env > 0:
                hum_vals = safe_numeric(env_df["humidity"])
                avg_hum = hum_vals.mean() if hum_vals is not None else None
                col3.metric(
                    "평균 습도",
                    f"{avg_hum:.1f}%" if avg_hum is not None else "데이터 없음",
                )
            else:
                col3.metric("평균 습도", "데이터 없음")

            if "생중량" in growth_df.columns and n_growth > 0:
                bio_vals = safe_numeric(growth_df["생중량"])
                avg_bio = bio_vals.mean() if bio_vals is not None else None
                col4.metric(
                    "평균 생중량",
                    f"{avg_bio:.2f}g" if avg_bio is not None else "데이터 없음",
                )
            else:
                col4.metric("평균 생중량", "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ─── Tab 2: 환경 데이터 ──────────────────────────────────────────────
with tab2:
    try:
        if len(env_df) == 0:
            st.info("환경 CSV 파일을 사이드바에 업로드해 주세요.")
        else:
            st.header("🌡️ 환경 데이터 분석")

            # 환경 요약 표
            st.subheader("환경 요약 (평균 · 최소 · 최대)")
            env_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns
            ]
            if env_cols:
                env_summary = []
                for c in env_cols:
                    vals = safe_numeric(env_df[c])
                    if vals is not None:
                        env_summary.append(
                            {
                                "측정 항목": c,
                                "평균": round(vals.mean(), 2),
                                "최소": round(vals.min(), 2),
                                "최대": round(vals.max(), 2),
                            }
                        )
                if env_summary:
                    env_summary_df = pd.DataFrame(env_summary)
                    # 한글 표시명
                    name_map = {
                        "temperature": "온도(°C)",
                        "humidity": "습도(%)",
                        "ph": "pH",
                        "ec": "EC(mS/cm)",
                    }
                    env_summary_df["측정 항목"] = env_summary_df["측정 항목"].map(
                        name_map
                    )
                    st.dataframe(env_summary_df, width="stretch", use_container_width=False)
                else:
                    st.info("환경 데이터에서 숫자 값을 찾을 수 없습니다.")
            else:
                st.info("환경 데이터에 인식된 측정 항목이 없습니다.")

            # 날짜별 시계열
            st.subheader("날짜별 시계열")

            if "time" in env_df.columns and "temperature" in env_df.columns:
                daily_temp = (
                    env_df[["time", "temperature"]]
                    .dropna()
                    .copy()
                )
                if len(daily_temp) > 0:
                    daily_temp["date"] = daily_temp["time"].dt.date
                    daily_avg = (
                        daily_temp.groupby("date")["temperature"]
                        .mean()
                        .reset_index()
                    )
                    daily_avg["date"] = pd.to_datetime(daily_avg["date"])
                    fig = line_chart(
                        daily_avg, "date", "temperature", "일별 평균 온도 변화", "#2196F3"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("온도 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                else:
                    st.info("온도 데이터가 없습니다.")
            else:
                st.info("온도 열이 없어 온도 시계열을 표시할 수 없습니다.")

            if "time" in env_df.columns and "humidity" in env_df.columns:
                daily_hum = (
                    env_df[["time", "humidity"]].dropna().copy()
                )
                if len(daily_hum) > 0:
                    daily_hum["date"] = daily_hum["time"].dt.date
                    daily_avg_h = (
                        daily_hum.groupby("date")["humidity"].mean().reset_index()
                    )
                    daily_avg_h["date"] = pd.to_datetime(daily_avg_h["date"])
                    fig = line_chart(
                        daily_avg_h, "date", "humidity", "일별 평균 습도 변화", "#9C27B0"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("습도 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                else:
                    st.info("습도 데이터가 없습니다.")
            else:
                st.info("습도 열이 없어 습도 시계열을 표시할 수 없습니다.")

            if "time" in env_df.columns and ("ph" in env_df.columns or "ec" in env_df.columns):
                env_plot_cols = [c for c in ["ph", "ec"] if c in env_df.columns]
                for c in env_plot_cols:
                    daily_c = (
                        env_df[["time", c]].dropna().copy()
                    )
                    if len(daily_c) > 0:
                        daily_c["date"] = daily_c["time"].dt.date
                        daily_avg_c = (
                            daily_c.groupby("date")[c].mean().reset_index()
                        )
                        daily_avg_c["date"] = pd.to_datetime(daily_avg_c["date"])
                        color = "#4CAF50" if c == "ph" else "#FF9800"
                        fig = line_chart(
                            daily_avg_c, "date", c, f"일별 평균 {c.upper()} 변화", color
                        )
                        if fig is not None:
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.info(f"{c} 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
                    else:
                        st.info(f"{c} 데이터가 없습니다.")
            else:
                st.info("pH·EC 열이 없어 해당 시계열을 표시할 수 없습니다.")

            # 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch", use_container_width=False)
                download_csv(env_df, f"{school_name}_환경데이터")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ─── Tab 3: 생육 결과 ──────────────────────────────────────────────
with tab3:
    try:
        if len(growth_df) == 0:
            st.info("생육 결과 CSV 파일을 사이드바에 업로드해 주세요.")
        else:
            st.header("📊 생육 결과 분석")

            # 🥇 핵심 결과 카드
            st.subheader("🥇 핵심 결과 카드")
            if "생중량" in growth_df.columns:
                bio_vals = safe_numeric(growth_df["생중량"])
                if bio_vals is not None:
                    avg_bio = bio_vals.mean()
                    max_idx = bio_vals.idxmax()
                    min_idx = bio_vals.idxmin()
                    # 개체번호는 원래 값 그대로 표시
                    max_id = growth_df.loc[max_idx, "개체번호"]
                    min_id = growth_df.loc[min_idx, "개체번호"]
                    max_val = bio_vals.max()
                    min_val = bio_vals.min()

                    card_col1, card_col2, card_col3, card_col4 = st.columns(4)
                    card_col1.metric("평균 생중량", f"{avg_bio:.2f} g")
                    card_col2.metric(
                        "최대 생중량", f"{max_val:.2f} g", delta=f"개체 {max_id}"
                    )
                    card_col3.metric(
                        "최소 생중량", f"{min_val:.2f} g", delta=f"개체 {min_id}"
                    )
                    std_bio = bio_vals.std()
                    cv_bio = (std_bio / avg_bio * 100) if avg_bio != 0 else 0
                    card_col4.metric(
                        "표준편차 / CV",
                        f"σ={std_bio:.2f}g, CV={cv_bio:.1f}%",
                    )

                    st.caption(
                        "💡 CV(변동계수) = 표준편차 ÷ 평균 × 100. "
                        "값이 클수록 개체 간 차이가 크다는 뜻입니다."
                    )
                else:
                    st.info("생중량 데이터가 충분하지 않습니다.")
            else:
                st.info("생중량 열이 없어 핵심 결과 카드를 표시할 수 없습니다.")

            # 개체별 생육 비교 (막대 그래프)
            st.subheader("개체별 생육 비교")

            measure_map = {
                "잎 수": "잎 수",
                "지상부 길이": "지상부 길이",
                "지하부 길이": "지하부 길이",
                "생중량": "생중량",
            }

            # 선택된 측정 항목 먼저 표시
            sel_col = measure_map.get(selected_measure)
            if sel_col and sel_col in growth_df.columns:
                vals = safe_numeric(growth_df[sel_col])
                if vals is not None:
                    fig = bar_chart(
                        growth_df,
                        "개체번호",
                        sel_col,
                        f"개체별 {sel_col}",
                        None,
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

            # 4개 항목 한 번에 표시 (서브플롯)
            all_measures = [
                m for m in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                if m in growth_df.columns
            ]
            if len(all_measures) >= 1:
                sub_fig = make_subplots(
                    rows=len(all_measures),
                    cols=1,
                    subplot_titles=all_measures,
                    vertical_spacing=0.12,
                )
                for i, m in enumerate(all_measures):
                    vals = safe_numeric(growth_df[m])
                    if vals is not None:
                        ids = growth_df["개체번호"].values
                        sub_fig.add_trace(
                            go.Bar(
                                x=ids,
                                y=vals.values,
                                name=m,
                                marker_color="#4CAF50",
                            ),
                            row=i + 1,
                            col=1,
                        )
                sub_fig.update_layout(
                    height=250 * len(all_measures) + 100,
                    margin=dict(l=40, r=20, t=60, b=40),
                    showlegend=False,
                )
                sub_fig = plotly_font(sub_fig)
                st.plotly_chart(sub_fig, width="stretch")

            # 생중량 분포 (박스플롯)
            st.subheader("생중량 분포")
            if "생중량" in growth_df.columns:
                fig = boxplot_fig(growth_df, "생중량", "개체별 생중량 분포 (박스플롯)")
                if fig is not None:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("생중량 데이터가 충분하지 않아 박스플롯을 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 분포 그래프를 표시할 수 없습니다.")

            # 상관관계 분석
            st.subheader("상관관계 분석 (산점도 + 추세선)")

            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                x = safe_numeric(growth_df["잎 수"])
                y = safe_numeric(growth_df["생중량"])
                if x is not None and y is not None and len(x) == len(y) and len(x) >= 2:
                    r = np.corrcoef(x, y)[0, 1]
                    st.markdown(f"**잎 수 vs 생중량**: 상관계수 r = {r:.3f}")
                    fig = scatter_with_trend(
                        x, y, "잎 수 vs 생중량 (추세선 포함)", "잎 수", "생중량(g)"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info("잎 수 vs 생중량 데이터가 충분하지 않습니다.")
            else:
                st.info("잎 수 또는 생중량 열이 없어 상관관계 분석을 표시할 수 없습니다.")

            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                x = safe_numeric(growth_df["지상부 길이"])
                y = safe_numeric(growth_df["생중량"])
                if x is not None and y is not None and len(x) == len(y) and len(x) >= 2:
                    r = np.corrcoef(x, y)[0, 1]
                    st.markdown(f"**지상부 길이 vs 생중량**: 상관계수 r = {r:.3f}")
                    fig = scatter_with_trend(
                        x,
                        y,
                        "지상부 길이 vs 생중량 (추세선 포함)",
                        "지상부 길이(mm)",
                        "생중량(g)",
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info("지상부 길이 vs 생중량 데이터가 충분하지 않습니다.")
            else:
                st.info("지상부 길이 또는 생중량 열이 없어 상관관계 분석을 표시할 수 없습니다.")

            # 생중량 상위 25% vs 하위 25% 비교
            st.subheader("생중량 상위 25% vs 하위 25% 비교")
            if "생중량" in growth_df.columns:
                bio_vals = safe_numeric(growth_df["생중량"])
                if bio_vals is not None and len(bio_vals) >= 4:
                    q75 = bio_vals.quantile(0.75)
                    q25 = bio_vals.quantile(0.25)
                    top = growth_df[growth_df["생중량"] >= q75].copy()
                    bottom = growth_df[growth_df["생중량"] <= q25].copy()

                    comp_rows = []
                    for _, row in top.iterrows():
                        comp_rows.append(
                            {
                                "구분": "상위 25%",
                                "개체번호": row["개체번호"],
                                "생중량(g)": round(row["생중량"], 2),
                                "잎 수": row.get("잎 수", ""),
                                "지상부 길이(mm)": row.get("지상부 길이", ""),
                                "지하부 길이(mm)": row.get("지하부 길이", ""),
                            }
                        )
                    for _, row in bottom.iterrows():
                        comp_rows.append(
                            {
                                "구분": "하위 25%",
                                "개체번호": row["개체번호"],
                                "생중량(g)": round(row["생중량"], 2),
                                "잎 수": row.get("잎 수", ""),
                                "지상부 길이(mm)": row.get("지상부 길이", ""),
                                "지하부 길이(mm)": row.get("지하부 길이", ""),
                            }
                        )
                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_df, width="stretch", use_container_width=False)
                else:
                    st.info("개체 수가 충분하지 않아 상위·하위 25% 비교를 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 상위·하위 비교를 표시할 수 없습니다.")

            # 원본 테이블 + 다운로드
            with st.expander("📄 생육 결과 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", use_container_width=False)
                download_csv(growth_df, f"{school_name}_생육결과데이터")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ─── 푸터 ─────────────────────────────────────────────────────────────
st.caption("© 극지식물 생육 분석 대시보드 — Streamlit 기반")
