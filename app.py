import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="전기차 충전소 대시보드", page_icon="🔌", layout="wide")


# ── 1. 가상 데이터 생성 ─────────────────────────────────────────
@st.cache_data
def make_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 충전소: (지역, 유형, 급속기 대수, 완속기 대수, 수요 가중치)
    stations = {
        "판교 테크노밸리": ("경기", "업무지구", 4, 10, 1.4),
        "강남 코엑스": ("서울", "상업시설", 6, 8, 1.3),
        "잠실 아파트단지": ("서울", "주거지", 2, 16, 1.0),
        "안성 휴게소": ("경기", "고속도로", 8, 0, 1.2),
        "인천공항 T2": ("인천", "교통거점", 6, 6, 0.9),
        "해운대 마린시티": ("부산", "주거지", 2, 12, 0.8),
    }
    # 유형별 시간대 수요 패턴 (0~23시)
    h = np.arange(24)
    patterns = {
        "업무지구": np.exp(-((h - 10) ** 2) / 8) + 0.6 * np.exp(-((h - 15) ** 2) / 6),
        "상업시설": np.exp(-((h - 14) ** 2) / 12) + 0.8 * np.exp(-((h - 19) ** 2) / 6),
        "주거지": np.exp(-((h - 21) ** 2) / 6) + 0.5 * np.exp(-((h - 7) ** 2) / 3),
        "고속도로": 0.6 * np.exp(-((h - 11) ** 2) / 10) + np.exp(-((h - 17) ** 2) / 10),
        "교통거점": 0.7 * np.exp(-((h - 8) ** 2) / 8) + 0.7 * np.exp(-((h - 18) ** 2) / 8),
    }
    for k in patterns:
        patterns[k] = 0.05 + patterns[k] / patterns[k].max()

    holidays = set(pd.to_datetime(["2026-10-03", "2026-10-09"])) | \
        set(pd.date_range("2026-09-24", "2026-09-27"))  # 개천절·한글날·추석 연휴

    dates = pd.date_range("2026-08-01", "2026-10-31", freq="D")
    rows = []
    for d in dates:
        weekend = d.dayofweek >= 5
        holiday = d in holidays
        growth = 1 + 0.003 * (d - dates[0]).days  # 전기차 보급 증가 추세
        for name, (region, kind, n_fast, n_slow, w) in stations.items():
            if kind == "업무지구":
                day_f = 0.35 if (weekend or holiday) else 1.0
            elif kind == "고속도로":
                day_f = 2.2 if holiday else (1.4 if weekend else 1.0)
            else:
                day_f = 1.15 if (weekend or holiday) else 1.0
            for hour in range(24):
                for ctype, n_chargers in (("급속", n_fast), ("완속", n_slow)):
                    if n_chargers == 0:
                        continue
                    lam = 0.35 * n_chargers * w * day_f * growth * patterns[kind][hour]
                    lam *= 1.0 if ctype == "급속" else 0.45
                    for _ in range(rng.poisson(lam)):
                        if ctype == "급속":
                            kwh = float(np.clip(rng.normal(32, 9), 5, 70))
                            minutes = kwh / rng.uniform(55, 90) * 60
                            price = 347
                        else:
                            kwh = float(np.clip(rng.normal(22, 7), 5, 70))
                            minutes = kwh / 7 * 60
                            price = 295
                        rows.append((d, hour, name, region, kind, ctype, holiday,
                                     round(kwh, 1), round(minutes), round(kwh * price)))

    df = pd.DataFrame(rows, columns=["날짜", "시간", "충전소", "지역", "유형", "충전기",
                                     "공휴일", "충전량_kWh", "충전시간_분", "요금_원"])
    df["요일"] = df["날짜"].dt.dayofweek.map(dict(enumerate("월화수목금토일")))
    df["구분"] = np.where((df["날짜"].dt.dayofweek >= 5) | df["공휴일"], "주말·휴일", "평일")
    return df.drop(columns="공휴일")


df = make_data()

# ── 2. 사이드바 필터 ───────────────────────────────────────────
st.sidebar.header("필터")
regions = st.sidebar.multiselect("지역", sorted(df["지역"].unique()),
                                 default=sorted(df["지역"].unique()))
station_pool = sorted(df.loc[df["지역"].isin(regions), "충전소"].unique())
stations = st.sidebar.multiselect("충전소", station_pool, default=station_pool)
ctype = st.sidebar.radio("충전기 종류", ["전체", "급속", "완속"], horizontal=True)
min_d, max_d = df["날짜"].min().date(), df["날짜"].max().date()
period = st.sidebar.slider("기간", min_value=min_d, max_value=max_d,
                           value=(min_d, max_d), format="MM/DD")

view = df[df["충전소"].isin(stations)
          & df["날짜"].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))]
if ctype != "전체":
    view = view[view["충전기"] == ctype]

if view.empty:
    st.warning("선택한 조건에 맞는 데이터가 없어요. 필터를 다시 골라 주세요.")
    st.stop()

# ── 3. 헤더 & KPI ─────────────────────────────────────────────
st.title("🔌 전기차 충전소 이용 현황")
st.caption("2026년 8~10월 · 6개 충전소 가상 충전 기록 (추석 연휴 9/24~27 포함)")

days = view["날짜"].nunique()
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 충전 횟수", f"{len(view):,}회", f"하루 평균 {len(view)/days:,.0f}회",
          delta_color="off")
c2.metric("총 충전량", f"{view['충전량_kWh'].sum()/1000:,.1f} MWh")
c3.metric("총 요금 수입", f"{view['요금_원'].sum()/1e6:,.1f}백만 원")
c4.metric("평균 충전 시간", f"{view['충전시간_분'].mean():,.0f}분")

st.divider()

# ── 4. 탭별 차트 ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 이용 추이", "🏢 충전소 비교", "⏰ 시간대 패턴", "📋 데이터"])

with tab1:
    daily = view.groupby(["날짜", "유형"]).size().reset_index(name="충전 횟수")
    fig = px.area(daily, x="날짜", y="충전 횟수", color="유형",
                  title="일별 충전 횟수 (충전소 유형별)")
    fig.add_vrect(x0="2026-09-24", x1="2026-09-28", fillcolor="gray", opacity=0.15,
                  line_width=0, annotation_text="추석 연휴", annotation_position="top left")
    st.plotly_chart(fig, width="stretch")
    st.info("추석 연휴에는 고속도로 휴게소 충전이 크게 늘고, 업무지구 충전은 거의 멈춰요.")

with tab2:
    left, right = st.columns(2)
    by_st = (view.groupby(["충전소", "충전기"])["충전량_kWh"].sum()
                 .reset_index().sort_values("충전량_kWh"))
    left.plotly_chart(px.bar(by_st, x="충전량_kWh", y="충전소", color="충전기",
                             orientation="h", title="충전소별 총 충전량 (kWh)"),
                      width="stretch")
    wk = view.groupby(["충전소", "구분"])["날짜"].agg(["size", "nunique"]).reset_index()
    wk["하루 평균 횟수"] = wk["size"] / wk["nunique"]
    right.plotly_chart(px.bar(wk, x="충전소", y="하루 평균 횟수", color="구분",
                              barmode="group", title="평일 vs 주말·휴일 하루 평균 충전 횟수"),
                       width="stretch")

    st.plotly_chart(px.box(view, x="충전기", y="충전시간_분", color="충전기",
                           points=False, title="충전기 종류별 충전 시간 분포"),
                    width="stretch")

with tab3:
    heat = (view.groupby(["유형", "시간"]).size().unstack(fill_value=0)
                .reindex(columns=range(24), fill_value=0))
    heat = heat.div(heat.sum(axis=1), axis=0) * 100  # 유형별 하루 중 비중(%)
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Teal",
                              labels=dict(x="시간", y="충전소 유형", color="비중(%)"),
                              title="충전소 유형별 시간대 이용 비중"),
                    width="stretch")
    st.caption("업무지구는 오전 출근 직후, 주거지는 퇴근 후 밤 시간에 몰려요.")

with tab4:
    st.dataframe(view.sort_values(["날짜", "시간"]), width="stretch", height=420)
    st.download_button("CSV 다운로드", view.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ev_charging_filtered.csv", mime="text/csv")
