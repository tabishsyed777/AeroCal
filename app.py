from __future__ import annotations

import pandas as pd
import streamlit as st

from aerocal.calibration import FEATURES, run_site_holdout
from aerocal.data import make_demo_data, validate_collocations

st.set_page_config(page_title="AeroCal", page_icon="🌫️", layout="wide")

st.title("AeroCal")
st.caption("A site-aware calibration workbench for low-cost PM2.5 sensors")
st.info(
    "Educational prototype. Demo readings are synthetic. Results are not regulatory "
    "measurements, health advice, or validation of a real sensor."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload collocation CSV", type=["csv"])
    st.caption("Required: station_id, sensor_pm25, reference_pm25, humidity, temperature.")
    st.divider()
    coverage_pct = st.slider("Prediction interval target", 50, 99, 90, 1)
    st.caption("Intervals use residuals from a separate calibration station.")

if uploaded is None:
    data = make_demo_data()
    st.caption("Showing a reproducible synthetic demo with six simulated stations.")
else:
    try:
        source = pd.read_csv(uploaded)
        data, issues = validate_collocations(source)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    if issues:
        for issue in issues:
            st.error(issue)
        st.stop()
    st.caption(f"Loaded {len(data):,} complete rows from {data['station_id'].nunique()} stations.")

stations = sorted(data["station_id"].astype(str).unique())
if len(stations) < 3:
    st.error("At least three stations are needed for site-separated evaluation.")
    st.stop()

col_a, col_b = st.columns([2, 1])
with col_a:
    test_station = st.selectbox("Hold out a test station", stations, index=len(stations) - 1)
with col_b:
    st.metric("Stations", len(stations))

if st.button("Run calibration", type="primary", use_container_width=True):
    try:
        result = run_site_holdout(data, test_station, coverage=coverage_pct / 100)
    except Exception as exc:
        st.error(f"Calibration could not run: {exc}")
        st.stop()

    st.subheader("Held-out station results")
    st.write(
        f"Model fit on **{', '.join(result['fit_stations'])}** · "
        f"interval calibrated on **{result['calibration_station']}** · "
        f"test station **{result['test_station']}**"
    )
    raw, corrected = result["scores"]["Raw sensor"], result["scores"]["ML corrected"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw sensor MAE", f"{raw['MAE']:.2f} µg/m³")
    m2.metric("Corrected MAE", f"{corrected['MAE']:.2f} µg/m³", f"{raw['MAE'] - corrected['MAE']:+.2f} vs raw")
    m3.metric("Test interval coverage", f"{result['observed_coverage']:.1%}", f"{result['target_coverage']:.0%} target")
    m4.metric("Mean interval width", f"{result['interval_width']:.2f} µg/m³")

    st.subheader("Error metrics")
    metrics = pd.DataFrame(result["scores"]).T
    st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)

    scored = result["predictions"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Reference vs prediction**")
        chart = scored[["reference_pm25", "raw_prediction", "corrected_prediction"]].rename(
            columns={"reference_pm25": "Reference", "raw_prediction": "Raw sensor", "corrected_prediction": "ML corrected"}
        )
        st.line_chart(chart, height=300)
    with right:
        st.markdown("**Prediction interval on held-out station**")
        interval = scored[["reference_pm25", "corrected_prediction", "interval_low", "interval_high"]].copy()
        if "timestamp" in scored and scored["timestamp"].notna().any():
            interval.index = scored["timestamp"]
        interval = interval.drop(columns="reference_pm25").rename(
            columns={"corrected_prediction": "Corrected", "interval_low": "Interval low", "interval_high": "Interval high"}
        )
        st.line_chart(interval, height=300)

    st.subheader("Reading the result")
    st.write(
        "This is a station-level holdout: the model never sees the selected test site during fitting. "
        "The interval width is calibrated on another site. Coverage on a different site may fall below "
        "the target when site conditions shift."
    )
    with st.expander("Show held-out rows"):
        st.dataframe(scored, use_container_width=True)
else:
    st.markdown("### How to use")
    st.write("Pick a station to hold out, set a target interval level, then run calibration.")
    st.write("With your own data, each row should be a paired low-cost sensor and reference-monitor measurement.")

with st.expander("Data and method notes"):
    st.write(
        "The model uses sensor PM2.5, relative humidity, and temperature. The train, calibration, "
        "and test splits are grouped by station to reduce leakage from repeated readings at the same site."
    )
    st.write(
        "Split-conformal residual intervals assume calibration and future test errors are sufficiently "
        "exchangeable. Cross-site shifts can invalidate nominal coverage. The displayed test coverage "
        "is a diagnostic, not a guarantee."
    )
    st.write("Model inputs:", ", ".join(FEATURES))
