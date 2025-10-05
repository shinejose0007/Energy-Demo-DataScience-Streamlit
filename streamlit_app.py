import streamlit as st
import pandas as pd
from pathlib import Path
from scripts import data_connectors, forecasting, optimization, abtest, plot_utils, anomaly_detection, plot_dashboard

from scripts import data_connectors, anomaly_detection
df = data_connectors.load_from_csv("sample_data/consumption.csv")
res = anomaly_detection.combined_anomalies(df, window_days=7, z_thresh=3.0, top_n=10, contamination=0.01)
joined = res["joined"]
print("Counts:", res["summary"])
print(joined.head(10).to_string())


st.set_page_config(page_title="Energy — WOW Dashboard", layout="wide")
st.title("Energy — WOW Dashboard")

st.sidebar.header("Controls")
use_sample = st.sidebar.checkbox("Use sample data (included)", value=True)
upload_consumption = st.sidebar.file_uploader("Upload consumption CSV", type=["csv"])
upload_ev = st.sidebar.file_uploader("Upload EV profile CSV", type=["csv"])
action = st.sidebar.selectbox("Action", ["Dashboard", "Forecast", "Optimize", "Anomaly detect", "A/B test", "Explainability"])
run_button = st.sidebar.button("Run")

DATA_DIR = Path("sample_data")

def load_consumption(use_sample, uploaded):
    if use_sample and (uploaded is None):
        return data_connectors.load_from_csv(DATA_DIR / "consumption.csv")
    if uploaded is not None:
        return data_connectors.load_from_csv(uploaded)
    return None

def load_ev(use_sample, uploaded):
    if use_sample and (uploaded is None):
        return pd.read_csv(DATA_DIR / "ev_profile.csv")
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return None

consumption_df = load_consumption(use_sample, upload_consumption)
ev_df = load_ev(use_sample, upload_ev)

if consumption_df is None:
    st.warning("No consumption data available.")
else:
    st.subheader("Consumption Data"); st.dataframe(consumption_df.head(48))

if action == "Dashboard" and run_button:
    st.header("Heatmap: hour vs day")
    fig = plot_dashboard.heatmap_hour_day(consumption_df)
    st.plotly_chart(fig, use_container_width=True)
    st.header("Pareto: feature importance (example)")
    # example importance (fake) - in real use replace by real importance df
    imp_df = pd.DataFrame({"importance":[0.6,0.25,0.1,0.05]}, index=["lag24","lag1","hour","dayofweek"])
    fig_p = plot_dashboard.pareto_feature_importance(imp_df)
    st.plotly_chart(fig_p, use_container_width=True)
    st.header("Forecast quick")
    model, mae = forecasting.train_forecast(consumption_df); st.write(f"MAE: {mae:.3f}")
    fc = forecasting.forecast_next(consumption_df, model, n_steps=24)
    fig2 = plot_utils.plot_price_and_forecast_plotly(consumption_df, fc); st.plotly_chart(fig2, use_container_width=True)
    st.header("Optimization + animated & cumulative plots (requires EV profile)")
    price_series = consumption_df.set_index("datetime")["price_eur_per_kwh"]
    try:
        evp = ev_df.copy()
        if "arrival_dt" not in evp.columns:
            base_day = pd.to_datetime(price_series.index[0]).normalize()
            evp["arrival_dt"] = base_day + pd.to_timedelta(evp["arrival_hour"], unit="h")
            evp["departure_dt"] = base_day + pd.to_timedelta(evp["departure_hour"], unit="h")
        schedules = optimization.schedule_multiple_evs(price_series, evp, household_cap_kw=11.0)
        frames = []
        for ev, df in schedules.items():
            df = df.copy(); df["ev_id"]=ev; frames.append(df)
        sched_all = pd.concat(frames, ignore_index=True)
        st.subheader("Animated Price vs Charging")
        anim = plot_dashboard.animated_price_charging(sched_all); st.plotly_chart(anim, use_container_width=True)
        st.subheader("Cumulative energy vs cost")
        cum = plot_dashboard.cumulative_energy_cost(sched_all); st.plotly_chart(cum, use_container_width=True)
        st.subheader("Waterfall (cost breakdown)")
        breakdown = [{"label": ev, "value": float(df["cost_eur"].sum())} for ev, df in schedules.items()]
        wf = plot_dashboard.waterfall_energy_cost(breakdown); st.plotly_chart(wf, use_container_width=True)
    except Exception as e:
        st.error(f"Optimization/dashboard plots failed: {e}")

if action == "Forecast" and run_button:
    st.info("Training forecasting model..."); model, mae = forecasting.train_forecast(consumption_df); st.success(f"Model trained. MAE = {mae:.3f}")
    n_steps = st.sidebar.number_input("Forecast horizon (hours)", min_value=1, max_value=168, value=24)
    fc = forecasting.forecast_next(consumption_df, model, n_steps=n_steps); st.subheader("Forecast (next periods)"); st.dataframe(fc); fig = plot_utils.plot_price_and_forecast_plotly(consumption_df, fc); st.plotly_chart(fig, use_container_width=True)

if action == "Optimize" and run_button:
    st.info("Running optimization..."); price_series = consumption_df.set_index("datetime")["price_eur_per_kwh"]; evp = ev_df.copy()
    if "arrival_dt" not in evp.columns:
        base_day = pd.to_datetime(price_series.index[0]).normalize()
        evp["arrival_dt"] = base_day + pd.to_timedelta(evp["arrival_hour"], unit="h")
        evp["departure_dt"] = base_day + pd.to_timedelta(evp["departure_hour"], unit="h")
    schedules = optimization.schedule_multiple_evs(price_series, evp, household_cap_kw=11.0)
    for ev, sched in schedules.items():
        st.subheader(f"Schedule for {ev}"); st.dataframe(sched); st.download_button(f"Download schedule_{ev}", data=sched.to_csv(index=False).encode("utf-8"), file_name=f"schedule_{ev}.csv", mime="text/csv")

#if action == "Anomaly detect" and run_button:
    #st.info("Detecting anomalies..."); az = anomaly_detection.seasonal_zscore(consumption_df); st.dataframe(az.loc[az["is_anomaly"], ["datetime","consumption_kwh","z_score"]].head(200)); st.download_button("Download anomalies", data=az.loc[az["is_anomaly"]].to_csv(index=False).encode("utf-8"), file_name="anomalies.csv", mime="text/csv")

# Anomaly detect (replace previous block)
if action == "Anomaly detect" and run_button:
    st.info("Detecting anomalies...")
    z_thresh = st.sidebar.slider("MAD z-threshold", min_value=0.5, max_value=8.0, value=3.0, step=0.1)
    window_days = st.sidebar.slider("Window days (per hour)", min_value=1, max_value=30, value=7, step=1)
    top_n = st.sidebar.number_input("Top N suspicious rows", min_value=5, max_value=200, value=20)
    contamination = st.sidebar.slider("Iforest contamination", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

    res = anomaly_detection.combined_anomalies(consumption_df, window_days=window_days, z_thresh=z_thresh, top_n=top_n, contamination=contamination)
    joined = res["joined"]
    summary = res["summary"]

    st.write(f"MAD flagged: {summary['mad_n_flagged']}, IsolationForest flagged: {summary['iforest_n_flagged']}, Combined: {summary['combined_n_flagged']}")

    # If there are combined flagged anomalies — show them
    if summary["combined_n_flagged"] > 0:
        st.subheader("Flagged anomalies (combined)")
        st.dataframe(joined.loc[joined["is_anomaly_combined"]].sort_values("datetime").reset_index(drop=True))
        st.download_button("Download flagged anomalies", joined.loc[joined["is_anomaly_combined"]].to_csv(index=False).encode("utf-8"), file_name="anomalies_flagged.csv", mime="text/csv")
    else:
        st.warning("No rows exceeded the chosen threshold. Showing top suspicious rows by |z_score|.")
        topn = joined.reindex(joined["z_score"].abs().sort_values(ascending=False).index).head(top_n)
        st.subheader(f"Top {top_n} suspicious rows")
        st.dataframe(topn[["datetime", "consumption_kwh", "z_score", "is_anomaly", "is_anomaly_iforest"]].reset_index(drop=True))
        st.download_button("Download top suspicious rows", topn.to_csv(index=False).encode("utf-8"), file_name="anomalies_topn.csv", mime="text/csv")

    # Histogram of |z_score|
    try:
        import plotly.express as px
        df_plot = joined.copy()
        df_plot["abs_z"] = df_plot["z_score"].abs()
        fig = px.histogram(df_plot, x="abs_z", nbins=60, title="Distribution of |MAD z-score|")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write(joined["z_score"].describe())

    # Quick fail-safe debug lines (visible always) to help debugging if nothing displays:
    st.caption(f"Rows in joined: {len(joined)}. NaNs in z_score: {joined['z_score'].isna().sum()}. Sample head:")
    st.write(joined.head(8))


if action == "A/B test" and run_button:
    st.info("Running A/B tests"); data_A, data_B = abtest.simulate_ab_test(); res_classic = abtest.analyze_ab_test(data_A, data_B); st.subheader("Classic results"); st.write(res_classic)

if action == "Explainability" and run_button:
    st.info("Permutation importance (simple)"); model, mae = forecasting.train_forecast(consumption_df); df_fe = forecasting.feature_engineer(consumption_df); X = df_fe[["hour","dayofweek","lag1","lag24"]]; y = df_fe["consumption_kwh"]
    import numpy as np
    def perm_importance(model,X,y,n_repeats=5):
        rng = np.random.RandomState(42); base = ((y-model.predict(X))**2).mean(); imps={}
        for col in X.columns:
            scores=[] 
            for _ in range(n_repeats):
                Xp=X.copy(); Xp[col]=rng.permutation(Xp[col].values); scores.append(((y-model.predict(Xp))**2).mean()-base)
            imps[col]=np.mean(scores)
        import pandas as pd; return pd.DataFrame.from_dict(imps,orient="index",columns=["importance"]).sort_values("importance",ascending=False)
    imp = perm_importance(model,X,y,n_repeats=10); st.dataframe(imp); st.download_button("Download importance", data=imp.to_csv(index=True).encode("utf-8"), file_name="feature_importance.csv", mime="text/csv")
