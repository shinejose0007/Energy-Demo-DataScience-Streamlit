import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def heatmap_hour_day(df, value_col="consumption_kwh"):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["day"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["datetime"].dt.hour
    pivot = df.pivot_table(index="hour", columns="day", values=value_col, aggfunc="mean").fillna(0)
    fig = px.imshow(pivot, labels=dict(x="day", y="hour", color=value_col), aspect="auto", origin="lower", title="Heatmap: Hour vs Day (mean consumption)")
    fig.update_layout(yaxis=dict(dtick=1))
    return fig

def pareto_feature_importance(imp_df):
    imp = imp_df.copy().reset_index().rename(columns={"index":"feature"})
    imp = imp.sort_values("importance", ascending=False)
    imp["cumulative"] = imp["importance"].cumsum() / imp["importance"].sum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=imp["feature"], y=imp["importance"], name="importance"), secondary_y=False)
    fig.add_trace(go.Scatter(x=imp["feature"], y=imp["cumulative"], name="cumulative", mode="lines+markers"), secondary_y=True)
    fig.update_yaxes(title_text="importance", secondary_y=False)
    fig.update_yaxes(title_text="cumulative %", secondary_y=True, tickformat="0%")
    fig.update_layout(title="Pareto: Feature Importance", xaxis_tickangle=-45)
    return fig

def waterfall_energy_cost(breakdown):
    # breakdown: list of dicts {"label":..., "value":...}
    x = [b["label"] for b in breakdown]
    y = [b["value"] for b in breakdown]
    measures = ["relative"] * len(y)
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=measures))
    fig.update_layout(title="Waterfall: Energy Cost Breakdown")
    return fig

def animated_price_charging(schedule_df):
    # schedule_df must contain: datetime, ev_id, charge_kw, price (or price_eur_per_kwh)
    df = schedule_df.copy()
    if "price" not in df.columns and "price_eur_per_kwh" in df.columns:
        df["price"] = df["price_eur_per_kwh"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df["frame"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
    fig = px.scatter(df, x="price", y="charge_kw", animation_frame="frame", color="ev_id", size="charge_kw", title="Animated: Price vs Charging over time", range_x=[df["price"].min()*0.9, df["price"].max()*1.1])
    return fig

def cumulative_energy_cost(schedule_df):
    df = schedule_df.copy()
    df = df.sort_values("datetime")
    if "energy_kwh" not in df.columns:
        df["energy_kwh"] = df["charge_kw"] * 1.0
    if "cost_eur" not in df.columns:
        df["cost_eur"] = df["energy_kwh"] * df.get("price", df.get("price_eur_per_kwh", 0))
    df["cum_energy"] = df.groupby("ev_id")["energy_kwh"].cumsum()
    df["cum_cost"] = df.groupby("ev_id")["cost_eur"].cumsum()
    fig = go.Figure()
    for ev, grp in df.groupby("ev_id"):
        fig.add_trace(go.Scatter(x=grp["cum_energy"], y=grp["cum_cost"], mode="lines+markers", name=f"{ev}"))
    fig.update_layout(title="Cumulative Energy vs Cost", xaxis_title="Cumulative energy (kWh)", yaxis_title="Cumulative cost (EUR)")
    return fig
