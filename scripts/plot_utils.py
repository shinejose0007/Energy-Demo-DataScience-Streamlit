import pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot_price_and_forecast_plotly(price_df, forecast_df, last_hours=168):
    price_df=price_df.copy(); price_df["datetime"]=pd.to_datetime(price_df["datetime"])
    if len(price_df)>last_hours: price_plot_df=price_df.iloc[-last_hours:].copy()
    else: price_plot_df=price_df.copy()
    forecast_df=forecast_df.copy(); forecast_df["datetime"]=pd.to_datetime(forecast_df["datetime"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=price_plot_df["datetime"], y=price_plot_df["price_eur_per_kwh"], name="price (EUR/kWh)", mode="lines"), secondary_y=False)
    fig.add_trace(go.Scatter(x=forecast_df["datetime"], y=forecast_df["forecast_kwh"], name="forecast (kWh)", mode="lines+markers"), secondary_y=True)
    fig.update_xaxes(title_text="datetime"); fig.update_yaxes(title_text="price (EUR/kWh)", secondary_y=False); fig.update_yaxes(title_text="forecast (kWh)", secondary_y=True)
    fig.update_layout(height=480, margin=dict(l=40, r=40, t=30, b=30)); return fig
