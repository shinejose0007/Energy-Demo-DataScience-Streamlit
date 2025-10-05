import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
def feature_engineer(df):
    df = df.copy(); df["datetime"]=pd.to_datetime(df["datetime"]); df["hour"]=df["datetime"].dt.hour; df["dayofweek"]=df["datetime"].dt.dayofweek
    df["lag1"]=df["consumption_kwh"].shift(1); df["lag24"]=df["consumption_kwh"].shift(24); df=df.dropna().reset_index(drop=True); return df
def train_forecast(df):
    df_fe = feature_engineer(df); features=["hour","dayofweek","lag1","lag24"]; X=df_fe[features]; y=df_fe["consumption_kwh"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    m = RandomForestRegressor(n_estimators=100, random_state=42); m.fit(X_train,y_train); preds=m.predict(X_test); mae=mean_absolute_error(y_test,preds); return m, mae
def forecast_next(df, model, n_steps=24):
    df=df.copy().reset_index(drop=True); last=df.iloc[-max(24,1):].copy().reset_index(drop=True); out=[]
    for step in range(n_steps):
        next_dt=last["datetime"].iloc[-1] + pd.Timedelta(hours=1); hour=next_dt.hour; dow=next_dt.dayofweek; lag1=last["consumption_kwh"].iloc[-1]
        lag24 = last["consumption_kwh"].iloc[-24] if len(last)>=24 else lag1
        Xpred = pd.DataFrame([{"hour":hour,"dayofweek":dow,"lag1":lag1,"lag24":lag24}]); pred=model.predict(Xpred)[0]; out.append({"datetime":next_dt,"forecast_kwh":float(pred)})
        new_row = pd.DataFrame([{"datetime":next_dt,"consumption_kwh":float(pred)}]); last = pd.concat([last, new_row], ignore_index=True)
        if len(last) > 48: last = last.iloc[-48:].reset_index(drop=True)
    return pd.DataFrame(out)
