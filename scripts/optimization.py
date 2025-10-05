import pandas as pd, pulp
def schedule_multiple_evs(price_series, ev_profiles, household_cap_kw=None):
    hours = sorted(pd.to_datetime(price_series.index).unique()); prob = pulp.LpProblem("multi_ev_scheduling", pulp.LpMinimize); charge={}
    for _, row in ev_profiles.iterrows():
        ev=str(row.ev_id); start=pd.to_datetime(row.arrival_dt) if "arrival_dt" in row else None; end=pd.to_datetime(row.departure_dt) if "departure_dt" in row else None
        allowed_hours = [h for h in hours if (start is None or (h>=start)) and (end is None or (h<end))]
        for h in allowed_hours:
            charge[(ev,h)] = pulp.LpVariable(f"c_{ev}_{h}", lowBound=0, upBound=float(row.max_charge_kw))
    prob += pulp.lpSum([float(price_series[h]) * charge[(ev,h)] for (ev,h) in charge])
    for ev_id, group in ev_profiles.groupby("ev_id"):
        ev=str(ev_id); vars_ev=[charge[(ev,h)] for (e,h) in charge.keys() if e==ev]
        required=float(group.iloc[0].get("required_kwh",0)); 
        if vars_ev: prob+=pulp.lpSum(vars_ev) >= required
    if household_cap_kw is not None:
        for h in hours:
            vars_at_h=[charge[(ev,h)] for (ev,h2) in charge.keys() if h2==h and (ev,h) in charge]
            if vars_at_h: prob+=pulp.lpSum(vars_at_h) <= household_cap_kw
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    schedules={}
    for ev_id, group in ev_profiles.groupby("ev_id"):
        ev=str(ev_id); rows=[]
        for (e,h), var in charge.items():
            if e==ev:
                val = pulp.value(var); rows.append({"datetime":h,"charge_kw":float(val if val is not None else 0.0),"price":float(price_series[h])})
        if rows:
            df=pd.DataFrame(rows).sort_values("datetime"); df["energy_kwh"]=df["charge_kw"]*1.0; df["cost_eur"]=df["energy_kwh"]*df["price"]; schedules[ev]=df
    return schedules
