import pandas as pd

def load_from_csv(path_or_buffer, parse_dates=["datetime"]):
    df = pd.read_csv(path_or_buffer, parse_dates=parse_dates)
    return df.sort_values(parse_dates[0]).reset_index(drop=True)
