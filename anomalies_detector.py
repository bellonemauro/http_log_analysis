import pandas as pd
import numpy as np


def has_suspicious_request_len(dataf):
    dataf = dataf[dataf["method"].notna()]
    dataf["resource_len"] = dataf["resource"].str.len()
    range_interquartilico = np.quantile(dataf['resource_len'], 0.75) - np.quantile(dataf['resource_len'], 0.25)

    dataf.loc[
        (dataf["resource_len"] <= np.quantile(dataf['resource_len'], 0.75)+(range_interquartilico*1.5))
        &
        (dataf["resource_len"] >= np.quantile(dataf['resource_len'], 0.25)-(range_interquartilico*1.5)),
        "has_suspicious_request_len"
    ] = 0
    dataf.fillna({"has_suspicious_request_len": 1}, inplace=True)
    return dataf
