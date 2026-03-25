import pandas as pd
import numpy as np


def has_suspicious_request_len(dataf):
    df = dataf.copy()
    df = df[df["method"].notna()]
    df["resource_len"] = df["resource"].str.len().fillna(0)

    q1 = df['resource_len'].quantile(0.25)
    q3 = df['resource_len'].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df["has_suspicious_request_len"] = 1

    mask_normale = (df["resource_len"] >= lower_bound) & (df["resource_len"] <= upper_bound)
    df.loc[mask_normale, "has_suspicious_request_len"] = 0

    return df


def entropy(s):
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy
