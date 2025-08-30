import numpy as np
import pandas as pd

# === Example HRV + ACRS data (replace with your real arrays) ===
# Raw HRV values (0 replaced with NaN)
ctl = [
    18.703112,
    18.780682,
    19.044653,
    18.596565,
    18.15902,
    18.625845,
    18.18761,
    17.759687,
    17.78887,
    17.370327,
    16.961632,
    17.056648,
    16.655334,
    16.7105,
    16.317331,
    16.968658,
    16.851755,
    16.455261,
    17.079815,
    17.336748,
    16.928844,
    16.530537,
    16.42394,
    16.037514,
    17.636557,
    18.04509,
    17.785217,
    17.36676,
    17.24049,
    17.68187,
    18.748129,
    18.307016,
    18.770359,
    18.328724,
    17.89748,
    18.676327,
    18.636885,
    18.76307,
    19.52155,
    19.06224,
    18.613739,
    18.928696,
    18.483335,
]

hrv = [
    84,
    94,
    95,
    103,
    101,
    91,
    78,
    82,
    78,
    109,
    116,
    107,
    117,
    0,
    108,
    85,
    88,
    110,
    129,
    116,
    124,
    124,
    117,
    138,
    117,
    137,
    135,
    0,
    98,
    131,
    138,
    131,
    127,
    85,
    92,
    92,
    97,
    104,
    101,
    86,
    124,
    105,
    105,
]

acrs = [
    0.6808408675518812,
    0.8119045996751121,
    1.041020424865098,
    0.8184289517852184,
    0.6455914751282417,
    0.9885965104614157,
    0.7782623094173785,
    0.614466760362124,
    0.7196658896289682,
    0.5686344543493773,
    0.45013041856903385,
    0.5979333721135818,
    0.47315408079905186,
    0.6008151093676325,
    0.47540950026428164,
    0.9342478045274317,
    0.9157877236531302,
    0.7222613295983527,
    1.1819817456562642,
    1.3857922848237225,
    1.08725774900865,
    0.8561755023602031,
    0.8495249439355942,
    0.6707456162538404,
    1.6962028045088013,
    1.9268410262873665,
    1.6303969963654112,
    1.279296271807824,
    1.1895755285625917,
    1.4824033373933359,
    2.164854623189683,
    1.6897800250057144,
    1.9381547371520838,
    1.5161232595212373,
    1.1912561196192077,
    1.6702594554236974,
    1.5742902983028295,
    1.6022068149164994,
    2.0374020096273746,
    1.5913567019117227,
    1.2491409803076619,
    1.4269987178769241,
    1.1219114980177662,
]
max = np.ceil(np.max(acrs))
dstd = np.std(acrs)
min = np.floor(np.min(acrs))


data = {
    "acrs" : acrs,
    "ctl" : ctl,
}
k = pd.DataFrame.from_dict(data)
k["aroll"] = k["acrs"].rolling(6).mean()
k["change"] = (k["ctl"]-k["ctl"].shift(6))/k["ctl"]
k = k.dropna()
k = k.sort_values(by="aroll")

def df_linpred(df, x, y, p):
    coef = np.polyfit(df[x],df[y],p)
    poly1d_fn = np.poly1d(coef)
    df["pred"] = [poly1d_fn(z) for z in df[x].to_numpy()]
    err = np.mean(np.abs(df["pred"]-df[y]))
    return df, poly1d_fn, err

k, interp_change, err = df_linpred(k, "aroll", "change", 2)

import matplotlib.pyplot as plt

plt.plot(k["aroll"], k["pred"])
plt.plot(k["aroll"], k["change"])
plt.title(str(err))
plt.show()


# Clean up HRV and align with ACRS
hrv_series = pd.Series(hrv).replace(0, np.nan).dropna()
acrs_series = pd.Series(acrs).iloc[hrv_series.index]

# Compute HRV ratio (7d vs 42d EWM)
hrv_short = hrv_series.ewm(span=4, adjust=False).mean()
hrv_long  = hrv_series.ewm(span=8, adjust=False).mean()

hrv_ratio = hrv_short / hrv_long

df_hrv_ratio = pd.DataFrame({
    "HRV_ratio": hrv_ratio,
    "ACRS": acrs_series
}).dropna()
df_hrv_ratio = df_hrv_ratio.sort_values(by="ACRS")


# Shift HRV ratio by +6 days (lag effect)
df_hrv_ratio["HRVratio_plus6"] = df_hrv_ratio["HRV_ratio"].shift(-1)
df_hrv_ratio = df_hrv_ratio.dropna()

df_hrv_ratio, interp_change, err = df_linpred(df_hrv_ratio, "ACRS", "HRVratio_plus6", 1)


plt.scatter(df_hrv_ratio["ACRS"], df_hrv_ratio["HRVratio_plus6"])
plt.plot(df_hrv_ratio["ACRS"], df_hrv_ratio["pred"])
plt.title(str(err))
plt.show()


# Bin ACRS values
bin_edges = np.arange(min, max, 0.2)
df_hrv_ratio["ACRS_bin"] = pd.cut(df_hrv_ratio["ACRS"], bins=bin_edges, include_lowest=True)

group_stats = df_hrv_ratio.groupby("ACRS_bin")["HRVratio_plus6"].agg(["mean", "std", "count"])
bin_centers = np.array([interval.mid for interval in group_stats.index], dtype=float)
mean_hrv    = group_stats["mean"].values
supp_prob   = df_hrv_ratio.groupby("ACRS_bin")["HRVratio_plus6"].apply(lambda x: (x < 1.0).mean()).values


plt.plot(bin_centers, supp_prob)
plt.show()


data = {
    "bin" : bin_centers,
    "mhrv" : mean_hrv,
    "prob" : supp_prob,
}
k = pd.DataFrame.from_dict(data)
print(k)

print(bin_edges, supp_prob)
# === Replacement interpolators (NumPy instead of SciPy) ===
def hrv_interp(x):
    return np.interp(x, bin_centers, mean_hrv, left=mean_hrv[0], right=mean_hrv[-1])

def risk_interp(x):
    return np.interp(x, bin_centers, supp_prob, left=supp_prob[0], right=supp_prob[-1])

# === Simulation using ACRS ===
def simulate_series(m=0.5, s=0.1, days=1000, low=min, high=max):
    rng = np.random.default_rng()
    acrs_path = np.clip(rng.normal(m, s, size=days), low, high)

    ctl_change = float(np.mean([interp_change(k) for k in acrs_path]))
    reward = float(np.mean(hrv_interp(acrs_path)))
    risk   = float(np.mean(risk_interp(acrs_path)))
    return {
        "mean": m, "std": s,
        "ctl_change_pct": ctl_change,
        "exp_hrv_ratio": reward,
        "suppression_risk": risk,
    }

# === Grid search across mean/std values ===
means = np.arange(min, max, 0.1)
stds  = [0.05, 0.1, 0.15, 0.2, 0.5, 1, 2]

rows = []
for m in means:
    for s in stds:
        rows.append(simulate_series(m, s, days=6))

results_acrs = pd.DataFrame(rows)

# Risk-adjusted score
results_acrs["score"] = (
    (results_acrs["exp_hrv_ratio"] - 1.0) +
    0.5*(results_acrs["ctl_change_pct"]/100.0) -
    results_acrs["suppression_risk"]
)

print(results_acrs.sort_values("score", ascending=False))
