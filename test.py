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

acwr = [
    0.6808408675518812,
    0.7437095734862025,
    0.8454693293702962,
    0.750578453601512,
    0.6663376657991456,
    0.8347484369165533,
    0.7410609200439201,
    0.6578882837293247,
    0.7115587443159684,
    0.6316973767966487,
    0.5607991613071195,
    0.6473351035912801,
    0.5746817806235528,
    0.6478953951108585,
    0.5751791760552017,
    0.8246591451133023,
    0.8146331939907743,
    0.7232032357311137,
    0.9391512730085192,
    1.0170649651249473,
    0.9029152256350167,
    0.8015769844621503,
    0.7966437407832712,
    0.7072328042862503,
    1.1915367608314933,
    1.2677338267639564,
    1.16742320321422,
    1.0363980961330728,
    0.9976684537388437,
    1.1143023899621478,
    1.3583618397334476,
    1.2059069047626332,
    1.289070709835651,
    1.1443924847141567,
    1.0159521340434519,
    1.2074993653730737,
    1.1703997207687873,
    1.1780479420478631,
    1.329326820872318,
    1.1801305617807771,
    1.047679136362662,
    1.1181482866014647,
    0.9926535443955324,
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

acrs = acwr


data = {
    "acrs" : acrs,
    "ctl" : ctl,
    
}
shift = 6
k = pd.DataFrame.from_dict(data)
k["aroll"] = k["acrs"].rolling(shift).mean()
k["astdroll"] = k["acrs"].rolling(shift).std()

k["change"] = (k["ctl"]-k["ctl"].shift(shift))/k["ctl"]
k = k.dropna()
k = k.sort_values(by="aroll")


dmax = np.ceil(np.max(k["aroll"]))
dstd = np.std(acrs)
dmin = np.floor(np.min(k["aroll"]))


def df_linpred(df, x, y, p, thres=1):
    df = df.sort_values(x)
    coef = np.polyfit(df[x],df[y],p)
    poly1d_fn = np.poly1d(coef)
    df["pred"] = [poly1d_fn(z) for z in df[x].to_numpy()]
    residuals = df[y] - df["pred"]
    err = np.mean(np.abs(residuals))
    resstd = residuals.std(ddof=1)

    lower = lambda x : poly1d_fn(x) - resstd
    higher = lambda x : poly1d_fn(x) +resstd
    df["pred_l"] = [lower(z) for z in df[x].to_numpy()]
    df["pred_h"] = [higher(z) for z in df[x].to_numpy()]

    rng = np.random.default_rng()
    final_s = lambda x: rng.normal(loc=poly1d_fn(x), scale=resstd)
    final = poly1d_fn
    risk = lambda x: max(0, min(1, (thres - lower(x)) / (higher(x) - lower(x))))

    return df, final_s, final, risk, err


def pareto_frontier(df, risk_col="suppression_risk", reward_col="ctl_change_pct"):
    risk = df[risk_col].to_numpy()
    reward = df[reward_col].to_numpy()
    points = np.array(list(zip(risk, reward)))
    is_efficient = np.ones(points.shape[0], dtype=bool)

    for i, (r, re) in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = ~(
                (risk[is_efficient] <= r) & (reward[is_efficient] < re)
            )
            is_efficient[i] = True
    return df[is_efficient].sort_values(risk_col)

k, interp_change_rand, interp_change,  risk_c, err = df_linpred(k, "aroll", "change", 2, 0)



import matplotlib.pyplot as plt

plt.plot(k["aroll"], k["pred"])
plt.plot(k["aroll"], k["pred_l"])
plt.plot(k["aroll"], k["pred_h"])
plt.plot(k["aroll"], k["change"])
plt.hlines([0], 0.6, 1.2)
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

print(df_hrv_ratio)

# Shift HRV ratio by +6 days (lag effect)
df_hrv_ratio["HRVratio_plus6"] = df_hrv_ratio["HRV_ratio"].shift(-shift)
df_hrv_ratio = df_hrv_ratio.dropna()

df_hrv_ratio, hrv_interp_rand, hrv_interp, risk_interp, err = df_linpred(df_hrv_ratio, "ACRS", "HRVratio_plus6", 1, 1)


plt.scatter(df_hrv_ratio["ACRS"], df_hrv_ratio["HRVratio_plus6"])
plt.plot(df_hrv_ratio["ACRS"], df_hrv_ratio["pred"])
plt.plot(df_hrv_ratio["ACRS"], df_hrv_ratio["pred_l"])
plt.plot(df_hrv_ratio["ACRS"], df_hrv_ratio["pred_h"])
plt.title(str(err))
plt.show()


test = {"ACRS" : np.linspace(dmin, dmax, 1000)}
df = pd.DataFrame.from_dict(test)
df = df_hrv_ratio
def func(x):
    if x > 1.1:
        return 1
    else:
        return 1-risk_c(x)

df["change"] = interp_change(df["ACRS"])
df["risk_c"] = [func(k) for k in df["ACRS"].to_numpy()]
df["risk"] = [risk_interp(k) for k in df["ACRS"].to_numpy()]
front = pareto_frontier(df, "risk", "risk_c")
front, interp_change_rand, interp_change,  risk_c, err = df_linpred(front, "risk", "risk_c", 1, 0)

df["score"] = df["risk_c"] / df["risk"]

plt.scatter(df["risk"], df["risk_c"], color="orange")
plt.scatter(front["risk"], front["risk_c"], color="blue")
plt.plot(front["risk"], front["pred"], color="blue")

plt.show()

plt.plot(df["ACRS"], df["risk"], label="hrv")
plt.plot(df["ACRS"], df["risk_c"], label="ctl")
plt.scatter(front["ACRS"], front["risk_c"], color="blue", label="front")
plt.scatter(front["ACRS"], front["risk"], color="blue", label="front")
plt.xlabel("acwr")
plt.ylabel("risk")
plt.hlines([0.5], dmin, dmax)
plt.legend()
plt.show()
print("front")
print(front.sort_values(by="risk"))