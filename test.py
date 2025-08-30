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

#acrs = acwr

def make_df(shift):
    data = {
        "ACRS" : acrs,
        "ctl" : ctl,
        "hrv":  pd.Series(hrv).replace(0, np.nan),
    }
    k = pd.DataFrame.from_dict(data).dropna()
    k["aroll"] = k["ACRS"].rolling(shift).mean()
    k["astdroll"] = k["ACRS"].rolling(shift).std()
    return k

shift = 6
hrvshort = 4
hrvlong = 6
k = make_df(shift)
k["change"] = (k["ctl"]-k["ctl"].shift(shift))/k["ctl"]
k["HRV_ratio"] = k["hrv"].ewm(span=hrvshort, adjust=False).mean() / k["hrv"].ewm(span=hrvlong, adjust=False).mean()


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
    #risk = lambda x: max(0, min(1, (thres - lower(x)) / (higher(x) - lower(x))))
    def risk(x):
        x = np.asarray(x, dtype=float)
        val = (thres - lower(x)) / (higher(x) - lower(x))
        val = np.clip(val, 0, 1)  # clamp between 0 and 1
        if val.shape == ():  # scalar
            return float(val)
        return val

    return df, final_s, final, risk, err

k, interp_change_rand, interp_change,  risk_c, err = df_linpred(k, "aroll", "change", 2, 0)
print(k.sort_values(by="pred", ascending=False))
cutpoint = k[k["pred"] == k["pred"].max()]["aroll"].to_numpy()[0]

import matplotlib.pyplot as plt

plt.plot(k["aroll"], k["pred"])
plt.plot(k["aroll"], k["pred_l"])
plt.plot(k["aroll"], k["pred_h"])
plt.plot(k["aroll"], k["change"])
plt.hlines([0], 0.6, 1.2)
plt.title(str(err))
plt.show()

df_hrv_ratio = k
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

def func(x):
    res = 1-risk_c(x)
    res[x > cutpoint] = 1
    return res
    

def eval(mean, std, hrv, reward):
    def normal_pdf(x, mu, sigma):
        return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    r = np.linspace(dmin, dmax, 1000)
    dx = r[1] - r[0]

    pdf = normal_pdf(r, mean, std)
    rew = np.sum(pdf * reward(r)) * dx
    pen = np.sum(pdf * hrv(r)) * dx

    return rew/pen

result = {
    "mean" : [],
    "std" : [],
    "val" : [],
}

for shift in range(2, 10):
    k = make_df(shift)
    if k.size > 0:
        #k["score"] = k.apply(lambda r : eval(r.ACRS, r.astdroll, risk_interp, func), axis=1)
        k["score"] = k.apply(lambda r : eval(r["aroll"], r.astdroll, risk_interp, func), axis=1)
        top = k.sort_values("score", ascending=False)
        vals = top[["aroll", "astdroll"]].head(1).to_numpy()[0]
        value = top[["score"]].head(1).to_numpy()[0]

        result["mean"].append(vals[0])
        result["std"].append(vals[1])
        result["val"].append(value[0])


        k["score"] = k.apply(lambda r : eval(r.ACRS, r.astdroll, risk_interp, func), axis=1)
        top = k.sort_values("score", ascending=False)
        vals = top[["ACRS", "astdroll"]].head(1).to_numpy()[0]
        value = top[["score"]].head(1).to_numpy()[0]

        result["mean"].append(vals[0])
        result["std"].append(vals[1])
        result["val"].append(value[0])

df = pd.DataFrame.from_dict(result).sort_values("val", ascending=False)
print(df)
print(eval(1, 0.1, risk_interp, func))



