
#acrs = acwr
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def make_df(shift, acrs, ctl, hrv):
    data = {
        "ACRS" : acrs,
        "ctl" : ctl,
        "hrv":  pd.Series(hrv).replace(0, np.nan),
    }
    k = pd.DataFrame.from_dict(data).dropna()
    k["aroll"] = k["ACRS"].rolling(shift).mean()
    k["astdroll"] = k["ACRS"].rolling(shift).std()
    return k

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

def optimize(acrs, ctl, hrv):
    
    shift = 6
    hrvshort = 4
    hrvlong = 6


    plot = True
    k = make_df(shift, acrs, ctl, hrv)

    k["change"] = (k["ctl"]-k["ctl"].shift(shift))/k["ctl"]
    k["HRV_ratio"] = k["hrv"].ewm(span=hrvshort, adjust=False).mean() / k["hrv"].ewm(span=hrvlong, adjust=False).mean()


    k = k.dropna()
    k = k.sort_values(by="aroll")


    dmax = np.ceil(np.max(k["ACRS"]))
    dmin = np.floor(np.min(k["ACRS"]))

    k, interp_change_rand, interp_change,  risk_c, hrv_err = df_linpred(k, "aroll", "change", 2, 0)
    print(k.sort_values(by="pred", ascending=False))
    cutpoint = k[k["pred"] == k["pred"].max()]["aroll"].to_numpy()[0]
    """
    if plot:
        plt.plot(k["aroll"], k["pred"])
        plt.plot(k["aroll"], k["pred_l"])
        plt.plot(k["aroll"], k["pred_h"])
        plt.plot(k["aroll"], k["change"])
        plt.hlines([0], 0.6, 1.2)
        plt.title(str(hrv_err))
        plt.show()
    """

    df_hrv_ratio = k
    print(df_hrv_ratio)

    # Shift HRV ratio by +6 days (lag effect)
    df_hrv_ratio["HRVratio_plus6"] = df_hrv_ratio["HRV_ratio"].shift(-shift)
    df_hrv_ratio = df_hrv_ratio.dropna()

    df_hrv_ratio, hrv_interp_rand, hrv_interp, risk_interp, err = df_linpred(df_hrv_ratio, "ACRS", "HRVratio_plus6", 1, 1)
    """
    if plot:
        plt.scatter(df_hrv_ratio["ACRS"], df_hrv_ratio["HRVratio_plus6"])
        plt.plot(df_hrv_ratio["ACRS"], df_hrv_ratio["pred"])
        plt.plot(df_hrv_ratio["ACRS"], df_hrv_ratio["pred_l"])
        plt.plot(df_hrv_ratio["ACRS"], df_hrv_ratio["pred_h"])
        plt.title(str(err))
        plt.show()
    """
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
        k = make_df(shift,  acrs, ctl, hrv)
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
    mean, std, val = df[["mean", "std", "val"]].head(1).to_numpy()[0]
    
    return {"mean": mean, "std": std, "val": val, "hrv_err": hrv_err}

    

#print(optimize(acwr, ctl, hrv))


