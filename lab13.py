import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")
dummy_data = np.loadtxt("date.csv")
x = dummy_data[:, 0]
y = dummy_data[:, 1]

def prepare_data(x, y, order):
    Xp = np.vstack([x**i for i in range(1, order + 1)])
    Xs = (Xp - Xp.mean(axis=1, keepdims=True)) / Xp.std(axis=1, keepdims=True)
    ys = (y - y.mean()) / y.std()
    return Xs, ys

# 1a
order = 5
Xs, ys = prepare_data(x, y, order)

with pm.Model() as model_p5:
    α = pm.Normal("α", 0, 1)
    β = pm.Normal("β", 0, 10, shape=order)
    ϵ = pm.HalfNormal("ϵ", 5)

    μ = α + pm.math.dot(β, Xs)
    y_pred = pm.Normal("y_pred", μ, ϵ, observed=ys)

    idata_p5 = pm.sample(2000, return_inferencedata=True, target_accept = 0.9)

idx = np.argsort(Xs[0])
α_m = idata_p5.posterior["α"].mean(("chain", "draw")).values
β_m = idata_p5.posterior["β"].mean(("chain", "draw")).values
y_m = α_m + np.dot(β_m, Xs)

plt.figure()
plt.scatter(Xs[0], ys, s=10)
plt.plot(Xs[0][idx], y_m[idx], label="order=5, sd=10")
plt.legend()
plt.title("ex 1a")
plt.show()

# 1b
with pm.Model() as model_p5_sd100:
    α = pm.Normal("α", 0, 1)
    β = pm.Normal("β", 0, 100, shape=order)
    ϵ = pm.HalfNormal("ϵ", 5)

    μ = α + pm.math.dot(β, Xs)
    y_pred = pm.Normal("y_pred", μ, ϵ, observed=ys)

    idata_p5_sd100 = pm.sample(2000, return_inferencedata=True, target_accept = 0.9)

with pm.Model() as model_p5_array:
    α = pm.Normal("α", 0, 1)
    β = pm.Normal("β", 0, np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    ϵ = pm.HalfNormal("ϵ", 5)

    μ = α + pm.math.dot(β, Xs)
    y_pred = pm.Normal("y_pred", μ, ϵ, observed=ys)

    idata_p5_array = pm.sample(2000, return_inferencedata=True, target_accept = 0.9)

plt.figure()
plt.scatter(Xs[0], ys, s=10)

for idata, label in zip(
    [idata_p5, idata_p5_sd100, idata_p5_array],
    ["sd=10", "sd=100", "array"]
):
    α_m = idata.posterior["α"].mean(("chain", "draw")).values
    β_m = idata.posterior["β"].mean(("chain", "draw")).values
    y_m = α_m + np.dot(β_m, Xs)
    plt.plot(Xs[0][idx], y_m[idx], label=label)

plt.legend()
plt.title("ex 1b")
plt.show()
print("the curves are identical")
# 2
N = 500
x_big = np.linspace(x.min(), x.max(), N)
y_big = np.interp(x_big, x, y) + np.random.normal(0, 0.2, N)

Xs_big, ys_big = prepare_data(x_big, y_big, order)

with pm.Model() as model_big:
    α = pm.Normal("α", 0, 1)
    β = pm.Normal("β", 0, 10, shape=order)
    ϵ = pm.HalfNormal("ϵ", 5)

    μ = α + pm.math.dot(β, Xs_big)
    y_pred = pm.Normal("y_pred", μ, ϵ, observed=ys_big)

    idata_big = pm.sample(2000, return_inferencedata=True, target_accept = 0.9)

plt.figure()
plt.scatter(Xs_big[0], ys_big, s=5)
α_m = idata_big.posterior["α"].mean(("chain", "draw")).values
β_m = idata_big.posterior["β"].mean(("chain", "draw")).values
y_m = α_m + np.dot(β_m, Xs_big)
idx = np.argsort(Xs_big[0])
plt.plot(Xs_big[0][idx], y_m[idx], label="order=5, N=500")
plt.legend()
plt.title("ex 2")
plt.show()

# 3
def fit_model(order, x, y):
    Xs_ord, ys_ord = prepare_data(x, y, order)
    with pm.Model() as model:
        α = pm.Normal("α", 0, 1)
        β = pm.Normal("β", 0, 10, shape=order)
        ϵ = pm.HalfNormal("ϵ", 5)

        μ = α + pm.math.dot(β, Xs_ord)
        y_pred = pm.Normal("y_pred", μ, ϵ, observed=ys_ord)

        idata = pm.sample(2000, return_inferencedata=True, target_accept = 0.9)
        pm.compute_log_likelihood(idata, model=model)
    return idata

idata_lin = fit_model(1, x, y)
idata_quad = fit_model(2, x, y)
idata_cubic = fit_model(3, x, y)

cmp_waic = az.compare(
    {
        "linear": idata_lin,
        "quadratic": idata_quad,
        "cubic": idata_cubic
    },
    ic="waic",
    scale="deviance"
)

cmp_loo = az.compare(
    {
        "linear": idata_lin,
        "quadratic": idata_quad,
        "cubic": idata_cubic
    },
    ic="loo",
    scale="deviance"
)

az.plot_compare(cmp_waic)
plt.title("WAIC comparison")
plt.show()

az.plot_compare(cmp_loo)
plt.title("LOO comparison")
plt.show()

merged = cmp_waic[["elpd_waic", "p_waic", "weight"]].copy()
merged.columns = ["elpd_WAIC", "p_WAIC", "WAIC_weight"]
merged["elpd_LOO"] = cmp_loo["elpd_loo"]
merged["p_LOO"] = cmp_loo["p_loo"]
merged["LOO_weight"] = cmp_loo["weight"]

print(merged)
print("cubic - overfitting")
print("linear - underfitting")
print("quadratic - good fit")
