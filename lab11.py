import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

df = pd.read_csv("Prices.csv")

y = df["Price"].values
x1 = df["Speed"].values
x2 = np.log(df["HardDrive"].values)

#a
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=1e4)
    beta1 = pm.Normal("beta1", mu=0, sigma=1e2)
    beta2 = pm.Normal("beta2", mu=0, sigma=1e2)
    sigma = pm.HalfNormal("sigma", sigma=1e3)

    mu = alpha + beta1 * x1 + beta2 * x2
    pm.Deterministic("mu", mu)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(draws=2000, tune=1000, target_accept=0.95)

#b
print("b")
summary = az.summary(trace,
                     var_names=["alpha","beta1","beta2","sigma"],
                     hdi_prob=0.95)

hdi_95_beta1 = az.hdi(trace.posterior["beta1"], hdi_prob=0.95)
hdi_95_beta2 = az.hdi(trace.posterior["beta2"], hdi_prob=0.95)

print(summary)
print("95% HDI for beta1:", hdi_95_beta1)
print("95% HDI for beta2:", hdi_95_beta2)

#c
print("c")
print("Both processor speed (beta1) and hard disk size (beta2) are useful predictors "
      "of the sale price because their 95% HDIs do not include 0, indicating "
      "a positive effect on the price.")
#d
print("d")
x1_new = 33.0
x2_new = np.log(540.0)

post = trace.posterior
alpha_draws = post["alpha"].stack(sample=("chain","draw")).values
beta1_draws = post["beta1"].stack(sample=("chain","draw")).values
beta2_draws = post["beta2"].stack(sample=("chain","draw")).values

mu_new_draws = alpha_draws + beta1_draws * x1_new + beta2_draws * x2_new

hdi_90_mu = az.hdi(mu_new_draws, hdi_prob=0.90)
print("90% HDI for expected sale price (mu):", hdi_90_mu)

#e
print("e")
sigma_draws = post["sigma"].stack(sample=("chain","draw")).values
rng = np.random.default_rng(12345)

y_pred_draws = mu_new_draws + rng.normal(0, sigma_draws, size=mu_new_draws.shape)

hdi_90_pred = az.hdi(y_pred_draws, hdi_prob=0.90)
print("90% HDI prediction interval (sale price):", hdi_90_pred)


premium = df["Premium"].str.lower().map({"yes": 1, "no": 0}).values
with pm.Model() as model_prem:
    alpha = pm.Normal("alpha", 0, 1e4)
    beta1 = pm.Normal("beta1", 0, 1e2)
    beta2 = pm.Normal("beta2", 0, 1e2)
    beta_prem = pm.Normal("beta_prem", 0, 1e3)
    sigma = pm.HalfNormal("sigma", 1e3)

    mu = alpha + beta1 * x1 + beta2 * x2 + beta_prem * premium
    pm.Deterministic("mu", mu)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    trace_prem = pm.sample(draws=2000, tune=2000, target_accept=0.95)

    prem_summary = az.summary(trace_prem,
                              var_names=["beta_prem"],
                              hdi_prob=0.95)

    prem_hdi = az.hdi(trace_prem.posterior["beta_prem"], hdi_prob=0.95)

print(prem_summary)
print("95% HDI for beta_prem:", prem_hdi)
print("bonus")
print("yes, beta_prem has a negative sign and its HDI does not include 0, "
      "so premium manufacturers will be expected to have lower prices on average")