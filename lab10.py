import numpy as np
import pymc as pm
import arviz as az

publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                      6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                  15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)

    mu = alpha + beta * publicity
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=sales)

    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=1)

print("a")
print(az.summary(idata, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95))

print("b")
print(az.hdi(idata, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95))

print("c")
new_publicity = np.array([2.5, 5.0, 8.0, 11.5])

posterior = idata.posterior.stack(sample=("chain", "draw"))
alpha_samps = posterior["alpha"].values
beta_samps = posterior["beta"].values
sigma_samps = posterior["sigma"].values

for x in new_publicity:
    mu_samps = alpha_samps + beta_samps * x
    y_pred = np.random.normal(mu_samps, sigma_samps)

    mean_pred = y_pred.mean()
    hdi_pred = az.hdi(y_pred, hdi_prob=0.95)

    print(f"publicity = {x}")
    print(f"mean predicted sales: {mean_pred}")
    print(f"95% HDI: {hdi_pred}")
