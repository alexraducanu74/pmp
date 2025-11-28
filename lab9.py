import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]
prior_mu = 10

fig_post, axes_post = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 8), constrained_layout=True)
fig_pred, axes_pred = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 8), constrained_layout=True)

results = {}

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        model_name = f"Y={Y}_theta={theta}"
        with pm.Model() as model:
            n = pm.Poisson("n", mu=prior_mu)
            y = pm.Binomial("y", n=n, p=theta, observed=Y)
            step = pm.Metropolis()

            idata = pm.sample(draws=3000, tune=2000, chains=2, step=step, random_seed=2025, progressbar=True)
            idata_pp = pm.sample_posterior_predictive(idata, var_names=["y"], extend_inferencedata=True)

        results[(Y, theta)] = {"model": model, "idata": idata, "idata_pp": idata_pp}

        ax = axes_post[i, j] if len(Y_values) > 1 else axes_post[j]
        az.plot_posterior(idata, var_names=["n"], ax=ax, hdi_prob=0.94)
        ax.set_title(model_name)

        y_pp = idata_pp.posterior_predictive["y"].values
        y_pp_flat = y_pp.reshape(-1)

        ax2 = axes_pred[i, j] if len(Y_values) > 1 else axes_pred[j]
        az.plot_dist(y_pp_flat, ax=ax2)
        ax2.set_xlabel("Y* (future buyers)")
        ax2.set_title(model_name)


fig_post.suptitle("a) posterior distributions", fontsize=14)
fig_pred.suptitle("c) posterior predictive distributions", fontsize=14)
plt.show()


for (Y, theta), obj in results.items():
    idata = obj["idata"]
    n_samples = idata.posterior["n"].values.reshape(-1)
    print(f"\nY={Y}, theta={theta} -> posterior for n:")
    print(az.summary(idata, var_names=["n"], hdi_prob=0.94))
    p_eq_Y = np.mean(n_samples == Y)
    print(f"P(n == observed Y) = {p_eq_Y}")

# README.MD duplicate:
# b) The posterior for n increases with the observed number of buyers Y and decreases as the purchase probability theta increases
# d) The posterior predictive for future buyers Y* is broader than the posterior for n, because it accounts for both uncertainty in n and randomness in individual purchase decisions.
# Even if n is somewhat certain, the predictive distribution spreads due to Binomial randomness.