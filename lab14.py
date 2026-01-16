import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

data = pd.read_csv('date_colesterol.csv')
X = data['Ore_Exercitii'].values
Y = data['Colesterol'].values
X_std = (X - np.mean(X)) / np.std(X)

clusters_list = [3, 4, 5]
models = []
idatas = []

for K in clusters_list:
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(K))
        alpha = pm.Normal('alpha', mu=0, sigma=10, shape=K)
        beta  = pm.Normal('beta', mu=0, sigma=10, shape=K)
        gamma = pm.Normal('gamma', mu=0, sigma=10, shape=K)
        sigma = pm.HalfNormal('sigma', sigma=10, shape=K)
        mu = alpha + beta * X_std[:, None] + gamma * X_std[:, None] ** 2

        y = pm.NormalMixture('y', w=w, mu=mu, sigma=sigma, observed=Y)

        idata = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=123, idata_kwargs={"log_likelihood": True})
    models.append(model)
    idatas.append(idata)

waic_scores = [az.waic(idata).elpd_waic for idata in idatas]
loo_scores = [az.loo(idata).elpd_loo for idata in idatas]

best_idx = np.argmax(waic_scores)
best_K = clusters_list[best_idx]
best_idata = idatas[best_idx]

summary = az.summary(best_idata, var_names=['w','alpha','beta','gamma','sigma'])
print("Best number of subpopulations:", best_K)
print(summary)
