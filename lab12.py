
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


#a
print("a")
df = pd.read_csv("date_promovare_examen.csv")
class_counts = df["Promovare"].value_counts()
print("distributia claselor:")
print(class_counts)
print("datele sunt balansate(250/250)")

X_studiu = df["Ore_Studiu"].values
X_somn = df["Ore_Somn"].values
y = df["Promovare"].values

X_studiu_std = (X_studiu - X_studiu.mean()) / X_studiu.std()
X_somn_std = (X_somn - X_somn.mean()) / X_somn.std()

with pm.Model() as logistic_model:
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta_studiu = pm.Normal("beta_studiu", mu=0, sigma=2)
    beta_somn = pm.Normal("beta_somn", mu=0, sigma=2)

    logits = alpha + beta_studiu * X_studiu_std + beta_somn * X_somn_std
    p = pm.Deterministic("p", pm.math.sigmoid(logits))

    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    trace = pm.sample(
        3000,
        tune=2000,
        target_accept=0.99,
        return_inferencedata=True
    )


#b
print("b")
alpha_mean = trace.posterior["alpha"].mean().item()
beta_studiu_mean = trace.posterior["beta_studiu"].mean().item()
beta_somn_mean = trace.posterior["beta_somn"].mean().item()

x1_vals = np.linspace(X_studiu_std.min(), X_studiu_std.max(), 100)
x2_vals = (-alpha_mean - beta_studiu_mean * x1_vals) / beta_somn_mean

print("coeficienti medii:")
print("alpha =", alpha_mean)
print("beta_studiu =", beta_studiu_mean)
print("beta_somn =", beta_somn_mean)

print("granita de decizie a modelului este dreapta cu ecuatia "
      "beta_studiu * x1 + beta_somn * x2 + alpha = 0, care corespunde "
      "probabilitatii egale cu 0.5")
plt.figure(figsize=(8, 6))

plt.scatter(
    X_studiu_std[y == 0],
    X_somn_std[y == 0],
    label="picat",
    alpha=0.6
)

plt.scatter(
    X_studiu_std[y == 1],
    X_somn_std[y == 1],
    label="promovat",
    alpha=0.6
)

plt.plot(x1_vals, x2_vals, color="black", linewidth=2, label="granita decizie")

plt.xlabel("ore de studiu (standardizat)")
plt.ylabel("ore de somn (standardizat)")
plt.legend()
plt.show()

print("din figura se observa ca datele sunt bine separate")

#c
print("c")

print("orele de somn influenteaza mai mult promovabilitatea, deoarece beta_somn > beta_studiu")
