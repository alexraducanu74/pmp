from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from pgmpy.inference import VariableElimination
#1a
print("1a")
model = MarkovNetwork()
edges = [("A1", "A2"), ("A1", "A3"), ("A2", "A4"), ("A2", "A5"), ("A3", "A4"), ("A4", "A5")]
model.add_edges_from(edges)

pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

cliques = list(nx.find_cliques(model))
print("cliques:")
print(cliques)

#1b
print("1b")
coefficients = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5}

def phi(clique, assignment):
    s = sum(coefficients[v] * assignment[v] for v in clique)
    return np.exp(s)

variables = ["A1", "A2", "A3", "A4", "A5"]
states = [-1, 1]
all_states = list(itertools.product(states, repeat=5))

joint = {}
for s in all_states:
    assignment = dict(zip(variables, s))
    prob = 1
    for clique in cliques:
        prob *= phi(clique, assignment)
    joint[tuple(s)] = prob

Z = sum(joint.values())
joint = {key: val / Z for key, val in joint.items()}

best_config = max(joint, key=joint.get)
max_prob = joint[best_config]

print("joint distribution")
for key, val in joint.items():
    print(key, val)

for var, val in zip(variables, best_config):
    print(f"{var} = {val}")
print(f"max probability: {max_prob}")


print("2")
size = 5
lambda_reg = 2.0
noise_fraction = 0.1
pixel_values = [0, 1]


original_image = np.random.choice(pixel_values, size=(size, size))
print("original\n", original_image)

noisy_image = original_image.copy()
num_noisy = int(size * size * noise_fraction)
indices = random.sample(range(size * size), num_noisy)

for idx in indices:
    i, j = divmod(idx, size)
    noisy_image[i, j] = 1 - noisy_image[i, j]

print("noisy\n", noisy_image)


model = MarkovNetwork()
variables = [f"X{i}{j}" for i in range(size) for j in range(size)]
model.add_nodes_from(variables)


for i in range(size):
    for j in range(size):
        if i < size - 1:
            model.add_edge(f"X{i}{j}", f"X{i+1}{j}")
        if j < size - 1:
            model.add_edge(f"X{i}{j}", f"X{i}{j+1}")


factors = []
#forall i-> exp(-lambda (x_i - y_i)^2)
for i in range(size):
    for j in range(size):
        var = f"X{i}{j}"
        y_ij = noisy_image[i, j]
        values = [np.exp(-lambda_reg * (x - y_ij) ** 2) for x in pixel_values]
        factors.append(DiscreteFactor([var], [2], values))

#forall ij-> exp(-(x_i - x_j)^2)
for i in range(size):
    for j in range(size):
        var = f"X{i}{j}"
        if i < size - 1:
            var2 = f"X{i+1}{j}"
            vals = np.zeros((2, 2))
            for xi in pixel_values:
                for xj in pixel_values:
                    vals[xi, xj] = np.exp(-((xi - xj) ** 2))
            factors.append(DiscreteFactor([var, var2], [2, 2], vals.flatten()))
        if j < size - 1:
            var2 = f"X{i}{j+1}"
            vals = np.zeros((2, 2))
            for xi in pixel_values:
                for xj in pixel_values:
                    vals[xi, xj] = np.exp(-((xi - xj) ** 2))
            factors.append(DiscreteFactor([var, var2], [2, 2], vals.flatten()))


model.add_factors(*factors)
bp = BeliefPropagation(model)
map_result = bp.map_query(variables=variables)

denoised_image = np.zeros((size, size), dtype=int)
for i in range(size):
    for j in range(size):
        denoised_image[i, j] = map_result[f"X{i}{j}"]

print("\nMAP estimate\n", denoised_image)

