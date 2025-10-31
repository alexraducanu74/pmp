import numpy as np
from hmmlearn.hmm import CategoricalHMM

#a
states = ["Difficult", "Medium", "Easy"]

A = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

B = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])

pi = np.array([1/3, 1/3, 1/3])

model = CategoricalHMM(n_components=3)
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B
print("desen: poza separata")
#b
# FB=0, B=1, S=2, NS=3
obs = np.array([[0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1]]).T
logprob = model.score(obs)
prob = np.exp(logprob)
print("b")
print(f"log prob: {logprob}")
print(f"prob: {prob}")


#c
print("c")
logprob_viterbi, hidden_states = model.decode(obs, algorithm="viterbi")
prob_viterbi = np.exp(logprob_viterbi)
print([states[i] for i in hidden_states])
print(f"prob:{prob_viterbi}")

