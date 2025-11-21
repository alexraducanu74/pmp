from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
from hmmlearn.hmm import CategoricalHMM


#1a)
model = DiscreteBayesianNetwork([
    ('O', 'H'),
    ('O', 'W'),
    ('W', 'R'),
    ('H', 'R'),
    ('H', 'E'),
    ('R', 'C')
])

cpd_o = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]])

cpd_h = TabularCPD(variable='H', variable_card=2,
                   values=[[0.9, 0.2],
                           [0.1, 0.8]],
                   evidence=['O'],
                   evidence_card=[2])

cpd_w = TabularCPD(variable='W', variable_card=2,
                   values=[[0.1, 0.6],
                           [0.9, 0.4]],
                   evidence=['O'],
                   evidence_card=[2])

cpd_r = TabularCPD(variable='R', variable_card=2,
                   values=[
                       [0.5, 0.9, 0.3, 0.6],
                       [0.5, 0.1, 0.7, 0.4]
                   ],
                   evidence=['H', 'W'],
                   evidence_card=[2, 2])


cpd_e = TabularCPD(variable='E', variable_card=2,
                   values=[[0.8, 0.2],
                           [0.2, 0.8]],
                   evidence=['H'],
                   evidence_card=[2])



cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.85, 0.40],
                           [0.15, 0.60]],
                   evidence=['R'],
                   evidence_card=[2])

model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)
assert model.check_model()


infer = VariableElimination(model)
result1 = infer.query(variables=['H'], evidence={'C': 0})
result2 = infer.query(variables=['E'], evidence={'C': 0})
result3 = infer.query(variables=['H', 'W'], evidence={'C': 0})
print("1b)")
print("P(H = yes| C = comfortable)",result1.values[0])
print("P(E = high | C = comfortable)", result2.values[0])
maxp = -1
maxp=max(result3.values[0][0],result3.values[0][1],result3.values[1][0],result3.values[1][1])
print("MAP:", maxp)
print(model.get_independencies())


print("1c")
print("d-connected: ",model.is_dconnected('W', 'E', observed=['H']))
print("(W ⟂ E | H) : no")
print("(O ⟂ C | R) : yes")
print("justificare: din model.get_independencies() de mai sus / din markov blankets + parinti din graf")

###2
print("\n2\n")

#a
states = ["W", "R", "S"]
pi = np.array([0.4, 0.3, 0.3])
A = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5]
])

B = np.array([
    [0.1, 0.7, 0.2],
    [0.05, 0.25, 0.7],
    [0.8, 0.15, 0.05]
])



model = CategoricalHMM(n_components=3)
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

#b
# L=0, M=1, H=2
obs = np.array([[1, 2, 0]]).T
logprob = model.score(obs)
prob = np.exp(logprob)
print("b")
print(f"log prob: {logprob}")
print(f"prob: {prob}")
#c
print("c")
_, hidden_states = model.decode(obs, algorithm="viterbi")
print([states[i] for i in hidden_states])
print("viterbi scales better with the size of the sequence i.e. it has better time complexity and"
      " it gets progressively better than brute force as we increase the dimension of the sequence")

#d
cnt = 0
for _ in range(10000):
    _, X = model.sample(3)
    if X[0] == 1 and X[1] == 2 and X[2] == 0:
        cnt = cnt + 1

print("d) empirical:", cnt/10000, f"forward:{prob}", "<=> forward prob > empirical prob")
