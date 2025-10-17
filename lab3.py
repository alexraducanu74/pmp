from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

model = DiscreteBayesianNetwork([
    ('S', 'O'),
    ('S', 'L'),
    ('S', 'M'),
    ('L', 'M')
])

cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])

cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3],
                           [0.1, 0.7]],
                   evidence=['S'],
                   evidence_card=[2])

cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['S'],
                   evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[
                       [0.8, 0.4, 0.4, 0.1],
                       [0.2, 0.6, 0.6, 0.9]
                   ],
                   evidence=['S', 'L'],
                   evidence_card=[2, 2])


model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)
assert model.check_model()

print("1a)")
print(model.get_independencies())

infer = VariableElimination(model)
result = infer.query(variables=['S'], evidence={'O': 1, 'L': 1, 'M': 1})
print("1b)")
print(result)


if result.values[1] > result.values[0]:
    print("spam")
else:
    print("not spam")

print("2)")

model = DiscreteBayesianNetwork([
    ('roll', 'added'),
    ('added', 'drawn')
])

cpd_roll = TabularCPD(
    variable='roll',
    variable_card=6,
    values=[
        [1/6],
        [1/6],
        [1/6],
        [1/6],
        [1/6],
        [1/6]
    ]
)

cpd_added = TabularCPD(
    variable='added',
    variable_card=3,
    evidence=['roll'],
    evidence_card=[6],
    values=[
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
    ]
)

cpd_drawn = TabularCPD(
    variable='drawn',
    variable_card=3,
    evidence=['added'],
    evidence_card=[3],
    values=[
        [0.4, 0.3, 0.3],
        [0.4, 0.5, 0.4],
        [0.2, 0.2, 0.3],
    ]
)


model.add_cpds(cpd_roll, cpd_added, cpd_drawn)
assert model.check_model()
infer = VariableElimination(model)
prob_red = infer.query(variables=['drawn'])


print(f"lab curent: {prob_red.values[0]}")
print("lab trecut: aprox. 0.299")

print("3a)")

def simulate_game(simulation_cnt):
    p0_wins = 0
    p1_wins = 0

    for _ in range(simulation_cnt):
        starter = np.random.choice(['P0', 'P1'])

        n = np.random.randint(1, 7)

        if starter == 'P0':
            m = np.random.binomial(2 * n, 4 / 7)
            if n >= m:
                p0_wins += 1
            else:
                p1_wins += 1
        else:
            m = np.random.binomial(2 * n, 0.5)
            if n >= m:
                p1_wins += 1
            else:
                p0_wins += 1

    return p0_wins, p1_wins

p0_wins, p1_wins = simulate_game(10000)

print(f"P0: {p0_wins/(p0_wins + p1_wins)}")
print(f"P1: {p1_wins/(p0_wins + p1_wins)}")
print("raspuns: P1")