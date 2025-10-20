from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import math


#3.2
model = DiscreteBayesianNetwork([
    ('starter', 'die'),
    ('die', 'heads')
])

cpd_starter = TabularCPD(
    variable='starter',
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={'starter': ['P0', 'P1']}
)

cpd_die = TabularCPD(
    variable='die',
    variable_card=6,
    values=[
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
    ],
    evidence=['starter'],
    evidence_card=[2],
    state_names={'die': [1, 2, 3, 4, 5, 6], 'starter': ['P0', 'P1']}
)

heads_values = [[0.0 for _ in range(6)] for _ in range(13)]
for n in range(1, 7):
    total_flips = 2 * n
    for m in range(13):
        if m <= total_flips:
            prob = math.comb(total_flips, m) * (0.5 ** total_flips)
        else:
            prob = 0.0
        heads_values[m][n-1] = prob

cpd_heads = TabularCPD(
    variable='heads',
    variable_card=13,
    values=heads_values,
    evidence=['die'],
    evidence_card=[6],
    state_names={'heads': list(range(13)), 'die': [1, 2, 3, 4, 5, 6]}
)

model.add_cpds(cpd_starter, cpd_die, cpd_heads)
model.check_model()

#3.3
result = VariableElimination(model)
probs = result.query(variables=['starter'], evidence={'heads': 1})

print(probs.state_names['starter'][0],":", probs.values[0])
print(probs.state_names['starter'][1],":", probs.values[1])
print("P0=P1=0.5")
