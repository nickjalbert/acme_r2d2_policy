# Run me:
#           python test.py

from policy import R2D2Policy
import numpy as np
from dm_env import specs
from collections import namedtuple


# TODO - Maybe this goes into AOS?
# TODO - copied directly from cartpole
# https://github.com/deepmind/acme/blob/master/acme/specs.py
EnvironmentSpec = namedtuple(
        'EnvironmentSpec', ['observations', 'actions', 'rewards', 'discounts']
)

spec = EnvironmentSpec(
        observations=specs.Array(
            shape=(4,),
            dtype=np.float32
        ),
        actions=specs.DiscreteArray(num_values=2),
        rewards=specs.DiscreteArray(num_values=2),
        discounts=specs.DiscreteArray(num_values=2)
)

policy = R2D2Policy(spec)

print()
print('Calling policy.improve()')
policy.improve()
print()

obs = np.float32(np.array([-0.10803831, -0.39921515,  0.21240302,  0.85913184]))
actions = [0, 1]
print('Calling policy.decide()')
print(f'\tChosen action: {policy.decide(obs, actions)}')
print()

# This should auto-cast the list to the correct type
print('Calling policy.decide() with a basic Python list')
obs2 = [-0.10803831, -0.39921515,  0.21240302,  0.85913184]
actions = [0, 1]
print(f'\tChosen action: {policy.decide(obs2, actions)}')
print()
