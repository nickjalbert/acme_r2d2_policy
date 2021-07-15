# Run me:
#           python test.py

from policy import R2D2Policy
import os
import numpy as np
from dm_env import specs
from collections import namedtuple
from pathlib import Path
import agentos


data_location = Path('.').absolute()
output_file = data_location / 'network'

try:
    os.remove(output_file)
except FileNotFoundError:
    pass

# TODO - setup to simulate AOS runtime env
if 'saved_data' not in agentos.__dict__:
    agentos.__dict__['saved_data'] = {}

agentos.__dict__['save_data'].__dict__['data_location'] = data_location

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
        discounts=specs.BoundedArray(
            shape=(),
            dtype=np.dtype('float32'),
            name='discount',
            minimum=0.0,
            maximum=1.0
        )
)

policy = R2D2Policy(spec)

print()
print('Calling policy.improve()')
assert not os.path.isfile(output_file)
policy.improve()
assert os.path.isfile(output_file)
os.remove(output_file)
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

print('Calling policy.observe()')
# Trajectory One
policy.observe(None, obs, np.int32(10), False, {})
policy.observe(np.int32(0), obs, np.int32(10), False, {})
policy.observe(np.int32(0), obs, np.int32(10), True, {})
# Trajectory Two
policy.observe(None, obs, np.int32(11), False, {})
policy.observe(np.int32(1), obs, np.int32(11), False, {})
policy.observe(np.int32(1), obs, np.int32(11), True, {})
