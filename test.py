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
# Checkpoint files
output_file_1 = data_location / 'checkpoint'
output_file_2 = data_location / 'network-1.data-00000-of-00001'
output_file_3 = data_location / 'network-1.index'

try:
    os.remove(output_file_1)
    os.remove(output_file_2)
    os.remove(output_file_3)
except FileNotFoundError:
    pass

agentos.__dict__['save_data'].__dict__['data_location'] = data_location
agentos.__dict__['save_tensorflow'].__dict__['data_location'] = data_location
agentos.__dict__['restore_data'].__dict__['data_location'] = data_location
agentos.__dict__['restore_tensorflow'].__dict__['data_location'] = data_location

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
assert not os.path.isfile(output_file_1)
assert not os.path.isfile(output_file_2)
assert not os.path.isfile(output_file_3)
policy.improve()
assert os.path.isfile(output_file_1)
assert os.path.isfile(output_file_2)
assert os.path.isfile(output_file_3)
os.remove(output_file_1)
os.remove(output_file_2)
os.remove(output_file_3)
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
