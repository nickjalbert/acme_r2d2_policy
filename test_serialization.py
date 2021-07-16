from dm_env import specs
from policy import BasicRNN
from collections import namedtuple
import numpy as np
import pickle

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


network = BasicRNN(specs.DiscreteArray(num_values=2))


SEARCHABLE_TEST_FILE_NAME = 'test203948230.out'

with open(SEARCHABLE_TEST_FILE_NAME, "wb") as f:
    pickle.dump(network, f)


with open(SEARCHABLE_TEST_FILE_NAME, "rb") as f:
    network2 = pickle.load(f)


