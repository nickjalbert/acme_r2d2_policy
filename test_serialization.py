from dm_env import specs
from policy import BasicRNN
from collections import namedtuple
from acme.tf import utils
from acme.tf.savers import Snapshotter
import numpy as np
import tensorflow as tf
import pickle
import sys

loaded = tf.saved_model.load('./0c7c8136-e619-11eb-9062-00155dc4fd46/snapshots/network/')
print(loaded)
print(loaded.submodules)
print(list(loaded.signatures.keys()))  # ["serving_default"]
sys.exit(0)

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


network = BasicRNN(spec.actions)
utils.create_variables(network, [spec.observations])

print()
print(network([np.float32(1), np.float32(1), np.float32(1), np.float32(1)]))
print()

SEARCHABLE_TEST_FILE_NAME = 'test203948230.out'

with open(SEARCHABLE_TEST_FILE_NAME, "wb") as f:
    pickle.dump(network, f)


with open(SEARCHABLE_TEST_FILE_NAME, "rb") as f:
    network2 = pickle.load(f)

#snapshotter = Snapshotter(
#        objects_to_save={'network': network},
#        directory='.',
#)

#snapshotter.save(force=True)

