from acme import environment_loop
from acme import specs
from acme import wrappers
from acme import datasets
from acme.agents.tf import r2d2
from acme.agents.tf.r2d2 import learning
from acme.agents.tf import d4pg
from acme.agents.tf import actors
from acme.adders import reverb as adders
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from dm_env import TimeStep
from dm_env import StepType
import reverb
import trfl
import copy
import numpy as np
import sonnet as snt
import gym
import pyvirtualdisplay
import imageio
import base64
import agentos
import tensorflow as tf


# BasicRNN, taken from r2d2 test
# https://github.com/deepmind/acme/blob/master/acme/agents/tf/r2d2/agent_test.py
class BasicRNN(networks.RNNCore):
    def __init__(self, action_spec: specs.DiscreteArray):
        super().__init__(name="basic_r2d2_RNN_network")
        self._net = snt.DeepRNN(
            [
                snt.Flatten(),
                snt.VanillaRNN(16),
                snt.VanillaRNN(16),
                snt.VanillaRNN(16),
                snt.nets.MLP([16, 16, action_spec.num_values]),
            ]
        )

    def __call__(self, inputs, state):
        return self._net(inputs, state)

    def initial_state(self, batch_size: int, **kwargs):
        return self._net.initial_state(batch_size)

    def unroll(self, inputs, state, sequence_length):
        return snt.static_unroll(self._net, inputs, state, sequence_length)


class R2D2Policy(agentos.Policy):
    def __init__(self, environment_spec, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = BasicRNN(environment_spec.actions)
        agentos.restore_tensorflow('network', self.network)
        # TODO - Most of this configuration should be in the agent.ini
        self.num_observations = 0
        self.discount = np.float32(.99)

        # ======================
        # Create the R2D2 agent.
        # start agent code from acme/agents/tf/r2d2/agent.py
        # ======================
        BURN_IN_LENGTH = 2
        TRACE_LENGTH = 10
        REPLAY_PERIOD = 40
        COUNTER = None
        logger = loggers.TerminalLogger(label="agent", time_delta=10.0)
        BATCH_SIZE = 32
        PREFETCH_SIZE = tf.data.experimental.AUTOTUNE
        TARGET_UPDATE_PERIOD = 25
        IMPORTANCE_SAMPLING_EXPONENT = 0.2
        PRIORITY_EXPONENT = 0.6
        EPSILON = 0.01
        LEARNING_RATE = 1e-3
        MIN_REPLAY_SIZE = 50
        MAX_REPLAY_SIZE = 500
        SAMPLES_PER_INSERT = 32.0
        STORE_LSTM_STATE = True
        MAX_PRIORITY_WEIGHT = 0.9
        CHECKPOINT = False

        initial_state = self.network.initial_state(1)
        extra_spec = {
            'core_state': tf2_utils.squeeze_batch_dim(initial_state),
        }
        sequence_length = BURN_IN_LENGTH + TRACE_LENGTH + 1
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Prioritized(PRIORITY_EXPONENT),
            remover=reverb.selectors.Fifo(),
            max_size=MAX_REPLAY_SIZE,
            rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
            signature=adders.SequenceAdder.signature(
                environment_spec, extra_spec, sequence_length=sequence_length))

        # NB - must save ref to server or it gets killed
        self.reverb_server = reverb.Server([replay_table], port=None)
        address = f'localhost:{self.reverb_server.port}'


        # Component to add things into replay.
        adder = adders.SequenceAdder(
            client=reverb.Client(address),
            period=REPLAY_PERIOD,
            sequence_length=sequence_length,
        )

        # The dataset object to learn from.
        dataset = datasets.make_reverb_dataset(
            server_address=address,
            batch_size=BATCH_SIZE,
            prefetch_size=PREFETCH_SIZE)

        target_network = copy.deepcopy(self.network)
        tf2_utils.create_variables(
                self.network, [environment_spec.observations]
        )
        tf2_utils.create_variables(
                target_network, [environment_spec.observations]
        )

        self.learner = learning.R2D2Learner(
            environment_spec=environment_spec,
            network=self.network,
            target_network=target_network,
            burn_in_length=BURN_IN_LENGTH,
            sequence_length=sequence_length,
            dataset=dataset,
            reverb_client=reverb.TFClient(address),
            counter=COUNTER,
            logger=logger,
            discount=self.discount,
            target_update_period=TARGET_UPDATE_PERIOD,
            importance_sampling_exponent=IMPORTANCE_SAMPLING_EXPONENT,
            max_replay_size=MAX_REPLAY_SIZE,
            learning_rate=LEARNING_RATE,
            store_lstm_state=STORE_LSTM_STATE,
            max_priority_weight=MAX_PRIORITY_WEIGHT,
        )

        self.checkpointer = tf2_savers.Checkpointer(
            subdirectory='r2d2_learner',
            time_delta_minutes=60,
            objects_to_save=self.learner.state,
            enable_checkpointing=CHECKPOINT,
        )
        self.snapshotter = tf2_savers.Snapshotter(
            objects_to_save={'network': self.network}, time_delta_minutes=60.)

        def epsilon_greedy_fn(qs):
            return trfl.epsilon_greedy(qs, epsilon=EPSILON).sample()

        policy_network = snt.DeepRNN([
            self.network,
            epsilon_greedy_fn,
        ])

        self.actor = actors.RecurrentActor(
            policy_network, adder, store_recurrent_state=STORE_LSTM_STATE)
        self.observations_per_step = (
            float(REPLAY_PERIOD * BATCH_SIZE) / SAMPLES_PER_INSERT
        )

        self.min_observations = (
                REPLAY_PERIOD * max(BATCH_SIZE, MIN_REPLAY_SIZE)
        )


    def observe(self, action, observation, reward, done, info):
        if action is None:  # No action -> first step
            timestep = TimeStep(StepType.FIRST, None, None, observation)
            self.actor.observe_first(timestep)
        else:
            if done:
                timestep = TimeStep(
                        StepType.LAST, reward, self.discount, observation
                )
            else:
                timestep = TimeStep(
                        StepType.MID, reward, self.discount, observation
                )

            self.num_observations += 1
            self.actor.observe(action, next_timestep=timestep)

    def decide(self, observation, actions, should_learn=False):
        # TODO - ugly typing
        if type(observation) != type(np.array):
            observation = np.array(observation)
        if observation.dtype != np.zeros(1, dtype='float32').dtype:
            observation = np.float32(observation)
        return self.actor.select_action(observation)

    def improve(self, **kwargs):
        # ======================
        # improve the R2D2 agent.
        # code from:
        #   * acme/agents/agent.py
        #   * acme/agents/tf/r2d2/agent.py
        # ======================


        num_steps = 0
        n = self.num_observations - self.min_observations
        if n < 0:
            # Do not do any learner steps until you have seen min_observations.
            num_steps = 0
        if self.observations_per_step > 1:
            # One batch every 1/obs_per_step observations, otherwise zero.
            num_steps = int(n % int(self.observations_per_step) == 0)
        else:
            # Always return 1/obs_per_step batches every observation.
            num_steps = int(1 / self.observations_per_step)


        for _ in range(num_steps):
          # Run learner steps (usually means gradient steps).
          self.learner.step()
        if num_steps > 0:
          # Update the actor weights when learner updates.
          self.actor.update()

        # TODO - can probably skip builtin snapper/checker
        self.snapshotter.save()
        self.checkpointer.save()

        agentos.save_tensorflow('network', self.network)
