from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import r2d2
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from dm_env import TimeStep
from dm_env import StepType
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
        agentos.restore_data('network', self.network)

        # Create the R2D2 agent.
        agent_logger = loggers.TerminalLogger(label="agent", time_delta=10.0)
        self.agent = r2d2.R2D2(
                environment_spec=environment_spec,
                network=self.network,
                burn_in_length=2,
                trace_length=10,
                replay_period=40,
                logger=agent_logger,
                target_update_period=25,
                min_replay_size=50,
                max_replay_size=500,
                checkpoint=False,
        )

    def observe(self, action, observation, reward, done, info):
        # TODO - Where does discount go?  move to configuration params?
        # Acme expects it to be a part of the environment which
        # (overly) couples environment and agent
        DISCOUNT = np.float32(.99)
        if action is None:  # No action -> first step
            timestep = TimeStep(StepType.FIRST, None, None, observation)
            self.agent.observe_first(timestep)
        else:
            if done:
                timestep = TimeStep(
                        StepType.LAST, reward, DISCOUNT, observation
                )
            else:
                timestep = TimeStep(
                        StepType.MID, reward, DISCOUNT, observation
                )

            self.agent.observe(action, next_timestep=timestep)

    def decide(self, observation, actions, should_learn=False):
        # TODO - ugly typing
        if type(observation) != type(np.array):
            observation = np.array(observation)
        if observation.dtype != np.zeros(1, dtype='float32').dtype:
            observation = np.float32(observation)
        return self.agent.select_action(observation)

    def improve(self, **kwargs):
        self.agent.update()
        agentos.save_data('network', self.network)
