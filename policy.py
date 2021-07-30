from acme import specs
from acme.agents.tf import actors
from acme.tf import networks
from acme.tf import utils as tf2_utils
import trfl
import numpy as np
import sonnet as snt
import agentos


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
    @classmethod
    def ready_to_initialize(cls, shared_data):
        return "environment_spec" in shared_data

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        environment_spec = self.shared_data["environment_spec"]
        network = BasicRNN(environment_spec.actions)
        agentos.restore_tensorflow("network", network)
        self.shared_data["network"] = network
        self.shared_data["store_lstm_state"] = True

        # TODO - Most of this configuration should be in the agent.ini

        # ======================
        # Create the R2D2 agent.
        # start agent code from acme/agents/tf/r2d2/agent.py
        # ======================
        # BURN_IN_LENGTH = 2
        # TRACE_LENGTH = 10
        # REPLAY_PERIOD = 40
        # COUNTER = None
        # logger = loggers.TerminalLogger(label="agent", time_delta=10.0)
        # BATCH_SIZE = 32
        # PREFETCH_SIZE = tf.data.experimental.AUTOTUNE
        # TARGET_UPDATE_PERIOD = 25
        # IMPORTANCE_SAMPLING_EXPONENT = 0.2
        # PRIORITY_EXPONENT = 0.6
        # LEARNING_RATE = 1e-3
        # MIN_REPLAY_SIZE = 50
        # MAX_REPLAY_SIZE = 500
        # SAMPLES_PER_INSERT = 32.0
        # STORE_LSTM_STATE = True
        # MAX_PRIORITY_WEIGHT = 0.9
        # CHECKPOINT = False

        # initial_state = self.network.initial_state(1)
        # extra_spec = {
        #     "core_state": tf2_utils.squeeze_batch_dim(initial_state),
        # }
        # sequence_length = BURN_IN_LENGTH + TRACE_LENGTH + 1
        # replay_table = reverb.Table(
        #     name=adders.DEFAULT_PRIORITY_TABLE,
        #     sampler=reverb.selectors.Prioritized(PRIORITY_EXPONENT),
        #     remover=reverb.selectors.Fifo(),
        #     max_size=MAX_REPLAY_SIZE,
        #     rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        #     signature=adders.SequenceAdder.signature(
        #         environment_spec, extra_spec, sequence_length=sequence_length
        #     ),
        # )

        # # NB - must save ref to server or it gets killed
        # self.reverb_server = reverb.Server([replay_table], port=None)
        # address = f"localhost:{self.reverb_server.port}"

        # # Component to add things into replay.
        # adder = adders.SequenceAdder(
        #     client=reverb.Client(address),
        #     period=REPLAY_PERIOD,
        #     sequence_length=sequence_length,
        # )

        # # The dataset object to learn from.
        # dataset = datasets.make_reverb_dataset(
        #     server_address=address,
        #     batch_size=BATCH_SIZE,
        #     prefetch_size=PREFETCH_SIZE,
        # )

        # target_network = copy.deepcopy(self.network)
        # tf2_utils.create_variables(
        #     target_network, [environment_spec.observations]
        # )

        # self.learner = learning.R2D2Learner(
        #     environment_spec=environment_spec,
        #     network=self.network,
        #     target_network=target_network,
        #     burn_in_length=BURN_IN_LENGTH,
        #     sequence_length=sequence_length,
        #     dataset=dataset,
        #     reverb_client=reverb.TFClient(address),
        #     counter=COUNTER,
        #     logger=logger,
        #     discount=self.discount,
        #     target_update_period=TARGET_UPDATE_PERIOD,
        #     importance_sampling_exponent=IMPORTANCE_SAMPLING_EXPONENT,
        #     max_replay_size=MAX_REPLAY_SIZE,
        #     learning_rate=LEARNING_RATE,
        #     store_lstm_state=STORE_LSTM_STATE,
        #     max_priority_weight=MAX_PRIORITY_WEIGHT,
        # )

        EPSILON = 0.01
        tf2_utils.create_variables(network, [environment_spec.observations])

        def epsilon_greedy_fn(qs):
            return trfl.epsilon_greedy(qs, epsilon=EPSILON).sample()

        policy_network = snt.DeepRNN(
            [
                network,
                epsilon_greedy_fn,
            ]
        )
        ADDER = None
        self.actor = actors.RecurrentActor(
            policy_network,
            ADDER,
            store_recurrent_state=self.shared_data["store_lstm_state"],
        )

    def decide(self, observation, actions, should_learn=False):
        # TODO - ugly typing
        if not isinstance(observation, type(np.array)):
            observation = np.array(observation)
        if observation.dtype != np.zeros(1, dtype="float32").dtype:
            observation = np.float32(observation)
        action = self.actor.select_action(observation)
        self.shared_data["_prev_state"] = self.actor._prev_state
        return action
