from stable_baselines3 import HerReplayBuffer
from cldt.policy import Policy
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor, ResultsWriter


class TqcHerPolicy(Policy):
    def __init__(
        self,
        batch_size,
        buffer_size,
        gamma,
        learning_rate,
        policy,
        policy_kwargs,
        replay_buffer_kwargs,
        tau,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self.tau = tau

    def learn_online(self, env, n_timesteps, log_dir=None, device="auto"):
        if log_dir is not None:
            # Monitor the learning process
            env = Monitor(env, filename=log_dir)

        self.model = TQC(
            policy=self.policy,
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=self.replay_buffer_kwargs,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            policy_kwargs=self.policy_kwargs,
            tau=self.tau,
            verbose=1,
            device=device
        )

        self.model.learn(n_timesteps)

    def act(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path, env):
        tqc = TQC.load(path, env=env)
        policy = TqcHerPolicy(
            batch_size=tqc.batch_size,
            buffer_size=tqc.buffer_size,
            gamma=tqc.gamma,
            learning_rate=tqc.learning_rate,
            policy=tqc.policy,
            policy_kwargs=tqc.policy_kwargs,
            replay_buffer_kwargs=tqc.replay_buffer_kwargs,
            tau=tqc.tau,
        )
        policy.model = tqc

        return policy
