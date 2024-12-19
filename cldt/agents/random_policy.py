from cldt.agent import Agent


class RandomPolicy(Agent):
    action_space = None

    """
    A policy that takes random actions.
    """

    def learn_online(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return self.action_space.sample()

    @staticmethod
    def load(load_path, env=None):
        policy = RandomPolicy()
        policy.action_space = env.action_space

        return policy
