from cldt.policy import Policy


class ReachPolicy(Policy):
    """
    A policy for Reach environment that uses a simple formula to calculate the action.
    """

    def act(self, obs):
        current_position = obs["observation"][0:3]
        desired_position = obs["desired_goal"][0:3]
        act = 5.0 * (desired_position - current_position)

        return act
