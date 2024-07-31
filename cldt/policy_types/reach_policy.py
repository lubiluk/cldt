from cldt.policies import Policy


class ReachPolicy(Policy):
    def evaluate(self, env, num_episodes=1, max_ep_len=None, render=False):
        returns = []
        ep_lens = []

        for _ in range(num_episodes):
            obs, info = env.reset()

            done = False
            ep_len = 0
            ep_ret = 0

            while not done:
                if render:
                    env.render()

                current_position = obs["observation"][0:3]
                desired_position = obs["desired_goal"][0:3]
                act = 5.0 * (desired_position - current_position)

                obs2, rew, done, info = env.step(act)

                ep_ret += rew
                ep_len += 1
                obs = obs2
                if max_ep_len and ep_len >= max_ep_len:
                    break

            returns.append(ep_ret)
            ep_lens.append(ep_len)
                
        return returns, ep_lens
