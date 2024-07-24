from cldt.policies import Policy


class RandomPolicy(Policy):
    def evaluate(self, env, num_episodes=1, max_ep_len=None, record_trajectories=False):
        trajectories = []
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            ep_len = 0
            ep_ret = 0
            while not done:
                act = env.action_space.sample()
                obs2, rew, done, _ = env.step(act)
                ep_ret += rew
                ep_len += 1
                if record_trajectories:
                    trajectories.append((obs, act, rew, obs2, done))
                obs = obs2
                if max_ep_len and ep_len >= max_ep_len:
                    break
        return {"ep_len": ep_len, "ep_ret": ep_ret, "trajectories": trajectories}