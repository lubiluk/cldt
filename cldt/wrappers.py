import gymnasium as gym
import numpy as np


class StepAPICompatibility(gym.Wrapper):
    """This class is said to be implemented in gymnasium, but it is not."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, ter, tru, info = self.env.step(action)
        done = ter or tru
        return obs, reward, done, info


class TrajectoryRecorder(gym.Wrapper):
    def __init__(self, env):
        self.trajectories = []
        self.dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
        self.obs_keys = list(env.observation_space.keys()) if self.dict_obs else None
        self._trajectory = None
        self._last_obs = None
        super().__init__(env)

    def reset(self):
        self._trajectory = {
            "observations": {k: [] for k in self.obs_keys} if self.dict_obs else [],
            "actions": [],
            "rewards": [],
            "next_observations": (
                {k: [] for k in self.obs_keys} if self.dict_obs else []
            ),
            "terminals": [],
        }
        self.trajectories.append(self._trajectory)
        obs, info = self.env.reset()
        self._last_obs = obs

        return obs, info

    def step(self, action):
        res = self.env.step(action)

        if len(res) == 4:
            obs, reward, done, info = res
        else:
            obs, reward, ter, tru, info = res
            done = ter or tru

        if self.dict_obs:
            for k in self._last_obs.keys():
                self._trajectory["observations"][k].append(self._last_obs[k])
                self._trajectory["next_observations"][k].append(obs[k])
        else:
            self._trajectory["observations"].append(self._last_obs)
            self._trajectory["next_observations"].append(obs)

        self._trajectory["actions"].append(action)
        self._trajectory["rewards"].append(reward)
        self._trajectory["terminals"].append(done)
        self._last_obs = obs

        return obs, reward, done, info

    def numpy_trajectories(self):
        np_trajectories = []

        for path in self.trajectories:
            if self.dict_obs:
                observations = {
                    k: np.array(path["observations"][k]) for k in self.obs_keys
                }
                next_observations = {
                    k: np.array(path["next_observations"][k]) for k in self.obs_keys
                }
            else:
                observations = np.array(path["observations"])
                next_observations = np.array(path["next_observations"])

            actions = np.array(path["actions"])
            rewards = np.array(path["rewards"])
            terminals = np.array(path["terminals"])

            np_trajectories.append(
                {
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "next_observations": next_observations,
                    "terminals": terminals,
                }
            )

        return np_trajectories
