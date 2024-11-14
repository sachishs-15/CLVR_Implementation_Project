import gym
from gym.envs.registration import register

def make_env(gym_id, data_spec, seed):
    def thunk():
        env = gym.make(gym_id)
        env.set_config(data_spec)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        return env
    return thunk