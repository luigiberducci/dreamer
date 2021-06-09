import functools
import pathlib
import numpy as np
import tools
import wrappers
from dreamer import Dreamer, define_config
import argparse
import gym
import tensorflow as tf

#tf.config.run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=pathlib.Path, required=True)
args = parser.parse_args()

config = define_config()
config.task = "dmc_walker_walk"
config.action_repeat = 2
config.time_limit = 1000
config.log_scalar = False
config.log_images = False
config.logdir = args.logdir

datadir = config.logdir / 'episodes'
suite, task = config.task.split('_', 1)

env = wrappers.DeepMindControl(task)
env = wrappers.ActionRepeat(env, config.action_repeat)
env = wrappers.NormalizeActions(env)
env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
env = wrappers.RewardObs(env)


agent = Dreamer(config, datadir, env.action_space, writer=None)
if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
else:
    raise FileNotFoundError(f"missing checkpoint in {config.logdir}")

# simulation
step, episode = 0, 0
done = False
obs = None
agent_state = None

env.reset()
env.render()

actions = []
while not done:
    action, agent_state = agent(obs, done, agent_state)
    action = np.array(action)
    actions.append(action)
    obs, reward, done, info = env.step(action)
    env.render()
    obs = list(obs)
