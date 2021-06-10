import functools
import pathlib
import numpy as np
import tools
import wrappers
from dreamer import Dreamer, define_config
import argparse
import gym
import os
import sys
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

sys.path.append(str(pathlib.Path(__file__).parent))

import wrappers

#tf.config.run_functions_eagerly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=pathlib.Path, required=True)
parser.add_argument("--action_dist", type=str, choices=["tanh_normal", "norm_policy", "diag_policy"],
                    default="tanh_normal")
args = parser.parse_args()

config = define_config()
config.task = "dmc_walker_walk"
config.action_repeat = 2
config.time_limit = 1000
config.logdir = args.logdir
config.action_dist = args.action_dist

if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
assert config.precision in (16, 32), config.precision
if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))

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
    agent.load(config.logdir / 'variables.pkl', testing=True)
else:
    raise FileNotFoundError(f"missing checkpoint in {config.logdir}")

# simulation
step, episode = 0, 0
done = False
agent_state = None
obs = env.reset()

actions = []
tot_reward = 0.0
while not done:
    obs = {k: np.stack([obs[k]]) for k in obs}
    action, agent_state = agent(obs, np.array([done]), agent_state, training=False)
    action = np.array(action)
    obs, reward, done, info = env.step(action)
    #
    actions.append(action)
    tot_reward += reward
actions = np.squeeze(np.array(actions))
import matplotlib.pyplot as plt
for i in range(actions.shape[1]):
    plt.clf()
    plt.plot(np.arange(actions.shape[0]), actions[:, i])
    plt.title(f"Action: Component {i}")
    plt.savefig(f"{config.logdir}/{task}_reward_{int(tot_reward)}_action_{i}.pdf")
