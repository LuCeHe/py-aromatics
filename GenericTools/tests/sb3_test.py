import os, argparse, time

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import RecurrentPPO, QRDQN, TRPO

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
os.makedirs(EXPERIMENTS, exist_ok=True)


model_dict = {
    'RecurrentPPO': lambda env, seed: RecurrentPPO("MlpLstmPolicy", env, verbose=1, seed=seed),
    # 'CnnRecurrentPPO': lambda env, seed: RecurrentPPO("CnnLstmPolicy", env, verbose=1, seed=seed),
    'PPO': lambda env, seed: PPO("MlpPolicy", env, verbose=1, seed=seed),
    'A2C': lambda env, seed: A2C("MlpPolicy", env, verbose=1, seed=seed),
    'entA2C': lambda env, seed: A2C("MlpPolicy", env, verbose=1, seed=seed, ent_coef=.5),
    'QRDQN': lambda env, seed: QRDQN("MlpPolicy", env, verbose=1, seed=seed),
    'expQRDQN': lambda env, seed: QRDQN("MlpPolicy", env, verbose=1, seed=seed, exploration_final_eps=.05),
    'TRPO': lambda env, seed: TRPO("MlpPolicy", env, verbose=1, seed=seed),
}


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments", default='test', type=str, help="String to activate extra behaviors")
    parser.add_argument("--modelname", default='A2C', type=str, help="Model to train",
                        choices=model_dict.keys())
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--timesteps", default=10000, type=int, help="Time Steps")
    args = parser.parse_args()

    return args


def main(args, results):
    env_train = gym.make("CartPole-v1")

    # Wrap the environment
    print('Monitoring environments...')
    os.makedirs(EXPERIMENTS, exist_ok=True)

    env_train = Monitor(env_train, EXPERIMENTS)

    print('Training...')
    model = model_dict[args.modelname](env_train, args.seed)
    callbacks = []
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    return results


if __name__ == "__main__":
    results = {}
    time_start = time.perf_counter()

    args = get_argparse()

    # train
    results = main(args, results)

    time_elapsed = (time.perf_counter() - time_start)
    print('All done, in ' + str(time_elapsed) + 's')
