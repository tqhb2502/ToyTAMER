"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

# import asyncio
import gym

from tamer.agent import Tamer


# async def main():
def main():
    env = gym.make('MountainCar-v0', render_mode="human")
    # env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

    # hyperparameters
    # ====================================
    discount_factor = 1
    epsilon = 0.9  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 3
    tame = False  # set to false for vanilla Q learning
    # ====================================
    # discount_factor = 1
    # epsilon = 0  # vanilla Q learning actually works well with no random exploration
    # min_eps = 0
    # num_episodes = 3
    # tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.2

    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)

    # await agent.train(model_file_to_save='autosave')
    agent.train(model_file_to_save='autosave')
    # agent.play(n_episodes=1)
    agent.evaluate(n_episodes=5)


if __name__ == '__main__':
    # asyncio.run(main())
    main()
