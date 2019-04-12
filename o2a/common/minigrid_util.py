# Common definitions for the minigrid env
import numpy as np

N_OBJS = 13
N_COLS = 6

pixels = []
for i in range(N_OBJS):
    for j in range(N_COLS):
        pixels.append((i,j,0))

pixel_to_categorical = {pix:i for i, pix in enumerate(pixels)}
num_pixels = len(pixels)

#Rewards: -0.5/env.max_steps, or 1
#Make this not hard coded....
ENV_MAX_STEPS = 144

# The mode I typically used was regular. These rewards will also be encoded as
# integers.
mode_rewards = {"regular": [-1./ENV_MAX_STEPS, 1]}
reward_to_categorical = {mode: {reward:i for i, reward in enumerate(mode_rewards[mode])} for mode in mode_rewards.keys()}

# Helper functions to convert between the encoded integers and the actual
# values.

def pix_to_target(next_states):
    target = []
    assert next_states.shape[-1] == 3

    for pixel in next_states.reshape(-1, 3):
        target.append(pixel_to_categorical[tuple([np.ceil(pixel[0]), np.ceil(pixel[1]), np.ceil(pixel[2])])])
    return target

def target_to_pix(imagined_states):
    pixels = []
    to_pixel = {value: key for key, value in pixel_to_categorical.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))

    return np.array(pixels)

def rewards_to_target(mode, rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_categorical[mode][reward])
    return target


