from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, PPO, DQN, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

# Import our main environment
from env_gym import SimpleSimGym
# Environment Parameters
STARTING_BUDGET = 500
NUM_TARGETS = 2
PLAYER_FOV = 60
RENDER_MODE = "rgb_array"
# This simple toggle can be used to switch which environment we are training the notebook on
env_mode = 2  # 0 for GoLeft, 1 for GoDownLeft, 2 for SimpleSimGym


config = {
    "policy": 'MlpPolicy',
    "total_timesteps": 20_000,
    "logdir": "logs/",
    "savedir": "saved_models/"
}


# --------------------------------- SIMPLESIM -------------------------------- #
# Import our main environment
from env_gym import SimpleSimGym
# Environment Parameters
STARTING_BUDGET = 500
NUM_TARGETS = 2
PLAYER_FOV = 60
RENDER_MODE = "rgb_array"
# This simple toggle can be used to switch which environment we are training the notebook on
# env_mode = 2  # 0 for GoLeft, 1 for GoDownLeft, 2 for SimpleSimGym

env = SimpleSimGym(starting_budget=STARTING_BUDGET, num_targets=NUM_TARGETS, player_fov=PLAYER_FOV, render_mode=None)
vec_env = make_vec_env(SimpleSimGym, n_envs=1, monitor_dir=config["logdir"], env_kwargs=dict(starting_budget=STARTING_BUDGET, num_targets=NUM_TARGETS, player_fov=PLAYER_FOV, render_mode=RENDER_MODE))
# vec_env = VecFrameStack(vec_env, n_stack=4)
vec_env = VecFrameStack(vec_env, n_stack=1)

# ----------------------------------- ATARI ---------------------------------- #
# # There already exists an environment generator
# # that will make and wrap atari environments correctly.
# # Here we are also multi-worker training (n_envs=4 => 4 environments)
# vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# # Frame-stacking with 4 frames
# vec_env = VecFrameStack(vec_env, n_stack=4)


# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

# ----------------------------------- MODEL ---------------------------------- #
model = PPO("CnnPolicy", vec_env, verbose=1)
# model = DQN('MlpPolicy', vec_env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# model = ACER('CnnPolicy', vec_env, verbose=1)


# --------------------------------- TRAINING --------------------------------- #
model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
print("Done with training.")


# ----------------------------------- EVAL ----------------------------------- #
RENDER_MODE = "human"
env = SimpleSimGym(starting_budget=STARTING_BUDGET, num_targets=NUM_TARGETS, player_fov=PLAYER_FOV, render_mode=None)
eval_vec_env = make_vec_env(SimpleSimGym, n_envs=1, monitor_dir=config["logdir"], env_kwargs=dict(starting_budget=STARTING_BUDGET, num_targets=NUM_TARGETS, player_fov=PLAYER_FOV, render_mode=RENDER_MODE))

print("Running validation ...")
obs = eval_vec_env.reset()
i=0
while True:
    print(f"Step {i}")
    i+=1

    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = eval_vec_env.step(action)

    eval_vec_env.render()