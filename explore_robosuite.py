from active_gym import make_active_robosuite_env, RobosuiteEnvArgs

import imageio
import numpy as np

def make_env(task_name, seed, **kwargs):
    def thunk():
        env_args = RobosuiteEnvArgs(
            task=task_name, 
            seed=seed, 
            obs_size=(64, 64), 
            return_camera_matrix=True,
            robots="Panda",
            camera_names=["active_view"],
            use_object_obs=True,
            **kwargs
        )
        env = make_active_robosuite_env(env_args)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

task_kwargs = dict(
    horizon=500,
    move_camera_scale=0.1, # 10cm at most
    rotate_camera_scale=2.0,
    renderer="mujoco",
)
env = make_env(task_name="Lift", seed=0, **task_kwargs)()
# Dict('active_view_image': Box(-1.0, 1.0, (3, 64, 64), float32), 'object-state': Box(-inf, inf, (10,), float32), 'robot0_proprio-state': Box(-inf, inf, (32,), float32))
print(env.observation_space)

# collect a video 
obs, info = env.reset()
img = obs['active_view_image'].transpose(1,2,0) * 255
video = [img]
for _ in range(10):
    # action = env.action_space.sample()
    # import ipdb; ipdb.set_trace()
    action = {'motor_action': np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
              'sensory_action': np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)}
    obs, rew, term, trunc, info = env.step(action)
    img = obs['active_view_image'].transpose(1,2,0) * 255
    video.append(img)
video = np.array(video, dtype=np.uint8)
imageio.mimsave("explore_robosuite.mp4", video, fps=5)