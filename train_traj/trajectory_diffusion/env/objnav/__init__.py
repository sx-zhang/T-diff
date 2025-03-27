from gym.envs.registration import register
import trajectory_diffusion.env.objnav

register(
    id='objnav-traj-diff-v0',
    entry_point='envs.objnav.objnav_keypoints_env:ObjNavKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)