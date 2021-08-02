from gym.envs.registration import register
register(
    id='tmaze-v0',
    entry_point='tmaze.envs:Tmaze',
)