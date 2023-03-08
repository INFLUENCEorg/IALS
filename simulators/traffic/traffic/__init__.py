from gym.envs.registration import register
register(
    id='global-traffic-v0',
    entry_point='traffic.envs:GlobalTraffic',
)
register(
    id='local-traffic-v0',
    entry_point='traffic.envs:LocalTraffic',
)