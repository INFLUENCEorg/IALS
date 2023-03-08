from gym.envs.registration import register
register(
    id='global-warehouse-v0',
    entry_point='warehouse.envs:GlobalWarehouse',
)
register(
    id='local-warehouse-v0',
    entry_point='warehouse.envs:LocalWarehouse',
)
register(
    id='global-mini-warehouse-v0',
    entry_point='warehouse.envs:MiniWarehouse',
)
register(
    id='local-mini-warehouse-v0',
    entry_point='warehouse.envs:LocalMiniWarehouse',
)