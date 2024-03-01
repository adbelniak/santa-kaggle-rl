from easydict import EasyDict

collector_env_num = 10
n_episode = 10
evaluator_env_num = 3
num_simulations = 10
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.
action_space_size=6
eps_greedy_exploration_in_collect = False
td_steps = 5

# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
threshold_training_steps_for_final_temperature = int(5e5)
eps_greedy_exploration_in_collect = False
santa_alpha_zero_config = dict(
    exp_name='data_az_ptree/santa_env-mode_eval0',
    env=dict(
        observation_shape=(1,24, 26),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        
    ),
    policy=dict(
        model=dict(
            observation_shape=(1,24, 26),
            action_space_size=6,
            discrete_action_encoding_type='one_hot',
            fc_value_layers=[2],
            fc_policy_layers=[2],
        ),
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            decay=int(2e5),
        ),
        mcts_ctree=True,
        td_steps=td_steps,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
        cuda=True,
        env_type='not_board_games',
        simulation_env_name='santa_env',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        entropy_weight=0.0,
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

santa_alpha_zero_config = EasyDict(santa_alpha_zero_config)
