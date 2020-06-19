# ego_map
Code base for paper "EgoMap: Projective mapping and structured egocentric memory for Deep RL"


# Installation
Instructions to run the code for Baseline, Neural Map and EgoMap for the 4 scenarios:

The code has been testing on linux (ubuntu 16.04 LTS) with python3
Install dependencies detailed in requirements.txt
The scenarios themselves are from the following repository , but have been included for convenience.

## Scenarios
Possible scenarios/directorys are:
    labyrinth:
        labyrinth_maze{:003}.cfg
        resources/scenarios/custom_scenarios/labyrinth13
    find return:
        mino_maze{:003}.cfg
        resources/scenarios/custom_scenarios/mino11
    4-item:
        four_item_maze{:003}.cfg
        resources/scenarios/custom_scenarios/four_item5
    6-item:
        six_item_maze{:003}.cfg
        resources/scenarios/custom_scenarios/six_item5


set $SCENARIO_DIR as your chosen scenario directory and $SCENARIO as the scenario.
e.g. 
SCENARIO_DIR=resources/scenarios/custom_scenarios/labyrinth13
SCENARIO=labyrinth_maze{:003}.cfg


## Training

set your username
e.g. NAME=ANONYMOUS

### Baseline recurrent agent

```
python train_a2c.py --num_steps 128 --log_interval 10 --eval_freq 1000 --model_save_rate 1000 --eval_games 50 --num_frames 300000000 --gamma 0.99 --recurrent_policy --num_stack 1 --norm_obs --scenario_dir  $SCENARIO_DIR/train/ --scenario $SCENARIO --limit_actions --conv1_size 16 --conv2_size 32 --conv3_size 16 --hidden_size 128 --entropy_coef 0.001 --test_scenario_dir  $SCENARIO_DIR/test/ --multimaze --num_mazes_train 256 --num_mazes_test 64 --fixed_scenario --use_pipes --depth_as_obs --user_dir $NAME
```

### Neural Map agent
```
python train_a2c.py --num_steps 128 --log_interval 10 --eval_freq 1000 --model_save_rate 1000 --eval_games 50 --num_frames 300000000 --gamma 0.99 --recurrent_policy --num_stack 1 --norm_obs --scenario_dir  $SCENARIO_DIR/train/ --scenario $SCENARIO --limit_actions --conv1_size 16 --conv2_size 32 --conv3_size 16 --hidden_size 128 --entropy_coef 0.001 --test_scenario_dir  $SCENARIO_DIR/test/ --multimaze --num_mazes_train 256 --num_mazes_test 64 --fixed_scenario --use_pipes --ego_bin_dim 64 --ego_half_size 32 --ego_skip --reduce_blur --shift_thresh 120 --ego_hidden_size 32 --skip_world_shift --merge_later --ego_query --ego_num_chans 16 --ego_use_tanh --skip_cnn_relu --query_position --ego_query_cosine  --ego_query_scalar --neural_map 0 --nm_gru_op --nm_skip --depth_as_obs --user_dir $NAME
```

### EgoMap agent
```
python train_a2c.py --num_steps 128 --log_interval 10 --eval_freq 1000 --model_save_rate 1000 --eval_games 50 --num_frames 300000000 --gamma 0.99 --recurrent_policy --num_stack 1 --norm_obs --scenario_dir  $SCENARIO_DIR/train/ --scenario $SCENARIO --limit_actions --conv1_size 16 --conv2_size 32 --conv3_size 16 --hidden_size 128 --entropy_coef 0.001 --test_scenario_dir  $SCENARIO_DIR/test/ --multimaze --num_mazes_train 256 --num_mazes_test 64 --fixed_scenario --use_pipes --ego_model 0 --ego_bin_dim 64 --ego_half_size 12 --ego_skip --reduce_blur --shift_thresh 120 --ego_hidden_size 32 --skip_world_shift --merge_later --ego_query --ego_num_chans 16 --ego_use_tanh --skip_cnn_relu --query_position --ego_query_cosine  --ego_query_scalar --depth_as_obs --user_dir $NAME
```