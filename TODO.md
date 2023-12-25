爪子位置不对，学之后偏
为什么goal那么远
terrain里 step直接改成了flat
config 199 num_goals

Traceback (most recent call last):
  File "train.py", line 71, in <module>
    train(args)
  File "train.py", line 66, in train
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
  File "/home/qiwu/doggybot/extreme-parkour/rsl_rl/rsl_rl/runners/on_policy_runner.py", line 198, in learn_RL
    mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
  File "/home/qiwu/doggybot/extreme-parkour/rsl_rl/rsl_rl/algorithms/ppo.py", line 198, in update
    self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension
  File "/home/qiwu/doggybot/extreme-parkour/rsl_rl/rsl_rl/modules/actor_critic.py", line 286, in act
    self.update_distribution(observations, hist_encoding)
  File "/home/qiwu/doggybot/extreme-parkour/rsl_rl/rsl_rl/modules/actor_critic.py", line 283, in update_distribution
    self.distribution = Normal(mean, mean*0. + self.std)
  File "/home/qiwu/anaconda3/envs/parkour/lib/python3.8/site-packages/torch/distributions/normal.py", line 50, in __init__
    super(Normal, self).__init__(batch_shape, validate_args=validate_args)
  File "/home/qiwu/anaconda3/envs/parkour/lib/python3.8/site-packages/torch/distributions/distribution.py", line 55, in __init__
    raise ValueError(
ValueError: Expected parameter loc (Tensor of shape (36864, 14)) of distribution Normal(loc: torch.Size([36864, 14]), scale: torch.Size([36864, 14])) to satisfy the constraint Real(), but found invalid values:
tensor([[nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        ...,
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan]], device='cuda:0',
       grad_fn=<AddmmBackward0>)

## 12.19
env_ids in _post_physics_step_callback
correct offset
use gym.find_actor_rigid_body_index and change root_state definition
self.lookat_id
env_ids 和 robot_idx搞混了
632 error

## 12.22
-1.update observations
-2.p_gains wrong dimensions
3.visualization
4.extras in step? what info?

### 000-06有效果

## 12.25
1.cmd positive - action negative - close ; cmd negative - action positive - open
2.reward scale for display