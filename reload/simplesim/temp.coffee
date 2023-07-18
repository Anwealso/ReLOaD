tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_15_device_/job:localhost/replica:0/task:0/device:CPU:0}}
  Received incompatible tensor at flattened index 0 from table 'uniform_table'.  
  Specification has (dtype, shape): (int32, [?]).  
  Tensor has (dtype, shape): (int32, [2,1]).

Table signature:
  0: Tensor<name: 'step_type/step_type', dtype: int32, shape: [?]>
  1: Tensor<name: 'observation/0/count', dtype: float, shape: [?,1,1]>
  2: Tensor<name: 'observation/1/budget', dtype: float, shape: [?,1,1]>
  3: Tensor<name: 'observation/2/avg_conf', dtype: float, shape: [?,8,1]>
  4: Tensor<name: 'observation/3/curr_conf', dtype: float, shape: [?,8,1]>
  5: Tensor<name: 'action/action', dtype: int32, shape: [?]>
  6: Tensor<name: 'policy_info/dist_params/logits/CategoricalProjectionNetwork_logits', dtype: float, shape: [?,4]>
  7: Tensor<name: 'next_step_type/step_type', dtype: int32, shape: [?]>
  8: Tensor<name: 'reward/reward', dtype: float, shape: [?,1,1]>
  9: Tensor<name: 'discount/discount', dtype: float, shape: [?]> [Op:IteratorGetNext]

