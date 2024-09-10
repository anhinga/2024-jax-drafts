
from jax.tree_util import tree_map

def mult_v_value(multiplier, v_value):
    return tree_map(lambda x: multiplier*x, v_value)

def mult_mask_v_value(mult_mask, v_value):
    joint_keys = frozenset(mult_mask.keys()) & frozenset(v_value.keys())
    number_keys = {key for key in joint_keys if not isinstance(mult_mask[key], dict) and not isinstance(v_value[key], dict)}
    dict_keys = {key for key in joint_keys if isinstance(mult_mask[key], dict) and isinstance(v_value[key], dict)}
    mult_keys = {key for key in (joint_keys - number_keys - dict_keys) if isinstance(v_value[key], dict)}
    # the remaining case does not generate values and is omitted
    number_key_values = {key: (mult_mask[key]*v_value[key]) for key in number_keys}
    dict_key_values = {key: mult_mask_v_value(mult_mask[key], v_value[key]) for key in dict_keys}
    mult_key_values = {key: mult_v_value(mult_mask[key], v_value[key]) for key in mult_keys} 
    return {**number_key_values, **dict_key_values, **mult_key_values}

def add_v_values(*v_values):
    trees = [v_value for v_value in v_values if isinstance(v_value, dict)]
    numbers = [v_value for v_value in v_values if not isinstance(v_value, dict)]   
    if not trees:
        return sum(numbers)
    all_trees = [*trees, {':number': sum(numbers)}] if numbers else trees
    all_keys = frozenset(key for tree in all_trees for key in tree.keys())  
    merged = {
        key: add_v_values(*[tree[key] for tree in all_trees if key in tree])
        for key in all_keys
    } 
    return merged

def mult_mask_lin_comb(mult_mask, v_value):
    joint_keys = frozenset(mult_mask.keys()) & frozenset(v_value.keys())
    number_keys = {key for key in joint_keys if not isinstance(mult_mask[key], dict) and not isinstance(v_value[key], dict)}
    dict_keys = {key for key in joint_keys if isinstance(mult_mask[key], dict) and isinstance(v_value[key], dict)}
    mult_keys = {key for key in (joint_keys - number_keys - dict_keys) if isinstance(v_value[key], dict)}
    # the remaining case does not generate values and is omitted
    number_values = [mult_mask[key]*v_value[key] for key in number_keys]
    dict_values = [mult_mask_lin_comb(mult_mask[key], v_value[key]) for key in dict_keys]
    mult_values = [mult_v_value(mult_mask[key], v_value[key]) for key in mult_keys] 
    return add_v_values(*number_values, *dict_values, *mult_values)

dict_a = {'a': 2.0, 'b': 3.0, 'c': 2.5} 

dict_f = {'a': 2.0, 'b': 3.0, 'c': {':number': 6.0, 'u': 1.0}}

mult_mask_lin_comb(dict_a, dict_a)
# 19.25
mult_mask_lin_comb(dict_f, dict_f)
# 50.0
mult_mask_lin_comb(dict_a, dict_f)
# {':number': 28.0, 'u': 2.5}
mult_mask_lin_comb(dict_f, dict_a)
# 13.0

# gradients also work
# for example

from jax import grad
def self_apply(x):
    return mult_mask_lin_comb(x, x)

self_apply(dict_f)
# 50.0

grad_self_apply = grad(self_apply)
grad_self_apply(dict_f)
# {'a': Array(4., dtype=float32, weak_type=True), 'b': Array(6., dtype=float32, weak_type=True), 'c': {':number': Array(12., dtype=float32, weak_type=True), 'u': Array(2., dtype=float32, weak_type=True)}}
