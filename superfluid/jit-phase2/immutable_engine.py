
# down-movement

def apply_v_valued_matrix(v_valued_matrix, v_valued_args, level):
    if level==0:
        return mult_mask_lin_comb(v_valued_matrix, v_valued_args)
    return {key: apply_v_valued_matrix(v_valued_matrix[key], v_valued_args, level-1) for key in v_valued_matrix.keys()}

# superfluid version of up-movement

def up_movement_helper(input_tree):
    dict_of_functions = input_tree[':function']
    partial_results = [mult_v_value(dict_of_functions[k], activation_functions[k](input_tree)) for k in dict_of_functions.keys()]
    result = add_v_values(*partial_results)
    return {**{':function': dict_of_functions}, **result}


def up_movement(all_input_trees):
    return {neuron_name: up_movement_helper(all_input_trees[neuron_name]) for neuron_name in all_input_trees.keys()}

def two_stroke_cycle(current_output):
    new_input = apply_v_valued_matrix(current_output['self']['result'], current_output, 2)
    new_output = up_movement(new_input)
    return {'input': new_input, 'output': new_output}
