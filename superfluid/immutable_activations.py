def matrix_element(to_neuron, to_input, from_neuron, from_output, value = 1.0):
    return {to_neuron: {to_input: {from_neuron: {from_output: value}}}}

# activations

# accumulator with two arguments (used everywhere)

def accum_add_args(all_inputs):
    return {'result': add_v_values(getv(all_inputs, 'accum'), getv(all_inputs, 'delta'))}
    
# constant update matrices for the experiment of a self-referential machine
# running waves pattern through its own wave-connectivity matrix_element
# as described in Appendix B.2 of https://arxiv.org/abs/1706.00648 

def update_1(all_inputs):
    return {'result': add_v_values(matrix_element('self', 'delta', 'update-1', 'result', -1.0),
                                   matrix_element('self', 'delta', 'update-2', 'result'))}

def update_2(all_inputs):
    return {'result': add_v_values(matrix_element('self', 'delta', 'update-2', 'result', -1.0),
                                   matrix_element('self', 'delta', 'update-3', 'result'))}

def update_3(all_inputs):
    return {'result': add_v_values(matrix_element('self', 'delta', 'update-3', 'result', -1.0),
                                   matrix_element('self', 'delta', 'update-1', 'result'))}

# activations needed for Section 3 of https://arxiv.org/abs/1606.09470

def max_norm_dict(all_inputs):
    return {'norm': {':number': max_norm(getv(all_inputs, 'dict'))}}

def dot_product(all_inputs):
    #print(getv(all_inputs, 'x'))
    #print(getv(all_inputs, 'y'))
    result = mult_mask_lin_comb(getv(all_inputs, 'x'), getv(all_inputs, 'y'))
    return  {'dot': {':number': 0.0}, 'warning': {':number': 1.0}} if isinstance(result, dict) else {'dot': {':number': result}, 'warning': {':number': 0.0}}

def compare_scalars(all_inputs):
    #print("all_inputs ", all_inputs)
    x_dict, y_dict = getv(all_inputs, 'x'), getv(all_inputs, 'y')
    #print("v_value x_dict ", x_dict)
    x = getn(x_dict)
    y = getn(y_dict)
    return {'true': {':number': relu(x-y)}, 'false': {':number': relu(y-x)}}

def const_1(all_inputs):
    return {'const_1': {':number': 1.0}}

def const_end(all_inputs):
    return {'char': {'.': 1.0}}

def timer_add_one(all_inputs):
    return {'timer': {':number': getn(getv(all_inputs, 'timer')) + 1.0}}

import jax.lax

def input_dummy_helper(all_inputs):
    t = relu(getn(getv(all_inputs, 'timer')))
    s = "test string."
    i = min(round(t // 10), len(s) - 1)
    return {'char': {s[i:i+1]: 1.0} if t % 10 == 0 else {}}

def input_dummy(all_inputs):
    return jax.lax.stop_gradient(input_dummy_helper(all_inputs))

def output_dummy(all_inputs):
    return {'dict-1': getv(all_inputs, 'dict-1'), 'dict-2': getv(all_inputs, 'dict-2')}

activation_functions = {'accum_add_args': accum_add_args,
                        'max_norm_dict': max_norm_dict,
                        'dot_product': dot_product,
                        'compare_scalars': compare_scalars,
                        'const_1': const_1,
                        'const_end': const_end,
                        'timer_add_one': timer_add_one,
                        'input_dummy': input_dummy,
                        'output_dummy': output_dummy,
                        'update_1': update_1,
                        'update_2': update_2,
                        'update_3': update_3}

