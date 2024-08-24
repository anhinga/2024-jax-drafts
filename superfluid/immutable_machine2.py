
# handcrafted duplicate detector

# we are likely to want to distinguish between hard links and soft links

# we'll need to do this before optimization experiments, but now
# we are only trying to compute gradients, so perhaps we don't need to do that yet

init_matrix_elements = [matrix_element("self", "accum", "self", "result"),
                        matrix_element("timer", "timer", "timer", "timer"),
                        matrix_element("input", "timer", "timer", "timer"),
                        matrix_element("accum", "accum", "accum", "result"),
                        matrix_element("accum", "delta", "input", "char"),
                        matrix_element("norm", "dict", "accum", "result"),
                        matrix_element("compare", "x", "norm", "norm"),
                        matrix_element("compare", "y", "const_1", "const_1"),
                        matrix_element("dot", "x", "eos", "char"),
                        matrix_element("dot", "y", "accum", "result"),
                        matrix_element("output", "dict-1", "compare", "true"),
                        matrix_element("output", "dict-2", "dot", "dot"),
                        matrix_element("self", ":function", "self", ":function"),
                        matrix_element("timer", ":function", "timer", ":function"),
                        matrix_element("input", ":function", "input", ":function"),
                        matrix_element("accum", ":function", "accum", ":function"),
                        matrix_element("norm", ":function", "norm", ":function"),
                        matrix_element("const_1", ":function", "const_1", ":function"), 
                        matrix_element("eos", ":function", "eos", ":function"),                         
                        matrix_element("compare", ":function", "compare", ":function"), 
                        matrix_element("dot", ":function", "dot", ":function"), 
                        matrix_element("output", ":function", "output", ":function")]

init_matrix = {'result': add_v_values(*init_matrix_elements)}

# let's initialize carefully (this might be excessive)

initial_output = {'self': add_v_values(init_matrix, {':function': {'accum_add_args': 1.0}}),
                  'timer': add_v_values({'timer': {':number': 0.0}}, {':function': {'timer_add_one': 1.0}}),
                  'input': {':function': {'input_dummy': 1.0}},
                  'accum': add_v_values({'result': {}}, {':function': {'accum_add_args': 1.0}}),
                  'norm': add_v_values({'norm': {':number': 1.0}}, {':function': {'max_norm_dict': 1.0}}),
                  'const_1': add_v_values({'const_1': {':number': 1.0}}, {':function': {'const_1': 1.0}}),
                  'eos': add_v_values({'char': {'.': 1.0}}, {':function': {'const_end': 1.0}}),
                  'compare': {':function': {'compare_scalars': 1.0}},
                  'dot': {':function': {'dot_product': 1.0}},
                  'output': {':function': {'output_dummy': 1.0}}}

from pprint import pprint

from functools import reduce

def square(x):
    return x * x

def one_cycle(state):
    return two_stroke_cycle(state['output'])

def one_iteration(accum, y):
    previous_dict, previous_state = accum
    new_state = one_cycle(previous_state)
    new_dict = {**previous_dict, **{y: new_state}}
    return (new_dict, new_state)

result, last_state = reduce(one_iteration, range(150), ({}, {'input': {}, 'output': initial_output}))

# >>> [key for key in result]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]

# >>> [(key, result[key]['output']['timer']['timer'][':number']) for key in result]
# [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0), (5, 6.0), (6, 7.0), (7, 8.0), (8, 9.0), (9, 10.0), (10, 11.0), (11, 12.0), (12, 13.0), (13, 14.0), (14, 15.0), (15, 16.0), (16, 17.0), (17, 18.0), (18, 19.0), (19, 20.0), (20, 21.0), (21, 22.0), (22, 23.0), (23, 24.0), (24, 25.0), (25, 26.0), (26, 27.0), (27, 28.0), (28, 29.0), (29, 30.0), (30, 31.0), (31, 32.0), (32, 33.0), (33, 34.0), (34, 35.0), (35, 36.0), (36, 37.0), (37, 38.0), (38, 39.0), (39, 40.0), (40, 41.0), (41, 42.0), (42, 43.0), (43, 44.0), (44, 45.0), (45, 46.0), (46, 47.0), (47, 48.0), (48, 49.0), (49, 50.0), (50, 51.0), (51, 52.0), (52, 53.0), (53, 54.0), (54, 55.0), (55, 56.0), (56, 57.0), (57, 58.0), (58, 59.0), (59, 60.0), (60, 61.0), (61, 62.0), (62, 63.0), (63, 64.0), (64, 65.0), (65, 66.0), (66, 67.0), (67, 68.0), (68, 69.0), (69, 70.0), (70, 71.0), (71, 72.0), (72, 73.0), (73, 74.0), (74, 75.0), (75, 76.0), (76, 77.0), (77, 78.0), (78, 79.0), (79, 80.0), (80, 81.0), (81, 82.0), (82, 83.0), (83, 84.0), (84, 85.0), (85, 86.0), (86, 87.0), (87, 88.0), (88, 89.0), (89, 90.0), (90, 91.0), (91, 92.0), (92, 93.0), (93, 94.0), (94, 95.0), (95, 96.0), (96, 97.0), (97, 98.0), (98, 99.0), (99, 100.0), (100, 101.0), (101, 102.0), (102, 103.0), (103, 104.0), (104, 105.0), (105, 106.0), (106, 107.0), (107, 108.0), (108, 109.0), (109, 110.0), (110, 111.0), (111, 112.0), (112, 113.0), (113, 114.0), (114, 115.0), (115, 116.0), (116, 117.0), (117, 118.0), (118, 119.0), (119, 120.0), (120, 121.0), (121, 122.0), (122, 123.0), (123, 124.0), (124, 125.0), (125, 126.0), (126, 127.0), (127, 128.0), (128, 129.0), (129, 130.0), (130, 131.0), (131, 132.0), (132, 133.0), (133, 134.0), (134, 135.0), (135, 136.0), (136, 137.0), (137, 138.0), (138, 139.0), (139, 140.0), (140, 141.0), (141, 142.0), (142, 143.0), (143, 144.0), (144, 145.0), (145, 146.0), (146, 147.0), (147, 148.0), (148, 149.0), (149, 150.0)]

# result: see run-150.txt

def loss(state):
    trace, new_state = reduce(one_iteration, range(5), ({}, state))
    first = [trace[key]['input']['output']['dict-1'][':number'] for key in trace]
    second = [trace[key]['input']['output']['dict-1'][':number'] for key in trace]
    return sum([square(x-10.0) for x in first]) + sum([square(x-10.0) for x in second]) 

# >>> loss(last_state)
# 464.0

# this does not quite match Julia version (we do seem to have various "off-by-ones"; do we want to fix them?)

grad_loss = grad(loss)    

this_grad = grad_loss(last_state)

pprint(trim_v_value(this_grad, 0.0))

# {'output': {'accum': {':function': {'accum_add_args': Array(-328., dtype=float32, weak_type=True)},
#                       'result': {'.': Array(-80., dtype=float32, weak_type=True)}},
#             'compare': {':function': {'compare_scalars': Array(-348., dtype=float32, weak_type=True)},
#                         'true': {':number': Array(-28., dtype=float32, weak_type=True)}},
#             'const_1': {':function': {'const_1': Array(80., dtype=float32, weak_type=True)},
#                         'const_1': {':number': Array(28., dtype=float32, weak_type=True)}},
#             'input': {':function': {'input_dummy': Array(-24., dtype=float32, weak_type=True)}},
#             'norm': {':function': {'max_norm_dict': Array(-344., dtype=float32, weak_type=True)},
#                      'norm': {':number': Array(-28., dtype=float32, weak_type=True)}},
#             'self': {':function': {'accum_add_args': Array(-11692., dtype=float32, weak_type=True)},
#                      'result': {'accum': {':function': {'accum': {':function': Array(-448., dtype=float32, weak_type=True)}},
#                                           'accum': {'accum': {'result': Array(-304., dtype=float32, weak_type=True)}},
#                                           'delta': {'input': {'char': Array(-24., dtype=float32, weak_type=True)}}},
#                                 'compare': {':function': {'compare': {':function': Array(-888., dtype=float32, weak_type=True)}},
#                                             'x': {'norm': {'norm': Array(-456., dtype=float32, weak_type=True)}},
#                                             'y': {'const_1': {'const_1': Array(108., dtype=float32, weak_type=True)}}},
#                                 'const_1': {':function': {'const_1': {':function': Array(156., dtype=float32, weak_type=True)}}},
#                                 'input': {':function': {'input': {':function': Array(-24., dtype=float32, weak_type=True)}}},
#                                 'norm': {':function': {'norm': {':function': Array(-696., dtype=float32, weak_type=True)}},
#                                          'dict': {'accum': {'result': Array(-344., dtype=float32, weak_type=True)}}},
#                                 'output': {'dict-1': {'compare': {'true': Array(-432., dtype=float32, weak_type=True)}}},
#                                 'self': {':function': {'self': {':function': Array(-15460., dtype=float32, weak_type=True)}},
#                                          'accum': {'self': {'result': Array(-11692., dtype=float32, weak_type=True)}}}}}}}
