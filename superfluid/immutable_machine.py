
# self-referential machine

init_matrix_elements = [matrix_element("self", "accum", "self", "result"),
                        matrix_element("self", "delta", "update-1", "result"),
                        matrix_element("self", ":function", "self", ":function"), 
                        matrix_element("update-1", ":function", "update-1", ":function"), 
                        matrix_element("update-2", ":function", "update-2", ":function"), 
                        matrix_element("update-3", ":function", "update-3", ":function")]

init_matrix = {'result': add_v_values(*init_matrix_elements)}

initial_output = {'self': add_v_values(init_matrix, {':function': {'accum_add_args': 1.0}}),
                  'update-1': add_v_values(update_1({}), {':function': {'update_1': 1.0}}),
                  'update-2': add_v_values(update_2({}), {':function': {'update_2': 1.0}}),
                  'update-3': add_v_values(update_3({}), {':function': {'update_3': 1.0}})}

from pprint import pprint

pprint(init_matrix)
# {'result': {'self': {':function': {'self': {':function': 1.0}},
#                      'accum': {'self': {'result': 1.0}},
#                      'delta': {'update-1': {'result': 1.0}}},
#             'update-1': {':function': {'update-1': {':function': 1.0}}},
#             'update-2': {':function': {'update-2': {':function': 1.0}}},
#             'update-3': {':function': {'update-3': {':function': 1.0}}}}}

pprint(initial_output)
# {'self': {':function': {'accum_add_args': 1.0},
#           'result': {'self': {':function': {'self': {':function': 1.0}},
#                               'accum': {'self': {'result': 1.0}},
#                               'delta': {'update-1': {'result': 1.0}}},
#                      'update-1': {':function': {'update-1': {':function': 1.0}}},
#                      'update-2': {':function': {'update-2': {':function': 1.0}}},
#                      'update-3': {':function': {'update-3': {':function': 1.0}}}}},
#  'update-1': {':function': {'update_1': 1.0},
#               'result': {'self': {'delta': {'update-1': {'result': -1.0},
#                                             'update-2': {'result': 1.0}}}}},
#  'update-2': {':function': {'update_2': 1.0},
#               'result': {'self': {'delta': {'update-2': {'result': -1.0},
#                                             'update-3': {'result': 1.0}}}}},
#  'update-3': {':function': {'update_3': 1.0},
#               'result': {'self': {'delta': {'update-1': {'result': 1.0},
#                                             'update-3': {'result': -1.0}}}}}}

step1 = two_stroke_cycle(initial_output)
pprint(step1)
step2 = two_stroke_cycle(step1['output'])
pprint(step2)
step3 = two_stroke_cycle(step2['output'])
pprint(step3)
step4 = two_stroke_cycle(step3['output'])
pprint(step4)
