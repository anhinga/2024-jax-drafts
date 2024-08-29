import time

start_time = time.time()

initial_output_init = create_tree(rng_key, initial_output)

from pprint import pprint

from functools import reduce

def one_cycle(state):
    return two_stroke_cycle(state['output'])

def one_iteration(accum, y):
    previous_dict, previous_state = accum
    new_state = one_cycle(previous_state)
    new_dict = {**previous_dict, **{y: new_state}}
    return (new_dict, new_state)

def format_floats(obj, precision=2):
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {k: format_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_floats(x, precision) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(format_floats(x, precision) for x in obj)
    else:
        return obj
        
def convert_jax_to_plain(obj):
    if isinstance(obj, jnp.ndarray) and obj.size == 1:  # Check if it's a JAX array with a single element
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_jax_to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_jax_to_plain(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_jax_to_plain(x) for x in obj)
    else:
        return obj
        
print(time.time()-start_time, " seconds")

start_time = time.time()

result, last_state = reduce(one_iteration, range(150), ({}, {'input': {}, 'output': initial_output_init}))

print(time.time()-start_time, " seconds")

start_time = time.time()

r = format_floats(convert_jax_to_plain(result), precision=3)

pprint([(key, r[key]['output']['timer']['timer'][':number'], 
              r[key]['output']['input']['char'], 
              r[key]['input']['output']['dict-1'][':number'],
              r[key]['input']['output']['dict-1']['.'],
              r[key]['input']['output']['dict-2'][':number'],              
              r[key]['input']['output']['dict-2']['.'])  for key in result])


print(time.time()-start_time, " seconds")