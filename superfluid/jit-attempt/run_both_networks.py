import time
import optax

leaves, treedef = tree_util.tree_flatten(initial_output)
print("size (leaves):", len(leaves), len([k for k in leaves if k == SENTINEL]), 
                        len([k for k in leaves if k == 1.0]), len([k for k in leaves if k == 0.0]))

start_time = time.time()

changing_output = create_tree(rng_key, initial_output)

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

"""
start_time = time.time()

result, last_state = reduce(one_iteration, range(150), ({}, {'input': {}, 'output': initial_output_init}))

print(time.time()-start_time, " seconds")
"""

def loss_fn(changing_output):
    trace, _ = reduce(one_iteration, range(2), ({}, {'input': {}, 'output': changing_output}))
    trace_manual, _ = reduce(one_iteration, range(2), ({}, {'input': {}, 'output': initial_output_manual}))
    first = [trace[key]['input']['output']['dict-1'][':number'] for key in trace if key != 0]
    second = [trace[key]['input']['output']['dict-2'][':number'] for key in trace if key != 0]
    first_manual = [trace_manual[key]['input']['output']['dict-1'][':number'] for key in trace_manual if key != 0]
    second_manual = [trace_manual[key]['input']['output']['dict-2'][':number'] for key in trace_manual if key != 0]
    unregularized_loss = sum(square(x - y) for x, y in zip(first, first_manual)) + \
                         sum(square(x - y) for x, y in zip(second, second_manual))    
    # TODO: ADD REGULARIZATION (STANDARD AND NOVEL)
    loss = unregularized_loss
    return loss

# Mask to ensure updates only apply to optimizable weights
mask_tree = tree_util.tree_map(lambda x: x == SENTINEL, initial_output)

# Adam optimizer
optimizer = optax.adam(learning_rate=0.1) # might dial it back to 0.001, but we'll see; let's start ambitious
opt_state = optimizer.init(changing_output)

# Define the optimization step
@jax.jit
def step(changing_output, opt_state):
    start_time = time.time()
    print("about to compute gradient")
    loss, grads = jax.value_and_grad(loss_fn)(changing_output)
    print(time.time()-start_time, " seconds to compute gradient")
    
    # Apply mask to gradients
    start_time = time.time()
    grads = tree_util.tree_map(lambda g, m: g if m else 0, grads, mask_tree)
    print(time.time()-start_time, " seconds to apply mask to gradient")
    
    start_time = time.time() 
    updates, opt_state = optimizer.update(grads, opt_state)
    print(time.time()-start_time, " seconds to compute optimizer update") 
 
    start_time = time.time()
    new_tree = optax.apply_updates(changing_output, updates)
    print(time.time()-start_time, " seconds to apply optimizer update")
    
    return new_tree, opt_state, loss
    
start_time = time.time()
print("initial loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
   
import pickle

"""   
# Run optimization for 10 steps
def n_steps(n):
    for n_step in range(n):
        changing_output, opt_state, loss = step(changing_output, opt_state)
        start_time = time.time()
        with open('changing_output.pkl', 'wb') as f:
            pickle.dump(changing_output, f)
        with open('opt_state.pkl', 'wb') as f:
            pickle.dump(opt_state, f)
        print(time.time()-start_time, " seconds to pickle the checkpoint")
        print(f'step: {n_step} loss: {loss}')
"""

# Run optimization for 3 steps
@jax.jit
def n_steps(changing_output, opt_state):
    def loop_body(carry, n_step):
        carry_val, changing_output, opt_state = carry  # Unpack the carry
        changing_output, opt_state, loss = step(changing_output, opt_state)
        #start_time = time.time()
        #with open('changing_output.pkl', 'wb') as f:
        #    pickle.dump(changing_output, f)
        #with open('opt_state.pkl', 'wb') as f:
        #    pickle.dump(opt_state, f)
        #print(time.time()-start_time, " seconds to pickle the checkpoint")
        jax.debug.print(f'step: {n_step} loss: {loss}')
        # carry is the loop state, n_step is the iteration number
        new_carry_val = carry_val + n_step  # Example update
        return (new_carry_val, changing_output, opt_state), new_carry_val
  # Return the updated carry and value to store

    initial_carry_val = 0  # Initial state
    steps = jnp.arange(3)  # Sequence of steps

    # Initial carry: combine everything into a tuple
    initial_carry = (initial_carry_val, changing_output, opt_state)
    
    # Use jax.lax.scan for the loop
    final_carry, _  = jax.lax.scan(loop_body, initial_carry, steps)

    # Unpack the final carry to get the final values
    final_carry_val, final_changing_output, final_opt_state = final_carry

    return final_changing_output, final_opt_state

changing_output, opt_state = n_steps(changing_output, opt_state)

"""GPT-4o recommendation
import jax
import jax.numpy as jnp
import time
import pickle

# Assuming step(changing_output, opt_state) is a function you've defined elsewhere

@jax.jit
def n_steps(changing_output, opt_state):
    # Define the loop body. Everything needed must be in the `carry`.
    def loop_body(carry, n_step):
        carry_val, changing_output, opt_state = carry  # Unpack the carry

        # Perform one step of the optimization
        changing_output, opt_state, loss = step(changing_output, opt_state)

        # Timing and checkpointing (these side effects won't be JIT-compiled)
        start_time = time.time()
        with open('changing_output.pkl', 'wb') as f:
            pickle.dump(changing_output, f)
        with open('opt_state.pkl', 'wb') as f:
            pickle.dump(opt_state, f)
        print(time.time() - start_time, "seconds to pickle the checkpoint")
        print(f'step: {n_step} loss: {loss}')

        # Update the carry
        new_carry_val = carry_val + n_step  # Example of carry update
        return (new_carry_val, changing_output, opt_state), new_carry_val

    initial_carry_val = 0  # Initial state for the carry value
    steps = jnp.arange(3)  # Sequence of steps

    # Initial carry: combine everything into a tuple
    initial_carry = (initial_carry_val, changing_output, opt_state)

    # Use jax.lax.scan, passing the carry
    final_carry, _ = jax.lax.scan(loop_body, initial_carry, steps)

    # Unpack the final carry to get the final values
    final_carry_val, final_changing_output, final_opt_state = final_carry

    return final_changing_output, final_opt_state

# Example usage
changing_output = ...  # Define this according to your context
opt_state = ...  # Define this according to your context

# Call n_steps
final_changing_output, final_opt_state = n_steps(changing_output, opt_state)
"""

"""
start_time = time.time()

r = format_floats(convert_jax_to_plain(result), precision=3)

pprint([(key, r[key]['output']['timer']['timer'][':number'], 
              r[key]['output']['input']['char'], 
              r[key]['input']['output']['dict-1'][':number'],
              r[key]['input']['output']['dict-1']['.'],
              r[key]['input']['output']['dict-2'][':number'],              
              r[key]['input']['output']['dict-2']['.'])  for key in result])


print(time.time()-start_time, " seconds")

leaves, treedef = tree_util.tree_flatten(result)

print(len(leaves), " leaves in the result log")
"""
