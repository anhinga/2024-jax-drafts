import jax
import jax.numpy as jnp
import jax.nn as jnn

def max_norm(v_value): # we'll also have a slightly different activation function of this kind: v_value -> {'result': max_norm(v_value)} 
    if not isinstance(v_value, dict):
        return jnp.abs(v_value)
    elif len(v_value) == 0:
        return jnp.array(0.0)  # Return 0.0 if the dictionary is empty
    else:
        # Using jax.numpy.maximum to replace the max function
        return jnp.maximum(0.0, jnp.max(jnp.array([max_norm(v_value[key]) for key in v_value.keys()])))
    # the following has been re-written in a JAX-friendly way 
    # return abs(v_value) if not isinstance(v_value, dict) else max([0.0, *[max_norm(v_value[key]) for key in v_value.keys()]])

def trim_v_value(v_value, threshold): # keeping paths with max_norm strictly above treshold; keeping scalars
    return v_value if not isinstance(v_value, dict) else {key: trim_v_value(v_value[key], threshold) for key in v_value.keys() if max_norm(v_value[key]) > threshold}

def relu(x):
    return jnn.relu(x)
    # return max(0.0, x) re-written in a JAX-friendly way

def getv(v_value, key):
    return v_value.get(key, {})

def getn(v_value):
    return v_value.get(':number', 0.0)

def count(v_value):
    return 1.0 if not isinstance(v_value, dict) else sum([0.0, *[count(v_value[key]) for key in v_value.keys()]])

def map_item(jax_v_value):
    return tree_map(lambda x: x.item(), jax_v_value) 
