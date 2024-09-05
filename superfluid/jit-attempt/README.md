these changes in input-dummy-helper allow us to uncomment #@jax.jit on line 83 of run_both_networks.py,
but after running 1 step the program starts behaving strangely and ultimately breaks down.

the run @jax.jit commented out is still fine, although noticeably slower 
because way more {key: 0.0} are passed around instead of being omitted.
