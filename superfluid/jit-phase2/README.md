next phase of jax.jit exploration for this problem

let's reduce the learning rate to 0.01 and start scaling the problem up

start with length 11, then proceed to 140

one question is: how would jit compile time scale (it's really long now already,
and if it scales unfavorably, it might be a showstopper, or, at least,
might force a very careful design, because making changes and restarting
from a checkpoint might still be very expensive, if jit compilation is superlong).
