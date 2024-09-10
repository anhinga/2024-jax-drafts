next phase of jax.jit exploration for this problem

let's reduce the learning rate to 0.01 and start scaling the problem up

start with length 11, then proceed to 140

one question is: how would jit compile time scale (it's really long now already,
and if it scales unfavorably, it might be a showstopper, or, at least,
might force a very careful design, because making changes and restarting
from a checkpoint might still be very expensive, if jit compilation is superlong).

**note**: so far compile time looks horrible, takes tons of RAM (way over 10GB)
and runs forever (that's for length 11). **so I am not optimistic
that this will be usable right now.**

---

it ended up breaking like this which does look like an out-of-memory error:

```
size (leaves): 2944 2906 37 1
0.7761118412017822  seconds
initial loss  0.35485542  computed in  5.285385847091675  seconds
about to compute gradient
208.57648587226868  seconds to compute gradient
0.0029952526092529297  seconds to apply mask to gradient
8.653279304504395  seconds to compute optimizer update
0.9627285003662109  seconds to apply optimizer update
jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\anhin\Desktop\GitHub\2024-jax-drafts\superfluid\jit-phase2\main.py", line 655, in <module>
    changing_output, opt_state = n_steps(changing_output, opt_state)
MemoryError: bad allocation
```

This is a comment on this situation from GPT-4o:

https://chatgpt.com/share/fc3beba3-79fe-43d4-a9b2-66468a6a2f8f
