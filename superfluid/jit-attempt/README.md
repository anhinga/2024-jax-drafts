these changes in input-dummy-helper allow us to uncomment #@jax.jit on line 83 of run_both_networks.py,
but after running 1 step the program starts behaving strangely and ultimately breaks down.

the run @jax.jit commented out is still fine, although noticeably slower 
because way more {key: 0.0} are passed around instead of being omitted.

Perhaps one of the conditions mentioned in this chat is violated
(perhaps this is because function `step` is decorated with @jax.jit, but the code which
calls `step` is not decorated with @jax.jit, this might be the reason, I should try to fix that,
see whether this would work).

https://chatgpt.com/share/dcda0b70-5a83-410c-ab24-1d1c65e6ed0b

**ME:**

Hi, do you know why a JAX program might run OK without jax.jit, start running OK with jax.jit decorator enabled for one of its functions, but break down after first iteration. Is it that one should enable jax.jit everywhere then?

**GPT-4o:**

The issue you’re encountering could be related to how `jax.jit` works. When you apply `jax.jit` to a function, JAX compiles that function into a more efficient, static computation graph. This static graph is optimized for performance, but it assumes that the shapes and types of the inputs remain consistent across calls.

Some possible reasons for the behavior you're seeing include:

  **1. Input shape/type changes:** If the shape or type of the inputs to the `jax.jit`-decorated function changes between iterations, it could cause problems because the function was compiled for the initial input shape and type. JAX compiles functions to work with static shapes/types, and dynamic changes are not handled automatically without recompilation.

  **2. Mutable vs Immutable State:** If your function is using mutable state that is not handled properly (for example, trying to update variables in-place), it could lead to inconsistencies. Since `jax.jit` works with pure functions (no side effects), mutable state should be avoided, and updates to variables should be done using immutable patterns.

  **3. Tracing issues:** If some part of the code is traced during JIT compilation and behaves differently in subsequent iterations (e.g., due to control flow or operations on traced values), it could break after the first iteration. This can occur when using control flow primitives or when dealing with JAX transformations.

  **4. Partial usage of `jax.jit`:** You don’t necessarily need to apply `jax.jit` everywhere, but applying it selectively could lead to issues if the boundary between JIT-compiled and non-compiled code introduces inconsistencies, such as with shapes, types, or side effects. It’s usually good practice to apply `jax.jit` to performance-critical sections but ensure that all parts of the computation are compatible with it.

Potential solutions:

  **Shape Consistency:** Ensure that the input shapes and types remain consistent across iterations.
  **Immutable Data:** Avoid in-place mutations. Use functional updates to data structures.
  **Selective Use of `jax.jit`:** If you use `jax.jit` selectively, ensure the decorated functions interact properly with the rest of the code. It’s possible that by only applying `jax.jit` to one function, some dependencies or assumptions break after the first call.

If you can share more details about the error or the behavior you observe after the first iteration, I can help diagnose the issue more specifically.

**NOTE:** The conversation continues further at https://chatgpt.com/share/dcda0b70-5a83-410c-ab24-1d1c65e6ed0b

It has useful information about JAX-specific loops, about intricacies of usage of jax.lax.scan function, etc. 

**NOTE:**

There is also this interesting thing: https://jax.readthedocs.io/en/latest/_autosummary/jax.disable_jit.html

Perhaps, one could avoid this super-ugliness with it (TODO: try this).

**NOTE:**

> The error occurred in the __reduce__ method, which may indicate an attempt to serialize/pickle a traced value.

That's super-unpleasant. How does one checkpoint inside a jax.jit? (TODO: investigate)

See the discussion on pickling, printing, and slow compilation added to the end of

https://chatgpt.com/share/dcda0b70-5a83-410c-ab24-1d1c65e6ed0b

We should watch for slow compilation because GPT-4o says, in particular

> **Repeatedly long compilations:** If every iteration or function call involves re-compilation and takes over a minute, it may indicate that something is causing JAX to repeatedly trace and recompile, which shouldn't happen in typical JAX workflows.

This might be a problem with very dynamic configurations

**SOME SUCCESS:**

The program worked end-to-end; no printout of loss values though due to the issues
mentioned above which we still need to resolve:

```
size (leaves): 2944 2906 37 1
0.7891635894775391  seconds
initial loss  0.035475716  computed in  0.7960395812988281  seconds
about to compute gradient
21.114875316619873  seconds to compute gradient
0.0020148754119873047  seconds to apply mask to gradient
8.733324527740479  seconds to compute optimizer update
0.9306800365447998  seconds to apply optimizer update
step: Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=2/0)> loss: Traced<ShapedArray(float32[])>with<DynamicJaxprTrace(level=2/0)>
2024-09-08 12:07:09.900690: E external/xla/xla/service/slow_operation_alarm.cc:65]
********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
2024-09-08 12:13:16.846290: E external/xla/xla/service/slow_operation_alarm.cc:133] The operation took 8m6.9600373s

********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
>>>
```

---

One is supposed to use jax.debug.print() as in

https://jax.readthedocs.io/en/latest/debugging/index.html

https://jax.readthedocs.io/en/latest/debugging/print_breakpoint.html

However, in practice, it does not seem to work like advertised,
and GPT-4o also seems to say that it should not be used like this in

https://chatgpt.com/share/dcda0b70-5a83-410c-ab24-1d1c65e6ed0b

so I am not sure what's the correct way:

```
size (leaves): 2944 2906 37 1
0.7916030883789062  seconds
initial loss  0.03547571  computed in  0.8783204555511475  seconds
about to compute gradient
21.286386251449585  seconds to compute gradient
0.0010318756103515625  seconds to apply mask to gradient
8.418695449829102  seconds to compute optimizer update
1.3830230236053467  seconds to apply optimizer update
2024-09-09 01:13:06.997504: E external/xla/xla/service/slow_operation_alarm.cc:65]
********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
2024-09-09 01:18:44.571707: E external/xla/xla/service/slow_operation_alarm.cc:133] The operation took 7m37.5899471s

********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
step: Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=2/0)> loss: Traced<ShapedArray(float32[])>with<DynamicJaxprTrace(level=2/0)>
step: Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=2/0)> loss: Traced<ShapedArray(float32[])>with<DynamicJaxprTrace(level=2/0)>
step: Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=2/0)> loss: Traced<ShapedArray(float32[])>with<DynamicJaxprTrace(level=2/0)>
```
