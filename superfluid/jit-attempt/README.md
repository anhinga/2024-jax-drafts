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

However, this only works for values which can be computed during
tracing, the following conversation includes a more detailed explanation now:

https://chatgpt.com/share/dcda0b70-5a83-410c-ab24-1d1c65e6ed0b

The version which is currently committed still needs to be modified further:

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

---

Let's take this jax.debug.print() out, and run one step at a time first:

```
size (leaves): 2944 2906 37 1
0.8519880771636963  seconds
initial loss  0.035475712  computed in  0.8386034965515137  seconds
about to compute gradient
24.922187328338623  seconds to compute gradient
0.001995563507080078  seconds to apply mask to gradient
9.084045886993408  seconds to compute optimizer update
0.9593539237976074  seconds to apply optimizer update
>>> start_time = time.time()
>>> print("initial loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
initial loss  0.05428444  computed in  1.9356799125671387  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
about to compute gradient
21.254416704177856  seconds to compute gradient
0.0029921531677246094  seconds to apply mask to gradient
8.152381658554077  seconds to compute optimizer update
1.8211722373962402  seconds to apply optimizer update
2024-09-10 00:24:16.277113: E external/xla/xla/service/slow_operation_alarm.cc:65]
********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
2024-09-10 00:24:55.221611: E external/xla/xla/service/slow_operation_alarm.cc:133] The operation took 2m38.967864s

********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
loss  0.0007292521  computed in  340.73206329345703  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
loss  0.007392578  computed in  4.688348293304443  seconds
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
loss  0.007392578  computed in  17.867120027542114  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
loss  0.012197803  computed in  1.7861933708190918  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
loss  2.1855567e-05  computed in  1.6253340244293213  seconds
>>>
```

Yes, it looks like it starts progressing OK and saving time.

It looks like it need to jit-compile twice in our workflow before settling in.

Each jit-compilation is taking quite a long time, unfortunately.

Let's go back to 3 at a time:

```
size (leaves): 2944 2906 37 1
0.8257303237915039  seconds
initial loss  0.035475716  computed in  0.8300740718841553  seconds
about to compute gradient
22.04065179824829  seconds to compute gradient
0.001992464065551758  seconds to apply mask to gradient
8.976167678833008  seconds to compute optimizer update
0.9265081882476807  seconds to apply optimizer update
2024-09-10 00:33:21.087422: E external/xla/xla/service/slow_operation_alarm.cc:65]
********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
2024-09-10 00:39:11.512502: E external/xla/xla/service/slow_operation_alarm.cc:133] The operation took 7m50.4399906s

********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
>>> start_time = time.time()
>>> print("initial loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")
initial loss  0.0073925783  computed in  0.9378523826599121  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
about to compute gradient
21.542019367218018  seconds to compute gradient
0.002992391586303711  seconds to apply mask to gradient
8.198374271392822  seconds to compute optimizer update
1.855971097946167  seconds to apply optimizer update
2024-09-10 00:58:18.900262: E external/xla/xla/service/slow_operation_alarm.cc:133] The operation took 10m7.7993661s

********************************
[Compiling module jit_n_steps] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")\
...
loss  0.008919513  computed in  882.9559900760651  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")\
...
loss  0.0028272693  computed in  2.696471929550171  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")\
...
loss  0.00082237046  computed in  2.217298984527588  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")\
...
loss  0.00017617253  computed in  1.3293540477752686  seconds
>>> start_time = time.time()
>>> changing_output, opt_state = n_steps(changing_output, opt_state)
>>> print("loss ", loss_fn(changing_output), " computed in ", time.time()-start_time, " seconds")\
...
loss  0.00043086061  computed in  1.6900997161865234  seconds
>>>
```

It does save tons of time, after it stops recompiling.

Verifying correctness (meaning that the sequence of loss values does not depend on how I split it, 
and does not depend on this optimization). Well, almost (the difference seems to be related
to float64 vs float32, judging by what is printed without jax.jit):

```
size (leaves): 2944 2906 37 1
0.9013302326202393  seconds
initial loss  0.035475712  computed in  0.6728272438049316  seconds
about to compute gradient
20.856366872787476  seconds to compute gradient
0.002991914749145508  seconds to apply mask to gradient
19.8409423828125  seconds to compute optimizer update
0.36291027069091797  seconds to apply optimizer update
0.15926003456115723  seconds to pickle the checkpoint
step: 0 loss: 0.0354757122695446
about to compute gradient
20.958195686340332  seconds to compute gradient
0.001993894577026367  seconds to apply mask to gradient
0.8130216598510742  seconds to compute optimizer update
0.3370809555053711  seconds to apply optimizer update
0.09618377685546875  seconds to pickle the checkpoint
step: 1 loss: 0.05428445339202881
about to compute gradient
21.22022008895874  seconds to compute gradient
0.002960681915283203  seconds to apply mask to gradient
0.8408749103546143  seconds to compute optimizer update
0.41936516761779785  seconds to apply optimizer update
0.11873912811279297  seconds to pickle the checkpoint
step: 2 loss: 0.0007292521186172962
about to compute gradient
41.068897008895874  seconds to compute gradient
0.0025017261505126953  seconds to apply mask to gradient
0.8405508995056152  seconds to compute optimizer update
0.44580602645874023  seconds to apply optimizer update
0.12447023391723633  seconds to pickle the checkpoint
step: 3 loss: 0.00739257549867034
about to compute gradient
23.3280668258667  seconds to compute gradient
0.0029926300048828125  seconds to apply mask to gradient
0.7781815528869629  seconds to compute optimizer update
0.36182236671447754  seconds to apply optimizer update
0.15974164009094238  seconds to pickle the checkpoint
step: 4 loss: 0.01219780184328556
about to compute gradient
21.570862770080566  seconds to compute gradient
0.002031564712524414  seconds to apply mask to gradient
0.8262112140655518  seconds to compute optimizer update
0.35382723808288574  seconds to apply optimizer update
0.09986257553100586  seconds to pickle the checkpoint
step: 5 loss: 2.1855727027286775e-05
about to compute gradient
21.928231477737427  seconds to compute gradient
0.003579378128051758  seconds to apply mask to gradient
0.7629561424255371  seconds to compute optimizer update
0.35016417503356934  seconds to apply optimizer update
0.18241429328918457  seconds to pickle the checkpoint
step: 6 loss: 0.00891951099038124
about to compute gradient
21.05940008163452  seconds to compute gradient
0.0019915103912353516  seconds to apply mask to gradient
0.776029109954834  seconds to compute optimizer update
0.35353922843933105  seconds to apply optimizer update
0.09961581230163574  seconds to pickle the checkpoint
step: 7 loss: 0.0007916731992736459
about to compute gradient
20.884963750839233  seconds to compute gradient
0.0019943714141845703  seconds to apply mask to gradient
0.768965482711792  seconds to compute optimizer update
0.3523569107055664  seconds to apply optimizer update
0.14151978492736816  seconds to pickle the checkpoint
step: 8 loss: 0.004127935506403446
about to compute gradient
20.616408109664917  seconds to compute gradient
0.002991914749145508  seconds to apply mask to gradient
0.7481563091278076  seconds to compute optimizer update
0.35838842391967773  seconds to apply optimizer update
0.14621615409851074  seconds to pickle the checkpoint
step: 9 loss: 0.002827269723638892
about to compute gradient
20.621436834335327  seconds to compute gradient
0.0019948482513427734  seconds to apply mask to gradient
0.8227410316467285  seconds to compute optimizer update
0.3421146869659424  seconds to apply optimizer update
0.09692740440368652  seconds to pickle the checkpoint
step: 10 loss: 0.0017608627676963806
about to compute gradient
20.91184115409851  seconds to compute gradient
0.001995086669921875  seconds to apply mask to gradient
0.7673983573913574  seconds to compute optimizer update
0.3351883888244629  seconds to apply optimizer update
0.09459972381591797  seconds to pickle the checkpoint
step: 11 loss: 0.0031388315837830305
about to compute gradient
21.143537521362305  seconds to compute gradient
0.0019941329956054688  seconds to apply mask to gradient
0.7495288848876953  seconds to compute optimizer update
0.3453848361968994  seconds to apply optimizer update
0.09512090682983398  seconds to pickle the checkpoint
step: 12 loss: 0.0008223708136938512
about to compute gradient
20.88778042793274  seconds to compute gradient
0.0019888877868652344  seconds to apply mask to gradient
0.7552587985992432  seconds to compute optimizer update
0.36513662338256836  seconds to apply optimizer update
0.09516215324401855  seconds to pickle the checkpoint
step: 13 loss: 0.0025687427259981632
about to compute gradient
20.750274896621704  seconds to compute gradient
0.001994609832763672  seconds to apply mask to gradient
0.7585961818695068  seconds to compute optimizer update
0.339508056640625  seconds to apply optimizer update
0.09794187545776367  seconds to pickle the checkpoint
step: 14 loss: 0.0018979982705786824
about to compute gradient
20.79806160926819  seconds to compute gradient
0.002995014190673828  seconds to apply mask to gradient
1.139662504196167  seconds to compute optimizer update
0.3610661029815674  seconds to apply optimizer update
0.09606385231018066  seconds to pickle the checkpoint
step: 15 loss: 0.0001761727180564776
about to compute gradient
22.334072828292847  seconds to compute gradient
0.001995086669921875  seconds to apply mask to gradient
0.7743179798126221  seconds to compute optimizer update
0.3438103199005127  seconds to apply optimizer update
0.2990574836730957  seconds to pickle the checkpoint
step: 16 loss: 0.002153585199266672
about to compute gradient
21.401716709136963  seconds to compute gradient
0.002992391586303711  seconds to apply mask to gradient
0.7386503219604492  seconds to compute optimizer update
0.3581211566925049  seconds to apply optimizer update
0.09840965270996094  seconds to pickle the checkpoint
step: 17 loss: 0.0010261248098686337
about to compute gradient
20.97990083694458  seconds to compute gradient
0.002025604248046875  seconds to apply mask to gradient
0.7476904392242432  seconds to compute optimizer update
0.3432028293609619  seconds to apply optimizer update
0.09704899787902832  seconds to pickle the checkpoint
step: 18 loss: 0.0004308591305743903
```
