$ cat immutable_utils.py immutable_ops.py immutable_activations.py immutable_engine.py immutable_machine2.py feedforward_with_recurrences.py tree_masked_init.py run_both_networks.py > main.py

Running networks for 2 iterations only.

So 1 effective iteration for loss compute with key != 0 excluded

(TODO: I actually don't understand why it takes two iterations
to make the input values ']['output']['dict-1'][':number']
and ['output']['dict-2'][':number'] well defined; it would be nice
to understand that at some point.)

Learning rate: 0.1, 3 optimization steps

>python -i main.py

---

[...]

```
size (leaves): 2944 2906 37 1
0.8305931091308594  seconds
about to compute gradient
19.9718120098114  seconds to compute gradient
0.0019941329956054688  seconds to apply mask to gradient
18.295085430145264  seconds to compute optimizer update
0.33286166191101074  seconds to apply optimizer update
step: 0 loss: 0.0354757159948349
about to compute gradient
19.806503772735596  seconds to compute gradient
0.0020351409912109375  seconds to apply mask to gradient
0.7079586982727051  seconds to compute optimizer update
0.3329586982727051  seconds to apply optimizer update
step: 1 loss: 0.05428445339202881
about to compute gradient
19.24532675743103  seconds to compute gradient
0.002035379409790039  seconds to apply mask to gradient
0.7056536674499512  seconds to compute optimizer update
0.33129143714904785  seconds to apply optimizer update
step: 2 loss: 0.0007292517693713307
```

Interestingly, compute gradient takes this time each time,
but computing optimizer update becomes much faster the second time.

---

Now let's run networks for 11 iterations.

So 10 effective iterations for loss compute with key != 0 excluded

Mmmmm... gradient compute here scales not quite 10 times, but almost so:

```
size (leaves): 2944 2906 37 1
0.8378427028656006  seconds
about to compute gradient
147.42917728424072  seconds to compute gradient
0.0028443336486816406  seconds to apply mask to gradient
19.0939724445343  seconds to compute optimizer update
0.34244346618652344  seconds to apply optimizer update
step: 0 loss: 0.35485541820526123
about to compute gradient
144.35573720932007  seconds to compute gradient
0.0033926963806152344  seconds to apply mask to gradient
0.7157235145568848  seconds to compute optimizer update
0.339047908782959  seconds to apply optimizer update
step: 1 loss: 0.6878709197044373
about to compute gradient
144.10999464988708  seconds to compute gradient
0.003989219665527344  seconds to apply mask to gradient
0.7232344150543213  seconds to compute optimizer update
0.3300936222076416  seconds to apply optimizer update
step: 2 loss: 0.013595396652817726
```

---

Now let's run networks for 101 iterations.

So 100 effective iterations for loss compute with key != 0 excluded

Let's also do 5 steps instead of 3 to get a better feel for this learning rate

Timewise, the scaling is even (very slightly) worse than 10 times:

```
size (leaves): 2944 2906 37 1
0.8445765972137451  seconds
about to compute gradient
1580.3687193393707  seconds to compute gradient
0.004025936126708984  seconds to apply mask to gradient
18.390220642089844  seconds to compute optimizer update
0.3377394676208496  seconds to apply optimizer update
step: 0 loss: 169.5692138671875
about to compute gradient
1534.9859263896942  seconds to compute gradient
0.004983663558959961  seconds to apply mask to gradient
0.7345867156982422  seconds to compute optimizer update
0.3238065242767334  seconds to apply optimizer update
step: 1 loss: 119.47352600097656
about to compute gradient
1553.8293912410736  seconds to compute gradient
0.003979206085205078  seconds to apply mask to gradient
0.738060474395752  seconds to compute optimizer update
0.3301262855529785  seconds to apply optimizer update
step: 2 loss: 79.21060180664062
about to compute gradient
1546.7848880290985  seconds to compute gradient
0.003988742828369141  seconds to apply mask to gradient
0.7077651023864746  seconds to compute optimizer update
0.3258821964263916  seconds to apply optimizer update
step: 3 loss: 124.45919036865234
about to compute gradient
1543.0470190048218  seconds to compute gradient
0.0039577484130859375  seconds to apply mask to gradient
0.7192633152008057  seconds to compute optimizer update
0.3316986560821533  seconds to apply optimizer update
step: 4 loss: 76.60785675048828
```

too early to say if this learning rate would work
or if it would need adjustment

we need to be ready to adjust learning rate
in the process of running, and also to have some
checkpointing ready
