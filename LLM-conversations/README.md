### Conversations with GPT-4o on JAX technicalities

https://chatgpt.com/share/a4dd4764-091f-4b92-93da-d6d0cf145be1

(On JAX immutable tree optimization and random initialization intricacies,
see also **Random numbers** section in **JAX - The Sharp Bits**, 
https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)

[tree_init_prototype.py](tree_init_prototype.py) - a very nice LLM-generated prototype for
tree initialization with random leaves in JAX (that's what I would do too, compute
how many leaves are there, generate all the keys required by JAX way to generate
random numbers, and then use them all to initialize leaves, and then
unflatten the structure back into tree, avoiding any tree traversal and any complex logic)
