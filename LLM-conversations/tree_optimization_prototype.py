import optax

# Define a simple loss function
def loss_fn(tree):
    # Example loss: sum of all elements in the tree
    return sum([jnp.sum(leaf) for leaf in tree_util.tree_leaves(tree)])

# Initialize optimizer (e.g., gradient descent)
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(random_tree)

# Define a single optimization step
@jax.jit
def step(tree, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(tree)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_tree = optax.apply_updates(tree, updates)
    return new_tree, opt_state, loss

# Run a few optimization steps
for _ in range(10):
    random_tree, opt_state, loss = step(random_tree, opt_state)
    print(f'Loss: {loss}')
