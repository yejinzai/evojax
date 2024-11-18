import numpy as np
import jax
from evojax.task.slimevolley import SlimeVolley
from evojax.policy.mlp import MLPPolicy

def load_model(file_path):
    """Load a trained model from a .npz file."""
    data = np.load(file_path)
    return data['params']

def compete(cma_es_model_path, ga_model_path, max_steps=3000):
    # Initialize the environment
    test_task = SlimeVolley(test=True, max_steps=max_steps)

    # Initialize the policy (same architecture for both models)
    cma_policy = MLPPolicy(
        input_dim=test_task.obs_shape[0],
        hidden_dims=[20, ],  # Adjust this based on your configuration
        output_dim=test_task.act_shape[0],
        output_act_fn='tanh'
    )
    ga_policy = MLPPolicy(
        input_dim=test_task.obs_shape[0],
        hidden_dims=[20, ],  # Adjust this based on your configuration
        output_dim=test_task.act_shape[0],
        output_act_fn='tanh'
    )

    # Load the parameters for both models
    cma_es_params = load_model(cma_es_model_path)[None, :]
    ga_params = load_model(ga_model_path)[None, :]

    # Initialize JAX-compiled functions for the game loop
    task_reset_fn = jax.jit(test_task.reset)
    cma_policy_reset_fn = jax.jit(cma_policy.reset)
    ga_policy_reset_fn = jax.jit(ga_policy.reset)
    step_fn = jax.jit(test_task.step2)
    cma_action_fn = jax.jit(cma_policy.get_actions)
    ga_action_fn = jax.jit(ga_policy.get_actions)

    # Function to play a single match
    def play_match():
        key = jax.random.PRNGKey(0)[None, :]

        # Reset the environment and policies
        task_state = task_reset_fn(key)
        cma_policy_state = cma_policy_reset_fn(task_state)
        ga_policy_state = ga_policy_reset_fn(task_state)

        left_score = 0
        right_score = 0

        # Game loop: Compete for max_steps
        for _ in range(max_steps):
            # CMA-ES player (right) takes action
            cma_action, cma_policy_state = cma_action_fn(task_state, cma_es_params, cma_policy_state)
            # GA player (left) takes action
            ga_action, ga_policy_state = ga_action_fn(task_state, ga_params, ga_policy_state)

            # Step the environment forward with both actions
            task_state, reward, done = step_fn(task_state, ga_action, cma_action)  # left, right

            # Assuming reward contains the scores for left and right players
            left_score += reward[0]  # Left player (GA)
            right_score += reward[1]  # Right player (CMA-ES)

            if done:
                break

        return left_score, right_score

    # Run multiple matches and collect scores
    num_matches = 50000  # Adjust this to run more matches
    left_scores = []
    right_scores = []

    for _ in range(num_matches):
        left_score, right_score = play_match()
        left_scores.append(left_score)
        right_scores.append(right_score)

    # Calculate average and standard deviation for both players
    avg_left_score = np.mean(left_scores)
    std_left_score = np.std(left_scores)
    avg_right_score = np.mean(right_scores)
    std_right_score = np.std(right_scores)

    return avg_left_score, std_left_score, avg_right_score, std_right_score

if __name__ == "__main__":
    cma_es_model_path = './log/slimevolley/cma/model.npz'
    ga_model_path = './log/slimevolley/sga/model.npz'

    avg_left_score, std_left_score, avg_right_score, std_right_score = compete(cma_es_model_path, ga_model_path)
    print(f"GA (left) average score: {avg_left_score}, stddev: {std_left_score}")
    print(f"CMA-ES (right) average score: {avg_right_score}, stddev: {std_right_score}")
