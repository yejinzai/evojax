import os
import ray
from ray import tune

import argparse
import os
import shutil
import jax

from evojax.task.slimevolley import SlimeVolley
from evojax.policy.mlp import MLPPolicy
from evojax.algo import CMA, SimpleGA
from evojax import Trainer
from evojax import util
from ray import tune


def main(config):
    log_dir = './log/slimevolley/sga'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='SlimeVolley', log_dir=log_dir, debug=config["debug"])
    logger.info('EvoJAX SlimeVolley')
    logger.info('=' * 30)

    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[config["hidden_size"]],
        output_dim=train_task.act_shape[0],
        output_act_fn='tanh',
    )
    solver = SimpleGA(
        pop_size=config["pop_size"],
        param_size=policy.num_params,
        #init_stdev=config.init_std,
        seed=config["seed"],
        logger=logger,
        sigma=config["sigma"]  # Allow sigma to be tuned
    )

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config["max_iter"],
        log_interval=config["log_interval"],
        test_interval=config["test_interval"],
        n_repeats=config["n_repeats"],
        n_evaluations=config["num_tests"],
        seed=config["seed"],
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Get test score after training (or any relevant metric)
    test_result = trainer.run(demo_mode=True)  # Modify this line based on how your testing works
    tune.report(average_score=test_result['average_score'])  # Report score to Ray Tune


# Ray Tune integration
def run_experiment():
    ray.init(ignore_reinit_error=True)

    search_space = {
        "pop_size": tune.choice([128, 256, 512]),  # Range of population sizes
        "sigma": tune.uniform(0.01, 0.1),          # Sigma range
        "hidden_size": tune.choice([64, 128, 256]),# Hidden layer sizes
        "max_iter": 500,                           # Fixed parameters
        "log_interval": 10,
        "test_interval": 10,
        "n_repeats": 1,
        "num_tests": 5,
        "seed": tune.randint(1, 1000),
        "debug": False                             # Any additional config parameters
    }

    analysis = tune.run(
        main,
        config=search_space,
        num_samples=50,  # Number of hyperparameter configurations to try
        resources_per_trial={"cpu": 12, "gpu": 0},  # Adjust based on resources
    )

    print("Best config found: ", analysis.best_config)
    ray.shutdown()

if __name__ == '__main__':
    run_experiment()