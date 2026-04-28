# Assignment 3: REINFORCE and Actor-Critic Methods

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running experiments

Run all three algorithms (REINFORCE, AC, A2C) across 5 seeds and save results to `results_pg.csv`:

```bash
python run_experiments.py
```

Generate the learning curve plot (requires `results_pg.csv` and the DQN results in `Deep-Q-learning/`):

```bash
python plot_models.py
```

## Running individual algorithms

Each training script can also be run standalone to train a single model and show a learning curve:

```bash
python train_REINFORCE.py
python train_AC_model.py
python train_A2C_model.py
```

## File overview

| File | Description |
|------|-------------|
| `run_experiments.py` | Runs all three methods across 5 seeds, saves `results_pg.csv` |
| `plot_models.py` | Loads results and plots learning curves |
| `REINFORCE_model.py` | Policy network for REINFORCE |
| `AC_model.py` | Actor-Critic network (separate actor and Q-network) |
| `A2C_model.py` | Advantage Actor-Critic network (separate actor and V-network) |
| `train_REINFORCE.py` | Training loop for REINFORCE |
| `train_AC_model.py` | Training loop for Actor-Critic |
| `train_A2C_model.py` | Training loop for A2C |
| `results_pg.csv` | Pre-computed results for all three methods (5 seeds each) |
| `learning_curves_all.png` | Pre-generated plot |
| `Deep-Q-learning/` | DQN results and baseline from Assignment 1 (used for comparison plot) |
