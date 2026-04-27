import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pg = pd.read_csv('results_pg.csv')
dqn = pd.read_csv('Deep-Q-learning/results_dqn.csv')
baseline = pd.read_csv('Deep-Q-learning/BaselineDataCartPole.csv')

# common step grid shared by all methods
step_grid = np.arange(0, 1_000_001, 5000)


def get_mean_std(df, config):
    # interpolate each seed's (step, score) curve onto the common grid
    seed_curves = []
    for _, group in df[df['Config'] == config].groupby('Seed'):
        steps = group['Step'].values.astype(float)
        scores = group['Score'].values
        interp = np.interp(step_grid, steps, scores)
        # mask steps beyond this seed's data range with NaN
        interp[step_grid > steps[-1]] = np.nan
        seed_curves.append(interp)
    arr = np.array(seed_curves)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


fig, ax = plt.subplots(figsize=(12, 6))

# DQN configurations (dashed, thinner)
dqn_labels = {
    'naive':   'DQN Naive',
    'only_tn': 'DQN Target Network',
    'only_er': 'DQN Experience Replay',
    'tn_er':   'DQN TN + ER',
}
for config in ['naive', 'only_tn', 'only_er', 'tn_er']:
    mean, std = get_mean_std(dqn, config)
    smooth = pd.Series(mean).rolling(5, center=True).mean()
    line, = ax.plot(step_grid, smooth, label=dqn_labels[config], linewidth=1.5, linestyle='--')
    ax.fill_between(step_grid, (mean - std).clip(0, 500), (mean + std).clip(0, 500),
                    alpha=0.1, color=line.get_color())

# Policy gradient methods (solid, thicker)
pg_colors = {'REINFORCE': 'royalblue', 'AC': 'darkorange', 'A2C': 'forestgreen'}
pg_labels = {'REINFORCE': 'REINFORCE', 'AC': 'Actor-Critic (AC)', 'A2C': 'Advantage AC (A2C)'}
for config in ['REINFORCE', 'AC', 'A2C']:
    mean, std = get_mean_std(pg, config)
    smooth = pd.Series(mean).rolling(5, center=True).mean()
    line, = ax.plot(step_grid, smooth, label=pg_labels[config],
                    linewidth=2.5, color=pg_colors[config])
    ax.fill_between(step_grid, (mean - std).clip(0, 500), (mean + std).clip(0, 500),
                    alpha=0.15, color=line.get_color())

# provided baseline
baseline_mean = baseline.groupby('env_step')['Episode_Return_smooth'].mean()
ax.plot(baseline_mean.index, baseline_mean.values,
        label='Provided Baseline', color='black', linestyle=':', linewidth=2)

ax.set_xlabel('Environment Steps', fontsize=12)
ax.set_ylabel('Mean Return', fontsize=12)
ax.set_title('Learning Curves: Policy Gradient Methods vs DQN', fontsize=13)
ax.set_ylim(0, 520)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curves_all.png', dpi=150)
plt.show()
print("Saved to learning_curves_all.png")