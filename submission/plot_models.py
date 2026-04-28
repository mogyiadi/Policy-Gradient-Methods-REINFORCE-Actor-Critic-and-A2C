import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pg = pd.read_csv('results_pg.csv')
dqn = pd.read_csv('Deep-Q-learning/results_dqn.csv')
baseline = pd.read_csv('Deep-Q-learning/BaselineDataCartPole.csv')

# common step grid shared by all methods
step_grid = np.arange(0, 1_000_001, 5000)


def get_mean_std(df, config):
    # interpolate each seed's curve onto the common grid
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

# DQN configurations
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

# policy gradient methods
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

# inset zoomed into AC's actual step range, uses raw per-seed data since
# the global grid is too coarse to capture AC's 7000-step range
ax_inset = ax.inset_axes([0.42, 0.07, 0.38, 0.30])
ac_max_step = 0
for config in ['REINFORCE', 'AC', 'A2C']:
    all_steps, all_scores = [], []
    for _, group in pg[pg['Config'] == config].groupby('Seed'):
        all_steps.append(group['Step'].values)
        all_scores.append(group['Score'].values)
    # build a fine grid over this config's step range
    max_step = max(s[-1] for s in all_steps)
    fine_grid = np.linspace(0, max_step, 500)
    seed_curves = [np.interp(fine_grid, s, sc) for s, sc in zip(all_steps, all_scores)]
    mean = np.mean(seed_curves, axis=0)
    smooth = pd.Series(mean).rolling(10, center=True).mean()
    ax_inset.plot(fine_grid, smooth, color=pg_colors[config], linewidth=1.5)
    if config == 'AC':
        ac_max_step = max_step
ax_inset.set_ylim(0, 30)
ax_inset.set_xlim(0, ac_max_step * 1.05)
ax_inset.set_title('Zoom: AC performance', fontsize=8)
ax_inset.set_xlabel('Steps', fontsize=7)
ax_inset.set_ylabel('Return', fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves_all.png', dpi=150)
plt.show()
print("Saved to learning_curves_all.png")
