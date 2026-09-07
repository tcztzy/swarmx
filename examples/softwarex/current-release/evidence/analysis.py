import json
import os

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(os.environ['SWARMX_SCIENCE_INPUT_0'])
data['percent'] = 100 * data.germinated / data.total
means = data.groupby('treatment').percent.mean()
assert means['control'] == 79 and means['primed'] == 87
fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
ax.bar([0, 1], [means['control'], means['primed']], width=0.52, color=['#9ca3af', '#374151'])
for i, group in enumerate(['control', 'primed']):
    ax.scatter([i-0.08, i, i+0.08], data.loc[data.treatment == group, 'percent'], s=22, color='white', edgecolor='black', zorder=3)
    ax.text(i, means[group]+4, f'{means[group]:.0f}%', ha='center', fontsize=13)
ax.set(xticks=[0, 1], xticklabels=['Control (n = 3)', 'Primed (n = 3)'], ylabel='Germination (%)', ylim=(0, 100), title='Synthetic germination fixture')
ax.spines[['top', 'right']].set_visible(False)
fig.supxlabel('Difference: 8 percentage points | Demonstration data only', fontsize=9)
fig.savefig('germination.png', dpi=180)
plt.close(fig)
print(json.dumps({'controlMeanPercent': float(means['control']), 'primedMeanPercent': float(means['primed']), 'differencePercentagePoints': float(means['primed']-means['control']), 'replicatesPerGroup': 3, 'scope': 'synthetic demonstration; no biological inference'}))
