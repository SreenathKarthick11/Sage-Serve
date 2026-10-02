import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

import sys

assert len(sys.argv) == 3, "Usage: python3 Figure3.py </path/to/trace> <output_path>"

path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(path)
f=18

fig = plt.figure(figsize=(6, 6), dpi=128)
ax = plt.gca()

title = {'prod': 'Interactive Workloads',
         'dev': 'Non-Interactive Workloads'}
lines = []

for i, workload_type in enumerate(['prod', 'dev']):
    df_filtered = df[(df['workload_type'] == workload_type)].copy()
    df_filtered['second'] = df_filtered['arrival_timestamp'].astype(int)
    df_grouped = df_filtered.groupby('second')['request_id'].count().reset_index()
    df_grouped['rps'] = df_grouped['request_id'].rename('rps')
    df_grouped['4h_bins'] = df_grouped['second'] // (3600 * 4)
    df_grouped = df_grouped.groupby('4h_bins')['rps'].mean().reset_index()
    line, = ax.plot(df_grouped['4h_bins'], df_grouped['rps'], label=workload_type, color=f"C{i}", linewidth=2)
    lines.append(line)
ax.set_xlim(0, 42)

# Set major and minor ticks
ax.xaxis.set_major_locator(MultipleLocator(6))  # Major ticks every 2 units
ax.xaxis.set_minor_locator(MultipleLocator(1.5))  # Minor ticks every 0.5 units
ax.yaxis.set_major_locator(MultipleLocator(5))  # Major ticks every 1 unit
ax.yaxis.set_minor_locator(MultipleLocator(1))  # Minor ticks every 0.2 units

# Enable the grid
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

ax.grid(True, which='major', linewidth=1.0)  # Major grid
ax.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7)  

for pos, label in zip([6*i+3 for i in range(0, 7, 1)], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
    ax.text(pos, -0.35, label, ha='center', va='top', transform=ax.transData, fontsize=f)

ax.set_ylim(0, 30)
ax.set_yticks(list(range(0, 31, 5)))
ax.set_yticklabels(list(range(0, 31, 5)), fontsize=f)

ax.set_xlabel("Day of the Week", fontsize=f, labelpad=16)

# right y begin
ax2 = ax.twinx()
for i, workload_type in enumerate(['prod', 'dev']):
    df_filtered = df[(df['workload_type'] == workload_type)].copy()
    df_filtered['tokens'] = df_filtered['prompt_size'] + df_filtered['token_size']
    df_filtered['second'] = df_filtered['arrival_timestamp'].astype(int)
    df_grouped = df_filtered.groupby('second')['tokens'].sum().reset_index()
    df_grouped['tps'] = df_grouped['tokens'].rename('tps')
    df_grouped['4h_bins'] = df_grouped['second'] // (3600 * 4)
    df_grouped = df_grouped.groupby('4h_bins')['tps'].mean().reset_index()
    line, = ax2.plot(df_grouped['4h_bins'], df_grouped['tps']/1000, label=workload_type, color=f"C{i}", linewidth=2, linestyle='--')

ax2.set_ylabel("TPS (x1000)", fontsize=f)
ax2.set_ylim(0, 150)
ax2.set_yticks(list(range(0, 151, 25)))
ax2.set_yticklabels(list(range(0, 151, 25)), fontsize=f)
#right y end

ax.legend(lines, ['Interactive', 'Non-Interactive'], loc='upper right',bbox_to_anchor=(1.1, 1.15), ncol=2, fontsize=f)
plt.savefig(f"{output_path}/Figure3.pdf", dpi=256, bbox_inches='tight')
