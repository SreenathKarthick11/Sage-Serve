import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import seaborn as sns

import os
import sys

assert len(sys.argv) == 5, "Usage: python3 Figure4.py </path/to/sts_unified> </path/to/sts_siloed> </path/to/trace> <output_directory> "

sts_unified_path = sys.argv[1]
sts_siloed_path = sys.argv[2]
trace_path = sys.argv[3]
output_path = sys.argv[4]


f=18
prod_weight = 3
dev_weight = 1
spot_weight_per_15m = 1.5

prod = []
dev = []
spot = []

def get_merged_df(dir_path, sort_by='timestamp'):
    filenames = os.listdir(dir_path)
    frames = [pd.read_csv(os.path.join(dir_path, filenames.pop()))]
    for filename in filenames:
        filepath = os.path.join(dir_path, filename)
        frames.append(pd.read_csv(filepath))
    if sort_by != None:
        df = pd.concat(frames).sort_values(by=sort_by).reset_index()
        return df
    else:
        df = pd.concat(frames)
        return df

df = get_merged_df(os.path.join(sts_unified_path, 'global_router'), sort_by=None)
prod.append(len(df[df['workload_type'] == 'prod']) * prod_weight)
dev.append(len(df[df['workload_type'] == 'dev']) * dev_weight)

temp_df = pd.read_csv(trace_path)
dev_ids = set(temp_df[temp_df['workload_type'].str.startswith('d')]['request_id'])
print(f"Dev requests: {len(dev_ids)}")

df_silo = get_merged_df(os.path.join(sts_siloed_path, 'global_router'), sort_by=None)
prod.append(len(df_silo[~df_silo['request_id'].isin(dev_ids)]) * prod_weight)
dev.append(len(df_silo[df_silo['request_id'].isin(dev_ids)]) * dev_weight)

df = get_merged_df(os.path.join(sts_unified_path, 'memory'), sort_by='time')
df_silo = get_merged_df(os.path.join(sts_siloed_path, 'memory'), sort_by='time')
def process(df1):
    # filtered = df1[(df1['model'].str.startswith(model)) & (df1['region'] == region)].copy()
    df_grouped = df1.groupby('time').agg({'instance': lambda x: len(set(x))}).reset_index()
    df_grouped['time'] = df_grouped['time'] // (15*60)
    df_grouped = df_grouped.groupby('time').agg({'instance': lambda x: max(x)}).reset_index()
    return df_grouped
heur = process(df)
silo = process(df_silo)
spot.append((96*20*4*3 - heur['instance'].sum()) * spot_weight_per_15m)
spot.append((96*20*4*3 - silo['instance'].sum()) * spot_weight_per_15m)
total = [prod[i]+spot[i]+dev[i] for i in range(2)]
print(prod, dev, spot, total)


# Data for the two groups of 3 bars each
group1 = [5, 7, 9]  # First group values
group2 = [4, 6, 8]  # Second group values

# Bar width
bar_width = 1/2

# Create the plot
fig, ax = plt.subplots(1, 2, figsize=(8, 5), dpi=256)

# Plot the bars for each group
bars_group1 = ax[0].bar([1, 3], prod, bar_width, label='IW', color='C0')

bars_group2 = ax[0].bar([1.5, 3.5], dev, bar_width, label='NIW', color='C1')

bars_group3 = ax[0].bar([2, 4], spot, bar_width, label='Spot', color='C2')
for x , y in zip([2, 4], spot):
    ax[0].text(x , y, f" {round(y/1000, 1)}K", ha='center', va='bottom', fontsize=f, color="C2", rotation=90)

# Set the x-ticks to represent the two groups
ax[0].set_xticks([1.5, 3.5])  # Two ticks for two groups
ax[0].set_xticklabels(['Unified', 'Siloed'], fontsize=f)

# Set labels and title
ax[0].set_ylabel('Total Utility Over 1 Day', fontsize=f)

ax[0].set_yticks([1e5, 1e6, 1e7])
ax[0].set_yticklabels([1e5, 1e6, 1e7], fontsize=f)

# Add a legend
ax[0].legend(fontsize=f, ncol=4)
ax[0].set_yscale('log')
ax[0].set_ylim(1e4, 1e7)

ax[0].grid(True, which='major', linewidth=1.0)  # Major grid
ax[0].grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7) 
ax[0].legend(fontsize=f, loc='upper center', ncol=2)

# right
f=18
models = ['B', 'A', 'C', 'D']
region = 'westus'

df = get_merged_df(os.path.join(sts_unified_path, 'memory'), sort_by='time')
df_silo = get_merged_df(os.path.join(sts_siloed_path, 'memory'), sort_by='time')

def process(df1):
    df1['region_model'] = df1['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
    df1['region'] = df1.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
    df1['model'] = df1.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)
    df1['model_name'] = df1['model'].apply(lambda x: x[0])
    filtered = df1[((df1['model'].str.startswith(models[0])) | (df1['model'].str.startswith(models[1]))|(df1['model'].str.startswith(models[2])) | (df1['model'].str.startswith(models[3]))) & (df1['region'] == region)].copy()
    df_grouped = filtered.groupby(['time', 'model_name']).agg({'memory': 'sum', 'max_memory': 'sum'}).reset_index()
    df_grouped['util'] = df_grouped['memory'] / df_grouped['max_memory']
    df_grouped['time'] = df_grouped['time'] // (1)
    df_grouped = df_grouped.groupby(['time', 'model_name']).agg({'util': "mean"}).reset_index()
    return df_grouped

heur = process(df)
silo = process(df_silo)


dfa = {
    'util': [],
    'model':[],
    'strategy':[]
}
dfa['model'].extend(list(heur['model_name']))
dfa['util'].extend(list(heur['util']))
dfa['strategy'].extend(['Unified'] * len(heur))

dfa['model'].extend(list(silo['model_name']))
dfa['util'].extend(list(silo['util']))
dfa['strategy'].extend(['Siloed'] * len(silo))

dfa = pd.DataFrame(dfa)
dfa['util'] = dfa['util'] * 100

names = {
    'A': 'Llama2-70B',
    'B': 'Bloom-176B',
    'C': 'Llama3.1-8B',
    'D': 'Llama3.2-3B'
}
order = sorted(names.values())
dfa['model'] = dfa.apply(lambda row: names[row['model']], axis=1)

sns.boxplot(dfa, x='model', y='util', hue='strategy', ax=ax[1], whis=[5, 95], showfliers=False, order=order, width=0.5)
ax[1].grid(True, which='major', linewidth=1.0)  # Major grid
ax[1].grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7) 
ax[1].set_ylabel('Effective Memory Util (%)', fontsize=f)
ax[1].set_xticklabels(['B', 'L', 'L3.1', 'L3.2'], fontsize=f)
ax[1].set_yscale('log')
ax[1].set_xlabel('')

ax[1].legend(fontsize=f, loc='upper center', bbox_to_anchor=(0.5, 1.33), ncol=1)


plt.tight_layout()
plt.savefig(f"{output_path}/Figure5.pdf", dpi=256, bbox_inches='tight')
