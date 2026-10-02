import os
import matplotlib.pyplot as plt
import pandas as pd

import sys

assert len(sys.argv) == 6, "Usage: python3 Figures11_12_13_15.pdf </path/to/reactive> </path/to/lt_i> </path/to/lt_u> </path/to/lt_ua> </path/to/chiron> <output_path>"

reactive = sys.argv[1]
lt_i = sys.argv[2]
lt_u = sys.argv[3]
lt_ua = sys.argv[4]
chiron = sys.argv[5]
output_path = sys.argv[6]


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


df1 = get_merged_df(os.path.join(lt_i, 'global_router'), sort_by=None)
df2 = get_merged_df(os.path.join(lt_u, 'global_router'), sort_by=None)
df3 = get_merged_df(os.path.join(lt_ua, 'global_router'), sort_by=None)
df4 = get_merged_df(os.path.join(reactive, 'global_router'), sort_by=None)

from matplotlib.ticker import MultipleLocator

f=18
fig, ax = plt.subplots(2, 1, figsize=(8, 5), dpi=256, sharex=True)
models = ['B', 'A', 'B', 'C']
lines=[]
def process(df, model):
    filtered = df[(df['model_type'].str.startswith(model)) & (df['workload_type'] == 'prod') & (df['completion_time'] < (86400*7))].copy()
    filtered['completion_time'] = filtered['completion_time'] // (3600*3)
    df_grouped = filtered.groupby('completion_time').agg({'ttft': lambda x: x.quantile(0.5), 'response_time': lambda x: x.quantile(0.95)}).reset_index()
    return df_grouped

names = {
    'A': 'Llama2-70B',
    'B': 'Bloom-176B',
    'C': 'Llama3.1-8B',
    'D': 'Llama3.2-3B'
}

dp = {'Reactive':{'A':None},
      'LT-I':{'A':None},
      'LT-U':{'A':None},
      'LT-UA':{'A':None}
}

for df, st, c in [(df4, 'Reactive', 'C0'), (df1, 'LT-I', 'C3'), (df2, 'LT-U', 'C1'), (df3, 'LT-UA', 'C2')]:
    for model in ['A']:
        if dp[st][model] == None:
            dp[st][model] = process(df, model)
        dfa = dp[st][model]
        ax[0].plot(dfa['completion_time'], dfa['ttft'], label=st, color=c, linewidth=2)
        ax[1].plot(dfa['completion_time'], dfa['response_time'], label=st, color=c, linewidth=2)
        
for a in ax:
    a.set_xlim(0, 168/3)


    a.grid(True, which='major', linewidth=1.0)  # Major grid
    a.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7)  

ax[0].xaxis.set_major_locator(MultipleLocator(8))  # Major ticks every 2 units
ax[0].xaxis.set_minor_locator(MultipleLocator(2))  # Minor ticks every 0.5 units
ax[0].yaxis.set_major_locator(MultipleLocator(10))  # Major ticks every 1 unit
ax[0].yaxis.set_minor_locator(MultipleLocator(2))  # Minor ticks every 0.2 units

ax[1].xaxis.set_major_locator(MultipleLocator(8))  # Major ticks every 2 units
ax[1].xaxis.set_minor_locator(MultipleLocator(2))  # Minor ticks every 0.5 units
ax[1].yaxis.set_major_locator(MultipleLocator(100))  # Major ticks every 1 unit
ax[1].yaxis.set_minor_locator(MultipleLocator(20))  # Minor ticks every 0.2 units

ax[0].set_yticks([0, 10, 20])
ax[0].set_yticklabels([0, 10, 20], fontsize=f)
ax[1].set_yticks([0, 100, 200])
ax[1].set_yticklabels([0, 100, 200], fontsize=f)

ax[0].legend(loc='upper left', fontsize=f, ncols=2)
ax[0].set_ylim(0, 20)
ax[1].set_ylim(0, 200)
ax[0].set_ylabel('95%ile TTFT (s)', fontsize=f)
ax[1].set_ylabel('95%ile E2E (s)', fontsize=f)
for pos, label in zip([8*i+4 for i in range(0, 7, 1)], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
    ax[1].text(pos, -0.85, label, ha='center', va='top', fontsize=f)

ax[1].set_xticklabels([])

ax[1].set_xlabel("Day of the Week", fontsize=f, labelpad=16)

plt.savefig(f"{output_path}/Figure11.pdf", dpi=256, bbox_inches='tight')

f=18
model = 'A'
regions = ['westus', 'centralus', 'eastus']

tuesday = [
    ('Reactive', df4),
    ('LT-I', df1),
    ('LT-U', df2),
    ('LT-UA', df3)
]

custom_palette = {
    'Reactive': 'C0',
    'LT-U': 'C1',
    'LT-UA': 'C2',
    'LT-I': 'C3'
}

dfa = {
    'util': [],
    'strategy':[],
    'region': [],
}

def process(df1, region):
    filtered = df1[
        (df1['model'].str.startswith(model)) 
        & (df1['region'] == region) 
        & (df1['time'] <= 2*86400)
        & (df1['time'] >= 86400)
        ].copy()

    filtered['util'] = filtered['memory'] / filtered['max_memory']
    df_grouped = filtered.groupby('time').agg({'util': 'max'}).reset_index()
    df_grouped['time'] = df_grouped['time'] // (900)
    df_grouped = df_grouped.groupby('time').agg({'util': 'mean'}).reset_index()
    return list(df_grouped['util'])

for region in regions:
    for strategy, df in tuesday:
        util = process(df, region)
        dfa['util'].extend(util)
        dfa['region'].extend([region]*len(util))
        dfa['strategy'].extend([strategy]*len(util))

dfa = pd.DataFrame(dfa)

import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(8, 4))

sns.boxplot(data=dfa, x='region', y='util', hue='strategy', showfliers=False, whis=[5, 95], palette=custom_palette)
plt.ylim(0, 1)
plt.ylabel('Memory Utilization', fontsize=f)
plt.yticks(fontsize=f)
plt.xticks(fontsize=f)
plt.xlabel('Regions', fontsize=f)
plt.legend(fontsize=f, loc='upper center', bbox_to_anchor=(0.46, 1.25), ncol=4)

a=plt.gca()
a.yaxis.set_major_locator(MultipleLocator(0.2))  # Major ticks every 2 units
a.yaxis.set_minor_locator(MultipleLocator(0.04))  # Minor ticks every 0.5 units
a.grid(True, which='major', linewidth=1.0)  # Major grid
a.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7)  

plt.savefig(f"{output_path}/Figure12.pdf", dpi=256, bbox_inches='tight')

f=18
model = 'A'
regions = ['westus', 'eastus', 'centralus']

tuesday = [
    (reactive, 'Reactive'),
    (lt_i, 'LT-I'),
    (lt_u, 'LT-U'),
    (lt_ua, 'LT-UA')
]
custom_palette = {
    'Reactive': 'C0',
    'LT-U': 'C1',
    'LT-UA': 'C2',
    'LT-I': 'C3'
}
dfa = {
    'latency': [],
    'strategy':[],
    'metric': [],
}

for time, strategy in tuesday:
    path = os.path.join(time, 'global_router')
    df = get_merged_df(path, sort_by=None)
    df = df[(df['workload_type'] == 'prod') & (df['model_type'] == 'A')]
    dfa['latency'].extend(list(df['ttft']))
    print(strategy, "ttft", (df['ttft']).quantile(0.9))
    print(strategy, "e2e", (df['response_time']).quantile(0.9))
    dfa['strategy'].extend([strategy]*len(df))
    dfa['metric'].extend(["TTFT"]*len(df))
    dfa['latency'].extend(list(df['response_time']))
    dfa['strategy'].extend([strategy]*len(df))
    dfa['metric'].extend(["E2E"]*len(df))

dfa = pd.DataFrame(dfa)

import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(8, 4))

sns.boxplot(data=dfa, x='metric', y='latency', hue='strategy', showfliers=False, whis=[5, 95], palette=custom_palette)
plt.legend(loc='upper center')
plt.ylim(1, 110)
plt.grid(axis='y')
plt.ylabel('Latency (s)', fontsize=f)
plt.yticks([1, 10, 100], [1, 10, 100], fontsize=f)
plt.xticks(fontsize=f)
plt.xlabel('Latency Metric', fontsize=f)
plt.legend(fontsize=f, loc='upper center', bbox_to_anchor=(0.5, 1.33), ncol=2)
plt.grid(which='both', axis='y')
ax = fig.gca()
ax.set_yscale('log')
ax.set_yticks([1, 10, 100])
ax.set_yticklabels([1, 10, 100], fontsize=f)
plt.savefig(f"{output_path}/Figure14.pdf", dpi=256, bbox_inches='tight')

dfs = [(df4, 'Reactive'), (df1, 'LT-I'), (df2, 'LT-U'), (df3, 'LT-UA')]
def get_incidents(x, type):
    change = x['scaling'] - x['instance']
    if change <= 0:
        return 0
    elif x['instance'] >= 10 and x['scaling'] > 10:
        return 0 if type=='intra' else change
    elif x['instance'] < 10 and x['scaling'] <= 10:
        return change  if type=='intra' else  0
    else:
        return (10 - x['instance'])  if type=='intra' else  (x['scaling'] - 10)

def process(df, m, r):
    filtered = df[(df['model'].str.startswith(m)) & (df['region'] == r) & (df['time'] < 86400)].copy()
    df_grouped = filtered.groupby('time').agg({'instance': lambda x: len(set(x))}).sort_values(by='time').reset_index()
    df_grouped['scaling'] = df_grouped['instance'].shift(-1)
    df_grouped = df_grouped.iloc[:-1]

    df_grouped['scaling'] = df_grouped['scaling'].astype(int)
    inter = df_grouped.apply(lambda x: get_incidents(x, 'inter'), axis=1).sum()
    intra = df_grouped.apply(lambda x: get_incidents(x, 'intra'), axis=1).sum()
    return inter, intra

mp = {st:[] for _, st in dfs}

for df, strategy in dfs:
    for model in ['A', 'B']:
        inter, intra = 0, 0
        for region in regions:
            i1, i2 = process(df, model, region)
            inter+=i1
            intra+=i2
        mp[strategy].append(intra)
        mp[strategy].append(inter)

import matplotlib.pyplot as plt
import numpy as np
f=18

cat = ["LT-UA", "LT-U", "LT-I", "Reactive"]
group1 = [2.5 * mp[cat[i]][0] / 60 for i in range(4)]  # spot A
group2 = [10 * mp[cat[i]][1] / 60 for i in range(4)]  # inter A
group3 = [2.5 * mp[cat[i]][2] / 60 for i in range(4)]  # spot B
group4 = [10 * mp[cat[i]][3] / 60 for i in range(4)]  # inter B
fig = plt.figure(figsize=(8,4))
ax1 = fig.gca()

y_positions = 3 - np.arange(len(cat))

# Plot the horizontal stacked bars
ax1.barh(y_positions, group1, label='Spot > Llama2-70B', color='#4477AA')
ax1.barh(y_positions, group2, left=group1, label='Other > Llama2-70B', color='#EE6677')
ax1.barh(y_positions, group3, left=np.add(group1, group2), label='Spot > Bloom-176B', color='#228833')
ax1.barh(y_positions, group4, left=np.add(np.add(group1, group2), group3), label='Other > Bloom-176B', color='#CCBB44')

ax1.set_xlim(0, 90)

ax1.set_axisbelow(True)
ax1.minorticks_on()
ax1.grid(which='major', linewidth=1.0, alpha=0.8)
ax1.grid(which='minor', linewidth=0.6, alpha=0.5)

# Add labels and title
ax1.set_yticks(y_positions, cat, fontsize=f)
ax1.set_xlabel('Instance-Hours', fontsize=f)
ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
ax1.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], fontsize=f)
ax1.legend(fontsize=f)
plt.tight_layout()

# Save the figure
plt.savefig(f"{output_path}/Figure15.pdf", dpi=256, bbox_inches='tight')
