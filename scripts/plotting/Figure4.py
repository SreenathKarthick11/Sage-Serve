import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

import os

import sys

assert len(sys.argv) == 4, "Usage: python3 Figure4.py </path/to/sts_unified> </path/to/sts_siloed> <output_directory> "

sts_unified_path = sys.argv[1]
sts_siloed_path = sys.argv[2]
output_path = sys.argv[3]

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

df = get_merged_df(f"{sts_unified_path}/memory/", sort_by='time')
df['region_model'] = df['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
df['region'] = df.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
df['model'] = df.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)

df_silo = get_merged_df(f"{sts_siloed_path}/memory/", sort_by='time')
df_silo['region_model'] = df_silo['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
df_silo['region'] = df_silo.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
df_silo['model'] = df_silo.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)

models = ['B', 'A', 'C', 'D']

f=18
import matplotlib as mpl
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
region = 'centralus'
lines = {m:[] for m in models}
names = {
    'A': 'Llama2-70B',
    'B': 'Bloom-176B',
    'C': 'Llama3.1 & 3.2',
    'D': 'Llama3.2-3B'
}

for i, model in enumerate(models[:3]):
    def process(df1):
        filtered = df1[(df1['model'].str.startswith(model)) & (df1['region'] == region)].copy()
        df_grouped = filtered.groupby('time').agg({'instance': lambda x: len(set(x))}).reset_index()
        df_grouped['time'] = df_grouped['time'] // (15*60)
        df_grouped = df_grouped.groupby('time').agg({'instance': lambda x: max(x)}).reset_index()
        return df_grouped, round(df_grouped['instance'].sum() / 4, 1)
    heur, ha = process(df)
    silo, sa = process(df_silo)
    print(heur['instance'].min(), heur['instance'].max())
    print(silo['instance'].min(), silo['instance'].max())
    print(heur['time'].min(), heur['time'].max())
    print(silo['time'].min(), silo['time'].max())

    # break
    if i==0:
        axes[i].text(96+6+55, -0.2, ha, ha='right', va='bottom', fontsize=f, color="C0")
        axes[i].text(96+6+90, 11, sa, ha='right', va='bottom', fontsize=f, color="C1")
    elif i==1:
        axes[i].text(96+26+25, -0.2, ha, ha='right', va='bottom', fontsize=f, color="C0")
        axes[i].text(96+28+52, 11, sa, ha='right', va='bottom', fontsize=f, color="C1")
    else:
        axes[i].text(96+44+53, -0.2, f"{str(ha)} & {str(ha)}", ha='right', va='bottom', fontsize=f, color="C0")
        axes[i].text(96+44+53, 5, f"{str(sa)} & {str(sa)}", ha='right', va='bottom', fontsize=f, color="C1")

    l = axes[i].plot(heur['time'], heur['instance'], label="Unified", color=f"C0", linewidth=2)
    
    l = axes[i].plot(silo['time'], silo['instance'], label="Siloed", color=f"C1", linewidth=2)
    
    axes[i].set_xlim(96, 192)
    
    axes[i].set_title(f"{names[model]}", loc='center', fontsize=f)
    axes[i].set_yticks([0, 4, 8, 12, 16, 20])
    axes[i].set_yticklabels([0, 4, 8, 12, 16, 20], fontsize=f)
    axes[i].set_xlabel('Hours of the Day', fontsize=f)
    axes[i].set_xticks([96, 128, 160, 192])
    axes[i].set_xticklabels([0, 8, 16, 24], fontsize=f)
    
    # Set major and minor ticks
    axes[i].xaxis.set_major_locator(MultipleLocator(16))  # Major ticks every 2 units
    axes[i].xaxis.set_minor_locator(MultipleLocator(4))  # Minor ticks every 0.5 units
    axes[i].yaxis.set_major_locator(MultipleLocator(4))  # Major ticks every 1 unit
    axes[i].yaxis.set_minor_locator(MultipleLocator(1))  # Minor ticks every 0.2 units

    axes[i].grid(True, which='major', linewidth=1.0)  # Major grid
    axes[i].grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7)  



axes[0].set_ylabel('Instances Deployed', fontsize=f)
axes[2].legend(loc='upper right', fontsize=f)
plt.tight_layout()

plt.savefig(f"{output_path}/Figure4.pdf", dpi=256, bbox_inches='tight', pad_inches=0.1)
