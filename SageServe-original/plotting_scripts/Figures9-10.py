import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

import sys

assert len(sys.argv) == 6, "Usage: python3 Figures9-10.pdf </path/to/reactive> </path/to/lt_i> </path/to/lt_u> </path/to/lt_ua> </path/to/chiron> <output_path>"

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

df1 = get_merged_df(os.path.join(lt_i, 'memory'), sort_by='time')
df1['region_model'] = df1['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
df1['region'] = df1.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
df1['model'] = df1.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)

df2 = get_merged_df(os.path.join(lt_u, 'memory'), sort_by='time')
df2['region_model'] = df2['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
df2['region'] = df2.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
df2['model'] = df2.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)

df3 = get_merged_df(os.path.join(lt_ua, 'memory'), sort_by='time')
df3['region_model'] = df3['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
df3['region'] = df3.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
df3['model'] = df3.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)

df4 = get_merged_df(os.path.join(reactive, 'memory'), sort_by='time')
df4['region_model'] = df4['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
df4['region'] = df4.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
df4['model'] = df4.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)

df5 = get_merged_df(os.path.join(chiron, 'memory'), sort_by='time')
df5['region_model'] = df5['instance'].apply(lambda name: name[:name.index('_', name.index('_')+1)])
df5['region'] = df5.apply(lambda row: row['region_model'][:row['region_model'].index('_')], axis=1)
df5['model'] = df5.apply(lambda row: row['region_model'][row['region_model'].index('_')+1:], axis=1)

f=18
fig = plt.figure(figsize=(8, 4))
ax = fig.gca()
lines=[]
region = 'centralus'
model = ('A', 'A-p', 'A-d')
def process(df1):
    filtered = df1[ (
            (df1['model'].str.startswith(model)) 
                   & (df1['time'] <= 2*86400)
                   & (df1['time'] >= 86400)
                   )
                   ].copy()
    df_grouped = filtered.groupby('time').agg({'instance': lambda x: len(set(x))}).reset_index()
    df_grouped['time'] = df_grouped['time'] // (15*60)
    df_grouped = df_grouped.groupby('time').agg({'instance': lambda x: max(x)}).reset_index()
    return df_grouped, df_grouped['instance'].sum()/4

s1, i1 = process(df1)
s2, i2 = process(df2)
s3, i3 = process(df3)
s4, i4 = process(df4)
s5, i5 = process(df5)
print(i1, i2, i3, i4, i5)
lines.append(ax.plot(s4['time'], s4['instance'], label="Reactive", color=f"C0", linewidth=2))
lines.append(ax.plot(s1['time'], s1['instance'], label="LT-I", color=f"C3", linewidth=2))
lines.append(ax.plot(s2['time'], s2['instance'], label="LT-U", color=f"C1", linewidth=2))
lines.append(ax.plot(s3['time'], s3['instance'], label="LT-UA", color=f"C2", linewidth=2))
lines.append(ax.plot(s5['time'], s3['instance'], label="Chiron", color=f"C4", linewidth=2))

print(s4['time'].min(), s4['time'].max())
print(s1['time'].min(), s1['time'].max())
print(s2['time'].min(), s2['time'].max())
print(s3['time'].min(), s3['time'].max())
print(s5['time'].min(), s5['time'].max())

ax.set_ylim(0, 60)
ax.set_ylabel('Instances Deployed', fontsize=f)
ax.grid(True, which='major', linewidth=1.0)  # Major grid
ax.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7) 

ax.legend(fontsize=f, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
ax.set_xlabel('Hours of the Day', fontsize=f)
ax.set_xticks([96+0, 96+16, 96+32, 96+48, 96+64, 96+80, 96+96])
ax.set_xticklabels([24+0, 24+4, 24+8, 24+12, 24+16, 24+20, 24+24], fontsize=f)
ax.set_yticks([0, 20, 40, 60])
ax.set_yticklabels([0, 20, 40, 60], fontsize=f)

# Set major and minor ticks
ax.xaxis.set_major_locator(MultipleLocator(16))  # Major ticks every 2 units
ax.xaxis.set_minor_locator(MultipleLocator(4))  # Minor ticks every 0.5 units
ax.yaxis.set_major_locator(MultipleLocator(20))  # Major ticks every 1 unit
ax.yaxis.set_minor_locator(MultipleLocator(4))  # Minor ticks every 0.2 units

ax.grid(True, which='major', linewidth=1.0)  # Major grid
ax.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7)  
plt.savefig(f"{output_path}/Figure9.pdf", dpi=256, bbox_inches='tight')

f=18
fig = plt.figure(figsize=(8, 4))
ax = fig.gca()
lines=[]
regions = ['westus', 'centralus', 'eastus']
model = 'A'
def process(df1, region):
    filtered = df1[
        (df1['model'].str.startswith(model)) 
        & (df1['region'] == region) 
        & (df1['time'] <= 2*86400)
        & (df1['time'] >= 86400)
        ].copy()
    df_grouped = filtered.groupby('time').agg({'instance': lambda x: len(set(x))}).reset_index()
    df_grouped['time'] = df_grouped['time'] // (15*60)
    df_grouped = df_grouped.groupby('time').agg({'instance': lambda x: max(x)}).reset_index()
    return df_grouped, df_grouped['instance'].sum()/4
mmap = {}
id = 0 
for df, st, c in [(df4, 'Reactive', 'C0'), (df1, 'LT-I', 'C3'), (df2, 'LT-U', 'C1'), (df3, 'LT-UA', 'C2'), (df5, 'Chiron', 'C4')]:
    for region in regions:
        s, i = process(df, region)
        if st not in mmap:
            mmap[st] = []
        mmap[st].append(i)
    ax.bar([1+id, 6+id, 11+id], mmap[st], width=1, label=st, color=c)
    id+=1

ax.set_ylim(0, 350)
ax.set_xlim(0, 15)
ax.set_ylabel('Instance-Hours', fontsize=f)

ax.legend(fontsize=f, loc='upper right', ncol=2)
ax.set_yticks(list(range(0, 351, 50)))
ax.set_yticklabels(list(range(0, 351, 50)), fontsize=f)
ax.set_xticks([2.5, 7.5, 12.5])
ax.set_xticklabels(regions, fontsize=f)

ax.yaxis.set_major_locator(MultipleLocator(50))  # Major ticks every 1 unit
ax.yaxis.set_minor_locator(MultipleLocator(10))  # Minor ticks every 0.2 units

ax.grid(True, which='major', linewidth=1.0)  # Major grid
ax.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.7)  
plt.savefig(f"{output_path}/Figure10.pdf", dpi=256, bbox_inches='tight')
