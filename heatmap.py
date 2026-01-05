import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Sample data based on your layer report
data = {
    'Layer': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attn_Heads': [1/12, 0/12, 0/12, 0/12, 10/12, 0/12, 2/12, 0/12, 0/12, 9/12, 12/12, 4/12],
    'Attn_Neurons': [43/768, 0/768, 0/768, 0/768, 531/768, 0/768, 100/768, 0/768, 0/768, 391/768, 503/768, 167/768],
    'MLP_Hidden': [2632/3072, 0/3072, 0/3072, 0/3072, 0/3072, 2309/3072, 0/3072, 2018/3072, 0/3072, 0/3072, 0/3072, 1152/3072],
    'MLP_Output': [741/768, 0/768, 0/768, 0/768, 0/768, 744/768, 0/768, 730/768, 0/768, 0/768, 0/768, 650/768],
}

df = pd.DataFrame(data)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Neural Network Layer Activation Heatmaps', fontsize=16, fontweight='bold')

# Custom colormap: green for pruned (0), red for active (1)
colors_main = ['#2ecc71', '#ff4444']  # Green to red
cmap_main = LinearSegmentedColormap.from_list('pruned_green', colors_main)

# 1. Combined Overview Heatmap
ax1 = axes[0, 0]
heatmap_data = df[['Attn_Heads', 'Attn_Neurons', 'MLP_Hidden', 'MLP_Output']].T
sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap=cmap_main, 
            xticklabels=df['Layer'], yticklabels=['Attn Heads', 'Attn Neurons', 'MLP Hidden', 'MLP Output'],
            cbar_kws={'label': 'Activation Ratio (Green=Pruned, Red=Active)'}, ax=ax1, vmin=0, vmax=1)
ax1.set_title('Layer Component Activation Overview', fontweight='bold')
ax1.set_xlabel('Layer')
ax1.set_ylabel('Component')

# 2. Attention Components Heatmap
ax2 = axes[0, 1]
attn_data = df[['Attn_Heads', 'Attn_Neurons']].T
cmap_attn = LinearSegmentedColormap.from_list('attn', ['#2ecc71', '#3498db'])  # Green to blue
sns.heatmap(attn_data, annot=True, fmt='.2%', cmap=cmap_attn,
            xticklabels=df['Layer'], yticklabels=['Attention Heads', 'Attention Neurons'],
            cbar_kws={'label': 'Activation Ratio'}, ax=ax2, vmin=0, vmax=1)
ax2.set_title('Attention Block Activation', fontweight='bold')
ax2.set_xlabel('Layer')
ax2.set_ylabel('Component')

# 3. MLP Components Heatmap
ax3 = axes[1, 0]
mlp_data = df[['MLP_Hidden', 'MLP_Output']].T
cmap_mlp = LinearSegmentedColormap.from_list('mlp', ['#2ecc71', '#e67e22'])  # Green to orange
sns.heatmap(mlp_data, annot=True, fmt='.2%', cmap=cmap_mlp,
            xticklabels=df['Layer'], yticklabels=['MLP Hidden', 'MLP Output'],
            cbar_kws={'label': 'Activation Ratio'}, ax=ax3, vmin=0, vmax=1)
ax3.set_title('MLP Block Activation', fontweight='bold')
ax3.set_xlabel('Layer')
ax3.set_ylabel('Component')

# 4. Binary Pruning Status
ax4 = axes[1, 1]
# Create binary status (1 if any component active, 0 if fully pruned)
attn_status = (df['Attn_Heads'] > 0).astype(int)
mlp_status = (df['MLP_Hidden'] > 0).astype(int) | (df['MLP_Output'] > 0).astype(int)
binary_data = pd.DataFrame({
    'Attn Block': attn_status,
    'MLP Block': mlp_status
}).T

sns.heatmap(binary_data, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=df['Layer'], yticklabels=['Attention Block', 'MLP Block'],
            cbar_kws={'label': 'Status (0=Pruned, 1=Active)'}, ax=ax4, vmin=0, vmax=1)
ax4.set_title('Block Pruning Status', fontweight='bold')
ax4.set_xlabel('Layer')
ax4.set_ylabel('Block Type')

plt.tight_layout()
plt.savefig('layer_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional: Create a detailed single heatmap with all metrics
fig2, ax = plt.subplots(figsize=(14, 6))
detailed_data = df[['Attn_Heads', 'Attn_Neurons', 'MLP_Hidden', 'MLP_Output']].T

sns.heatmap(detailed_data, annot=True, fmt='.1%', cmap='viridis',
            xticklabels=df['Layer'], 
            yticklabels=['Attn Heads\n(x/12)', 'Attn Neurons\n(x/768)', 
                        'MLP Hidden\n(x/3072)', 'MLP Output\n(x/768)'],
            cbar_kws={'label': 'Active Component Ratio'}, ax=ax, vmin=0, vmax=1,
            linewidths=0.5, linecolor='gray')

ax.set_title('Detailed Layer-wise Component Activation Heatmap', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Component Type', fontsize=12)

# Add vertical lines to separate layers visually
for i in range(1, len(df)):
    ax.axvline(i, color='white', linewidth=2)

plt.tight_layout()
plt.savefig('detailed_layer_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Heatmaps generated successfully!")
print(f"\nSummary Statistics:")
print(f"Total layers: {len(df)}")
print(f"Layers with active attention: {(df['Attn_Heads'] > 0).sum()}")
print(f"Layers with active MLP: {((df['MLP_Hidden'] > 0) | (df['MLP_Output'] > 0)).sum()}")
print(f"Fully pruned layers: {((df['Attn_Heads'] == 0) & (df['MLP_Hidden'] == 0) & (df['MLP_Output'] == 0)).sum()}")