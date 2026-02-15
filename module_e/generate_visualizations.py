import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os

# Create output directory
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'bm25': '#3498db', 'semantic': '#e74c3c', 'hybrid': '#2ecc71'}

# Read data
perf_summary = pd.read_csv('Model Performance Summary.csv')
query_type = pd.read_csv('Results by Query Type.csv')
error_analysis = pd.read_csv('Error Analysis Summary.csv')
detailed_results = pd.read_csv('Detailed Query Results.csv')

print("Generating visualizations...")

# ============================================================================
# Figure 1: Model Performance Comparison (Bar Chart) - Precision@10
# ============================================================================
print("Creating Figure 1: Model Performance Comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

# Data for precision@10 by query type
query_types = ['Cross-Lingual', 'Same-Language', 'Code-Switched']
methods = ['BM25', 'Semantic', 'Hybrid']

# Extract data from query_type dataframe
cross_lingual = query_type[query_type['query_type'] == 'cross-lingual'][['method', 'avg_precision_at_10']]
same_language = query_type[query_type['query_type'] == 'same-language'][['method', 'avg_precision_at_10']]
code_switched = query_type[query_type['query_type'] == 'code-switched'][['method', 'avg_precision_at_10']]

bm25_scores = [
    cross_lingual[cross_lingual['method'] == 'bm25']['avg_precision_at_10'].values[0],
    same_language[same_language['method'] == 'bm25']['avg_precision_at_10'].values[0],
    code_switched[code_switched['method'] == 'bm25']['avg_precision_at_10'].values[0]
]

semantic_scores = [
    cross_lingual[cross_lingual['method'] == 'semantic']['avg_precision_at_10'].values[0],
    same_language[same_language['method'] == 'semantic']['avg_precision_at_10'].values[0],
    code_switched[code_switched['method'] == 'semantic']['avg_precision_at_10'].values[0]
]

hybrid_scores = [
    cross_lingual[cross_lingual['method'] == 'hybrid']['avg_precision_at_10'].values[0],
    same_language[same_language['method'] == 'hybrid']['avg_precision_at_10'].values[0],
    code_switched[code_switched['method'] == 'hybrid']['avg_precision_at_10'].values[0]
]

x = np.arange(len(query_types))
width = 0.25

bars1 = ax.bar(x - width, bm25_scores, width, label='BM25', color=colors['bm25'], edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, semantic_scores, width, label='Semantic', color=colors['semantic'], edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, hybrid_scores, width, label='Hybrid', color=colors['hybrid'], edgecolor='black', linewidth=1.2)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

ax.set_ylabel('Precision@10', fontsize=14, fontweight='bold')
ax.set_xlabel('Query Type', fontsize=14, fontweight='bold')
ax.set_title('PRECISION@10 COMPARISON', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(query_types, fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure1_precision_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 2: Precision@10 by Query Type (Horizontal Bar)
# ============================================================================
print("Creating Figure 2: Precision@10 by Query Type...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

query_type_data = [
    ('CROSS-LINGUAL QUERIES (English → Bangla)', cross_lingual),
    ('SAME-LANGUAGE QUERIES (Bangla → Bangla)', same_language),
    ('CODE-SWITCHED QUERIES (Mixed)', code_switched)
]

for idx, (title, data) in enumerate(query_type_data):
    ax = axes[idx]

    methods_list = ['BM25', 'Semantic', 'Hybrid']
    values = [
        data[data['method'] == 'bm25']['avg_precision_at_10'].values[0],
        data[data['method'] == 'semantic']['avg_precision_at_10'].values[0],
        data[data['method'] == 'hybrid']['avg_precision_at_10'].values[0]
    ]

    bars = ax.barh(methods_list, values, color=[colors['bm25'], colors['semantic'], colors['hybrid']],
                   edgecolor='black', linewidth=1.2)

    # Add value labels and best marker
    best_idx = values.index(max(values))
    for i, (bar, value) in enumerate(zip(bars, values)):
        label = f'{value:.2f}'
        if i == best_idx and value > 0:
            label += '  ★ Best'
        ax.text(value + 0.02, bar.get_y() + bar.get_height()/2, label,
                va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Precision@10', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure2_precision_by_query_type.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 3: Query Execution Time Comparison
# ============================================================================
print("Creating Figure 3: Query Execution Time Comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

methods_list = ['BM25', 'Semantic', 'Hybrid']
times = [
    perf_summary[perf_summary['method'] == 'bm25']['avg_time_ms'].values[0],
    perf_summary[perf_summary['method'] == 'semantic']['avg_time_ms'].values[0],
    perf_summary[perf_summary['method'] == 'hybrid']['avg_time_ms'].values[0]
]

bars = ax.barh(methods_list, times, color=[colors['bm25'], colors['semantic'], colors['hybrid']],
               edgecolor='black', linewidth=1.2)

# Add value labels and fastest marker
for i, (bar, time) in enumerate(zip(bars, times)):
    label = f'{time:.1f}ms'
    if i == 0:  # BM25 is fastest
        label += '  ★ Fastest'
    ax.text(time + 1, bar.get_y() + bar.get_height()/2, label,
            va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Average Response Time (milliseconds)', fontsize=12, fontweight='bold')
ax.set_title('Query Execution Time Comparison', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# Add trade-off text
textstr = 'Speed vs Quality Trade-off:\n'
textstr += '• BM25: 150x faster but fails cross-lingual\n'
textstr += '• Semantic: Good quality, moderate speed\n'
textstr += '• Hybrid: Best quality, slowest (acceptable for search)'
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/figure3_execution_time.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 4: Recall@30 Comparison
# ============================================================================
print("Creating Figure 4: Recall@30 Comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

methods_list = ['BM25', 'Semantic', 'Hybrid']
recall_values = [
    perf_summary[perf_summary['method'] == 'bm25']['recall_at_30'].values[0],
    perf_summary[perf_summary['method'] == 'semantic']['recall_at_30'].values[0],
    perf_summary[perf_summary['method'] == 'hybrid']['recall_at_30'].values[0]
]

bars = ax.barh(methods_list, recall_values, color=[colors['bm25'], colors['semantic'], colors['hybrid']],
               edgecolor='black', linewidth=1.2)

# Add value labels and best marker
best_idx = recall_values.index(max(recall_values))
for i, (bar, value) in enumerate(zip(bars, recall_values)):
    label = f'{value:.2f}'
    if i == best_idx:
        label += '  ★ Best'
    ax.text(value + 0.02, bar.get_y() + bar.get_height()/2, label,
            va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0, 1.0)
ax.set_xlabel('Recall@30 (Proportion of relevant documents found in top 30)', fontsize=11, fontweight='bold')
ax.set_title('Recall@30 Comparison', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# Add note
ax.text(0.5, -0.15, 'Hybrid achieves highest recall by combining signals from all methods.',
        transform=ax.transAxes, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/figure4_recall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 5: Method Performance Summary (Radar Chart)
# ============================================================================
print("Creating Figure 5: Method Performance Summary (Radar Chart)...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Categories
categories = ['Precision@10', 'Recall@30', 'MRR', 'NDCG@10', 'Speed\n(Normalized)']
N = len(categories)

# Normalize speed (invert so higher is better, then normalize to 0-1)
max_time = perf_summary['avg_time_ms'].max()
speed_normalized = [1 - (t / max_time) for t in perf_summary['avg_time_ms']]

# Values for each method
bm25_values = [
    perf_summary[perf_summary['method'] == 'bm25']['precision_at_10'].values[0],
    perf_summary[perf_summary['method'] == 'bm25']['recall_at_30'].values[0],
    perf_summary[perf_summary['method'] == 'bm25']['mrr'].values[0],
    perf_summary[perf_summary['method'] == 'bm25']['ndcg_at_10'].values[0],
    speed_normalized[0]
]

semantic_values = [
    perf_summary[perf_summary['method'] == 'semantic']['precision_at_10'].values[0],
    perf_summary[perf_summary['method'] == 'semantic']['recall_at_30'].values[0],
    perf_summary[perf_summary['method'] == 'semantic']['mrr'].values[0],
    perf_summary[perf_summary['method'] == 'semantic']['ndcg_at_10'].values[0],
    speed_normalized[1]
]

hybrid_values = [
    perf_summary[perf_summary['method'] == 'hybrid']['precision_at_10'].values[0],
    perf_summary[perf_summary['method'] == 'hybrid']['recall_at_30'].values[0],
    perf_summary[perf_summary['method'] == 'hybrid']['mrr'].values[0],
    perf_summary[perf_summary['method'] == 'hybrid']['ndcg_at_10'].values[0],
    speed_normalized[2]
]

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
bm25_values += bm25_values[:1]
semantic_values += semantic_values[:1]
hybrid_values += hybrid_values[:1]
angles += angles[:1]

# Plot
ax.plot(angles, bm25_values, 'o-', linewidth=2, label='BM25', color=colors['bm25'])
ax.fill(angles, bm25_values, alpha=0.15, color=colors['bm25'])

ax.plot(angles, semantic_values, 'o-', linewidth=2, label='Semantic', color=colors['semantic'])
ax.fill(angles, semantic_values, alpha=0.15, color=colors['semantic'])

ax.plot(angles, hybrid_values, 'o-', linewidth=2, label='Hybrid', color=colors['hybrid'])
ax.fill(angles, hybrid_values, alpha=0.15, color=colors['hybrid'])

# Fix axis to go in the right order
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('Method Performance Summary', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

# Add legend text
legend_text = 'BM25: Fast but limited to same-language\n'
legend_text += 'Semantic: Good cross-lingual, moderate speed\n'
legend_text += 'Hybrid: Best overall, combines strengths'
plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/figure5_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 6: Error Type Distribution (Confusion Matrix Style)
# ============================================================================
print("Creating Figure 6: Error Handling Success by Method...")

fig, ax = plt.subplots(figsize=(12, 8))

# Define error categories and counts
error_types = ['Translation\nFailure', 'NER\nMismatch', 'Cross-Script\nAmbiguity',
               'Code-\nSwitching', 'Same-Lang\nExact Match']

# Based on error analysis data
# Success counts out of total test cases
bm25_success = [0, 0, 0, 0, 3]
bm25_total = [4, 4, 3, 1, 3]

semantic_success = [4, 4, 3, 1, 2]
semantic_total = [4, 4, 3, 1, 3]

hybrid_success = [4, 4, 3, 1, 3]
hybrid_total = [4, 4, 3, 1, 3]

# Calculate success rates
bm25_rates = [s/t if t > 0 else 0 for s, t in zip(bm25_success, bm25_total)]
semantic_rates = [s/t if t > 0 else 0 for s, t in zip(semantic_success, semantic_total)]
hybrid_rates = [s/t if t > 0 else 0 for s, t in zip(hybrid_success, hybrid_total)]

x = np.arange(len(error_types))
width = 0.25

bars1 = ax.bar(x - width, bm25_rates, width, label='BM25', color=colors['bm25'], edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, semantic_rates, width, label='Semantic', color=colors['semantic'], edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, hybrid_rates, width, label='Hybrid', color=colors['hybrid'], edgecolor='black', linewidth=1.2)

# Add value labels with symbols
def autolabel_with_symbol(bars, success, total):
    for bar, succ, tot in zip(bars, success, total):
        height = bar.get_height()
        symbol = '✓' if height >= 0.8 else ('◐' if height >= 0.5 else '✗')
        label = f'{succ}/{tot}\n{symbol}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)

autolabel_with_symbol(bars1, bm25_success, bm25_total)
autolabel_with_symbol(bars2, semantic_success, semantic_total)
autolabel_with_symbol(bars3, hybrid_success, hybrid_total)

ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
ax.set_xlabel('Error Category', fontsize=12, fontweight='bold')
ax.set_title('ERROR HANDLING SUCCESS BY METHOD', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(error_types, fontsize=10)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# Add legend
legend_text = 'Legend: ✓ = Handled (>80%)  ◐ = Partial (50-80%)  ✗ = Failed (<50%)'
plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}/figure6_error_handling.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 7: Error Analysis Visualization (Section 5.3)
# ============================================================================
print("Creating Figure 7: Error Analysis Visualization...")

fig, ax = plt.subplots(figsize=(14, 8))

# Error categories with specific success rates
error_categories = ['Translation\nFailure', 'NER\nMismatch', 'Cross-Script\nAmbiguity',
                   'Code-\nSwitching', 'Lexical\nGap']

# Success rates (percentage)
bm25_err_rates = [0, 0, 0, 0, 50]
semantic_err_rates = [100, 100, 100, 100, 60]

y_pos = np.arange(len(error_categories))
height = 0.35

# Create horizontal bars
bars1 = ax.barh(y_pos - height/2, bm25_err_rates, height, label='BM25',
               color=colors['bm25'], edgecolor='black', linewidth=1.2)
bars2 = ax.barh(y_pos + height/2, semantic_err_rates, height, label='Semantic',
               color=colors['semantic'], edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        width = bar.get_width()
        label = f'{int(width)}%'
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, label,
                va='center', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(error_categories, fontsize=11)
ax.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Error Handling Success Rate by Method', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 110)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Add horizontal lines for separation
for i in range(len(error_categories) - 1):
    ax.axhline(y=i + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure7_error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Bonus: Overall Summary Dashboard
# ============================================================================
print("Creating Bonus: Overall Summary Dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall Precision@10
ax1 = fig.add_subplot(gs[0, 0])
methods_list = ['BM25', 'Semantic', 'Hybrid']
precision_vals = perf_summary['precision_at_10'].values
bars = ax1.bar(methods_list, precision_vals, color=[colors['bm25'], colors['semantic'], colors['hybrid']],
              edgecolor='black', linewidth=1.2)
ax1.set_title('Overall Precision@10', fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold')

# Overall Recall@30
ax2 = fig.add_subplot(gs[0, 1])
recall_vals = perf_summary['recall_at_30'].values
bars = ax2.bar(methods_list, recall_vals, color=[colors['bm25'], colors['semantic'], colors['hybrid']],
              edgecolor='black', linewidth=1.2)
ax2.set_title('Overall Recall@30', fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold')

# MRR
ax3 = fig.add_subplot(gs[0, 2])
mrr_vals = perf_summary['mrr'].values
bars = ax3.bar(methods_list, mrr_vals, color=[colors['bm25'], colors['semantic'], colors['hybrid']],
              edgecolor='black', linewidth=1.2)
ax3.set_title('Mean Reciprocal Rank (MRR)', fontweight='bold')
ax3.set_ylim(0, 1.1)
ax3.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold')

# Cross-lingual Performance
ax4 = fig.add_subplot(gs[1, :])
query_type_labels = ['Cross-Lingual', 'Same-Language', 'Code-Switched']
x = np.arange(len(query_type_labels))
width = 0.25

bm25_by_type = [
    query_type[(query_type['query_type'] == 'cross-lingual') & (query_type['method'] == 'bm25')]['avg_precision_at_10'].values[0],
    query_type[(query_type['query_type'] == 'same-language') & (query_type['method'] == 'bm25')]['avg_precision_at_10'].values[0],
    query_type[(query_type['query_type'] == 'code-switched') & (query_type['method'] == 'bm25')]['avg_precision_at_10'].values[0]
]

semantic_by_type = [
    query_type[(query_type['query_type'] == 'cross-lingual') & (query_type['method'] == 'semantic')]['avg_precision_at_10'].values[0],
    query_type[(query_type['query_type'] == 'same-language') & (query_type['method'] == 'semantic')]['avg_precision_at_10'].values[0],
    query_type[(query_type['query_type'] == 'code-switched') & (query_type['method'] == 'semantic')]['avg_precision_at_10'].values[0]
]

hybrid_by_type = [
    query_type[(query_type['query_type'] == 'cross-lingual') & (query_type['method'] == 'hybrid')]['avg_precision_at_10'].values[0],
    query_type[(query_type['query_type'] == 'same-language') & (query_type['method'] == 'hybrid')]['avg_precision_at_10'].values[0],
    query_type[(query_type['query_type'] == 'code-switched') & (query_type['method'] == 'hybrid')]['avg_precision_at_10'].values[0]
]

ax4.bar(x - width, bm25_by_type, width, label='BM25', color=colors['bm25'], edgecolor='black', linewidth=1.2)
ax4.bar(x, semantic_by_type, width, label='Semantic', color=colors['semantic'], edgecolor='black', linewidth=1.2)
ax4.bar(x + width, hybrid_by_type, width, label='Hybrid', color=colors['hybrid'], edgecolor='black', linewidth=1.2)

ax4.set_ylabel('Precision@10', fontweight='bold')
ax4.set_title('Precision@10 by Query Type', fontweight='bold', fontsize=13)
ax4.set_xticks(x)
ax4.set_xticklabels(query_type_labels)
ax4.legend()
ax4.set_ylim(0, 1.1)
ax4.grid(axis='y', alpha=0.3)

# Execution Time
ax5 = fig.add_subplot(gs[2, :2])
times = perf_summary['avg_time_ms'].values
bars = ax5.barh(methods_list, times, color=[colors['bm25'], colors['semantic'], colors['hybrid']],
               edgecolor='black', linewidth=1.2)
ax5.set_xlabel('Average Time (ms)', fontweight='bold')
ax5.set_title('Query Execution Time', fontweight='bold')
ax5.grid(axis='x', alpha=0.3)
for bar in bars:
    width = bar.get_width()
    ax5.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}ms',
            va='center', fontweight='bold')

# Key Findings
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
findings = [
    "KEY FINDINGS:",
    "",
    "✓ Hybrid: Best overall",
    "  (P@10=0.83, R@30=0.88)",
    "",
    "✓ Semantic: Best cross-lingual",
    "  (handles 100% of cases)",
    "",
    "✓ BM25: Fastest but limited",
    "  (0.5ms, same-lang only)",
    "",
    "✓ Trade-off: Quality vs Speed",
    "  (150x speed difference)"
]
findings_text = '\n'.join(findings)
ax6.text(0.1, 0.5, findings_text, fontsize=11, verticalalignment='center',
        family='monospace', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig.suptitle('CLIR System Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ All visualizations generated successfully!")
print(f"✓ Saved to: {output_dir}/")
print(f"\nGenerated files:")
print(f"  1. figure1_precision_comparison.png")
print(f"  2. figure2_precision_by_query_type.png")
print(f"  3. figure3_execution_time.png")
print(f"  4. figure4_recall_comparison.png")
print(f"  5. figure5_radar_chart.png")
print(f"  6. figure6_error_handling.png")
print(f"  7. figure7_error_analysis.png")
print(f"  8. summary_dashboard.png (bonus)")
