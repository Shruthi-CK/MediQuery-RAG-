"""Add NDCG breakdowns to Step 5.4 in Fine tuning codex v3.ipynb."""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK = 'Fine tuning codex v3.ipynb'

with open(NOTEBOOK, encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][64]['source'] = [
    "def print_breakdown(details_df, group_col, metric_col, label):\n",
    '    """Print a breakdown of a metric by a grouping column."""\n',
    '    print(f"\\n--- {label} ---")\n',
    "    grouped = details_df.groupby(group_col)[metric_col].agg(['mean', 'count'])\n",
    "    grouped.columns = [metric_col, 'Count']\n",
    "    grouped = grouped.sort_values('Count', ascending=False)\n",
    "    print(grouped.to_string())\n",
    "\n",
    'print("\\n" + "=" * 60)\n',
    'print("FINE-TUNED MODEL BREAKDOWN")\n',
    'print("=" * 60)\n',
    "\n",
    "print_breakdown(ft_details, 'doc_type', 'hit_at_5_reranked', 'Recall@5 (reranked) by Document Type')\n",
    "print_breakdown(ft_details, 'question_type', 'hit_at_5_reranked', 'Recall@5 (reranked) by Question Type')\n",
    "print_breakdown(ft_details, 'coverage_status', 'hit_at_5_reranked', 'Recall@5 (reranked) by Coverage Status')\n",
    "\n",
    "# NDCG breakdowns\n",
    "print_breakdown(ft_details, 'doc_type', 'ndcg_5_reranked', 'NDCG@5 (reranked) by Document Type')\n",
    "print_breakdown(ft_details, 'question_type', 'ndcg_5_reranked', 'NDCG@5 (reranked) by Question Type')\n",
    "print_breakdown(ft_details, 'coverage_status', 'ndcg_5_reranked', 'NDCG@5 (reranked) by Coverage Status')\n",
]
nb['cells'][64]['outputs'] = []
nb['cells'][64]['execution_count'] = None

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Updated {NOTEBOOK}: cell 64 — added NDCG@5 breakdowns')
