"""Fix 3: Add manual chapter pairs (Benefit Policy, Claims Processing) to training data.

Inserts two new cells after cell 21 (train/val split) in 4_Fine_Tuning.ipynb:
  - Cell 22 (new markdown): explanation
  - Cell 23 (new code): load eval_df, extract manual chapter pairs, combine with train_set

Then updates the existing InputExample cell (now shifted to cell 25) to use combined_train_set.
"""
import json

NOTEBOOK = '4_Fine_Tuning.ipynb'

with open(NOTEBOOK, encoding='utf-8') as f:
    nb = json.load(f)

# ── Insert after cell 21 (train/val split) ──
new_markdown = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### Step 1.5 \u2014 Add manual chapter pairs to training data\n',
        '\n',
        'The 24K training pairs come exclusively from NCDs and LCDs. The fine-tuned model becomes '
        'blind to **Claims Processing Manual** and **Benefit Policy Manual** chunks because it never '
        'sees them during training. We pull manual chapter queries from the 500-item eval CSV and add '
        'them as extra training pairs so the model learns what those chunks look like too.',
    ],
}

new_code = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [
        '# Load the eval CSV to get manual chapter questions\n',
        'eval_df_early = pd.read_csv(EVAL_CSV)\n',
        '\n',
        'manual_extra_pairs = []\n',
        '\n',
        'for _, row in eval_df_early.iterrows():\n',
        "    if row['doc_type'] not in ['Medicare Claims Processing Manual',\n",
        "                                'Medicare Benefit Policy Manual']:\n",
        '        continue\n',
        '\n',
        "    chunk_id = str(row['chunk_id'])\n",
        '    if chunk_id not in chunk_id_to_idx:\n',
        '        continue\n',
        '\n',
        '    idx = chunk_id_to_idx[chunk_id]\n',
        "    positive_text = all_chunks[idx]['text']\n",
        '\n',
        '    # Mine a hard negative using the pre-trained FAISS index\n',
        '    query_vec = pretrained_embed_model.encode(\n',
        "        row['question'], normalize_embeddings=True, convert_to_numpy=True\n",
        "    ).astype('float32').reshape(1, -1)\n",
        '\n',
        '    _, neg_indices = pretrained_index.search(query_vec, 20)\n',
        '    negative_text = None\n',
        "    gold_source = chunk_id.rsplit('_', 1)[0]\n",
        '\n',
        '    for neg_idx in neg_indices[0]:\n',
        '        if neg_idx == -1:\n',
        '            continue\n',
        "        if str(chunk_metadata[neg_idx]['source_id']) != gold_source:\n",
        "            negative_text = all_chunks[neg_idx]['text']\n",
        '            break\n',
        '\n',
        '    if negative_text is None:\n',
        '        continue\n',
        '\n',
        '    manual_extra_pairs.append({\n',
        "        'query':         row['question'],\n",
        "        'positive':      positive_text,\n",
        "        'negative':      negative_text,\n",
        "        'source_id':     gold_source,\n",
        "        'query_type':    row['question_type'],\n",
        "        'gold_chunk_id': chunk_id,\n",
        '    })\n',
        '\n',
        'print(f"Extra manual chapter pairs added: {len(manual_extra_pairs)}")\n',
        '\n',
        '# Combine with original train_set\n',
        'combined_train_set = train_set + manual_extra_pairs\n',
        'print(f"Combined training set: {len(combined_train_set):,}")',
    ],
}

# Insert at position 22 (after cell 21)
nb['cells'].insert(22, new_markdown)
nb['cells'].insert(23, new_code)

# ── Now the old cell 22 (InputExamples markdown) is at index 24 ──
# ── The old cell 23 (InputExamples code) is at index 25 ──
# Update cell 25 to use combined_train_set instead of train_set
nb['cells'][25]['source'] = [
    'from sentence_transformers import InputExample\n',
    '\n',
    '# Triplets: (query, positive, hard_negative)\n',
    'biencoder_train_examples = [\n',
    "    InputExample(texts=[p['query'], p['positive'], p['negative']])\n",
    '    for p in combined_train_set\n',
    ']\n',
    '\n',
    'print(f"Bi-encoder training examples: {len(biencoder_train_examples):,}")\n',
    'print(f"  Sample query:    {biencoder_train_examples[0].texts[0][:120]}...")\n',
    'print(f"  Sample positive: {biencoder_train_examples[0].texts[1][:120]}...")\n',
    'print(f"  Sample negative: {biencoder_train_examples[0].texts[2][:120]}...")',
]
nb['cells'][25]['outputs'] = []
nb['cells'][25]['execution_count'] = None

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f'Fixed {NOTEBOOK}: inserted cells 22-23, updated cell 25')
print(f'Total cells: {len(nb["cells"])}')
