"""Fix the bi-encoder IR evaluator to use the full corpus (all 8563 chunks) instead of just val positives."""
import json

NOTEBOOK = '4_Fine_Tuning.ipynb'

with open(NOTEBOOK, encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 27 (markdown): update description ──
nb['cells'][27]['source'] = [
    '### Step 2.2 \u2014 Set up IR evaluator against the full corpus\n',
    '\n',
    'The evaluator must search the full chunk corpus (all 8,563 chunks), not just the validation positives. '
    'Using only val positives as the corpus makes evaluation trivially easy \u2014 the model only has to pick '
    'the right answer from ~1,200 candidates instead of 8,563.',
]

# ── Cell 28 (code): rebuild evaluator with full corpus ──
nb['cells'][28]['source'] = [
    'from sentence_transformers.evaluation import InformationRetrievalEvaluator\n',
    '\n',
    '# Build the full corpus from all_chunks \u2014 every chunk the FAISS index contains.\n',
    '# Previously this only had the ~1202 val positives, making eval unrealistically easy.\n',
    "full_ir_corpus = {str(i): chunk['text'] for i, chunk in enumerate(all_chunks)}\n",
    '\n',
    '# Queries and relevant docs stay the same \u2014 we still use val_set queries,\n',
    '# but now each query has to find its gold chunk among all 8563, not just ~1202.\n',
    "ir_queries       = {str(i): p['query'] for i, p in enumerate(val_set)}\n",
    "ir_relevant_docs = {str(i): {str(chunk_id_to_idx[p['gold_chunk_id']])}\n",
    '                    for i, p in enumerate(val_set)}\n',
    '\n',
    'biencoder_evaluator = InformationRetrievalEvaluator(\n',
    '    ir_queries,\n',
    '    full_ir_corpus,\n',
    '    ir_relevant_docs,\n',
    "    name='mediquery-biencoder-val',\n",
    '    show_progress_bar=True\n',
    ')\n',
    '\n',
    'print(f"IR Evaluator: {len(ir_queries)} queries against {len(full_ir_corpus)} corpus docs")',
]
nb['cells'][28]['outputs'] = []
nb['cells'][28]['execution_count'] = None

with open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f'Fixed {NOTEBOOK}: cells 27 and 28')
