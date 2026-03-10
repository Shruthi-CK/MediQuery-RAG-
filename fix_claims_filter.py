"""Add CLAIMS_PROCESSING filter before reranking in 4_Fine_Tuning.ipynb and 5_RAG_Evaluation.ipynb."""
import json


def build_helpers_source(has_evaluate_fn=False):
    """Build the full helpers cell source with CLAIMS_PROCESSING filter."""
    lines = [
        'def retrieve_chunks(query, embed_model, index, all_chunks, chunk_metadata, top_k=20):\n',
        '    """Embed a query and retrieve top-k chunks from a FAISS index."""\n',
        '    query_vec = embed_model.encode(\n',
        '        query, normalize_embeddings=True, convert_to_numpy=True\n',
        "    ).astype('float32').reshape(1, -1)\n",
        '\n',
        '    scores, indices = index.search(query_vec, top_k)\n',
        '\n',
        '    results = []\n',
        '    for rank, idx in enumerate(indices[0]):\n',
        '        if idx == -1:\n',
        '            continue\n',
        '        meta = chunk_metadata[idx]\n',
        '        results.append({\n',
        "            'faiss_score': float(scores[0][rank]),\n",
        "            'text':        all_chunks[idx]['text'],\n",
        "            'title':       meta['title'],\n",
        "            'type':        meta['type'],\n",
        "            'states':      meta.get('states', ['ALL']),\n",
        "            'source_id':   meta['source_id'],\n",
        "            'chunk_idx':   meta['chunk_idx'],\n",
        '            \'chunk_id\':    f"{meta[\'source_id\']}_{meta[\'chunk_idx\']}"\n',
        '        })\n',
        '    return results\n',
        '\n',
        '\n',
        'def filter_by_state(results, state=None):\n',
        '    """Filter retrieved chunks to those covering a specific state."""\n',
        '    if state is None:\n',
        '        return results\n',
        "    return [r for r in results if 'ALL' in r['states'] or state in r['states']]\n",
        '\n',
        '\n',
        'def filter_claims_processing(results):\n',
        '    """Remove CLAIMS_PROCESSING chunks that confuse the reranker."""\n',
        "    return [r for r in results if r['type'] != 'CLAIMS_PROCESSING']\n",
        '\n',
        '\n',
        'def rerank_results(query, results, reranker_model, top_n=5, deduplicate=True):\n',
        '    """Rerank retrieved chunks using a cross-encoder.\n',
        '\n',
        '    Args:\n',
        '        deduplicate: If True, keep only one chunk per source_id (for RAG generation).\n',
        '                     If False, keep all chunks (for retrieval evaluation).\n',
        '    """\n',
        '    if not results:\n',
        '        return []\n',
        '\n',
        "    pairs = [(query, r['text']) for r in results]\n",
        '    scores = reranker_model.predict(pairs)\n',
        '\n',
        '    for i in range(len(results)):\n',
        "        results[i]['rerank_score'] = float(scores[i])\n",
        '\n',
        "    results.sort(key=lambda x: x['rerank_score'], reverse=True)\n",
        '\n',
        '    if not deduplicate:\n',
        '        return results[:top_n]\n',
        '\n',
        '    seen = set()\n',
        '    final = []\n',
        '    for r in results:\n',
        "        if r['source_id'] not in seen:\n",
        '            final.append(r)\n',
        "            seen.add(r['source_id'])\n",
        '        if len(final) == top_n:\n',
        '            break\n',
        '    return final\n',
        '\n',
        '\n',
        'def retrieve_and_rerank(query, embed_model, faiss_index, reranker_model,\n',
        '                        all_chunks, chunk_metadata, state=None,\n',
        '                        top_k_retrieve=20, top_n_rerank=5):\n',
        '    """Full retrieval pipeline: embed -> FAISS -> state filter -> filter claims -> rerank."""\n',
        '    retrieved = retrieve_chunks(query, embed_model, faiss_index,\n',
        '                                all_chunks, chunk_metadata, top_k=top_k_retrieve)\n',
        '    filtered = filter_by_state(retrieved, state)\n',
        '    filtered = filter_claims_processing(filtered)\n',
        '    reranked = rerank_results(query, filtered, reranker_model, top_n=top_n_rerank)\n',
        '    return reranked\n',
    ]

    if has_evaluate_fn:
        lines += [
            '\n',
            '\n',
            'def evaluate_retrieval(embed_model, faiss_index, reranker_model,\n',
            '                       all_chunks, chunk_metadata, eval_df,\n',
            '                       top_k_retrieve=20, top_n_rerank=5):\n',
            '    """Evaluate retrieval: Recall@5/10/20, MRR, before and after reranking."""\n',
            '    results_per_row = []\n',
            '\n',
            "    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc='Evaluating'):\n",
            "        question = row['question']\n",
            "        gold_chunk_id = str(row['chunk_id'])\n",
            '\n',
            '        retrieved = retrieve_chunks(\n',
            '            question, embed_model, faiss_index,\n',
            '            all_chunks, chunk_metadata, top_k=top_k_retrieve\n',
            '        )\n',
            "        retrieved_ids = [r['chunk_id'] for r in retrieved]\n",
            '\n',
            '        # Filter claims processing before reranking\n',
            '        filtered = filter_claims_processing(retrieved)\n',
            '\n',
            '        # Rerank to top-5 (no dedup for fair eval comparison)\n',
            '        reranked = rerank_results(question, filtered, reranker_model, top_n=top_n_rerank, deduplicate=False)\n',
            "        reranked_ids = [r['chunk_id'] for r in reranked]\n",
            '\n',
            "        hit_at_5  = 1 if gold_chunk_id in retrieved_ids[:5]  else 0\n",
            "        hit_at_10 = 1 if gold_chunk_id in retrieved_ids[:10] else 0\n",
            "        hit_at_20 = 1 if gold_chunk_id in retrieved_ids[:20] else 0\n",
            '\n',
            '        if gold_chunk_id in retrieved_ids:\n',
            '            mrr = 1.0 / (retrieved_ids.index(gold_chunk_id) + 1)\n',
            '        else:\n',
            '            mrr = 0.0\n',
            '\n',
            '        hit_at_5_reranked = 1 if gold_chunk_id in reranked_ids else 0\n',
            '        if gold_chunk_id in reranked_ids:\n',
            '            mrr_reranked = 1.0 / (reranked_ids.index(gold_chunk_id) + 1)\n',
            '        else:\n',
            '            mrr_reranked = 0.0\n',
            '\n',
            '        results_per_row.append({\n',
            "            'qa_id':              row.get('qa_id', ''),\n",
            "            'question_type':      row.get('question_type', ''),\n",
            "            'doc_type':           row.get('doc_type', ''),\n",
            "            'coverage_status':    row.get('coverage_status', ''),\n",
            "            'hit_at_5':           hit_at_5,\n",
            "            'hit_at_10':          hit_at_10,\n",
            "            'hit_at_20':          hit_at_20,\n",
            "            'mrr':                mrr,\n",
            "            'hit_at_5_reranked':  hit_at_5_reranked,\n",
            "            'mrr_reranked':       mrr_reranked,\n",
            '        })\n',
            '\n',
            '    results_df = pd.DataFrame(results_per_row)\n',
            '    summary = {\n',
            "        'Recall@5 (FAISS)':    results_df['hit_at_5'].mean(),\n",
            "        'Recall@10 (FAISS)':   results_df['hit_at_10'].mean(),\n",
            "        'Recall@20 (FAISS)':   results_df['hit_at_20'].mean(),\n",
            "        'MRR (FAISS)':         results_df['mrr'].mean(),\n",
            "        'Recall@5 (reranked)': results_df['hit_at_5_reranked'].mean(),\n",
            "        'MRR (reranked)':      results_df['mrr_reranked'].mean(),\n",
            '    }\n',
            '    return summary, results_df\n',
        ]

    lines += ['\n', '\n', "print('Helper functions defined.')"]
    return lines


def build_eval_fn_source():
    """Build standalone evaluate_retrieval cell with claims filter."""
    return [
        'def evaluate_retrieval(embed_model, faiss_index, reranker_model,\n',
        '                       all_chunks, chunk_metadata, eval_df,\n',
        '                       top_k_retrieve=20, top_n_rerank=5):\n',
        '    """\n',
        '    Evaluate retrieval performance on the eval set.\n',
        '\n',
        '    Returns a dict with Recall@5, Recall@10, Recall@20, MRR,\n',
        '    and per-row details for breakdown analysis.\n',
        '    """\n',
        '    results_per_row = []\n',
        '\n',
        '    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):\n',
        "        question = row['question']\n",
        "        gold_chunk_id = str(row['chunk_id'])\n",
        '\n',
        '        # Retrieve top-20\n',
        '        retrieved = retrieve_chunks(\n',
        '            question, embed_model, faiss_index,\n',
        '            all_chunks, chunk_metadata, top_k=top_k_retrieve\n',
        '        )\n',
        "        retrieved_ids = [r['chunk_id'] for r in retrieved]\n",
        '\n',
        '        # Filter claims processing before reranking\n',
        '        filtered = filter_claims_processing(retrieved)\n',
        '\n',
        '        # Rerank to top-5 (no dedup for fair eval comparison)\n',
        '        reranked = rerank_results(question, filtered, reranker_model, top_n=top_n_rerank, deduplicate=False)\n',
        "        reranked_ids = [r['chunk_id'] for r in reranked]\n",
        '\n',
        '        # Compute per-query metrics\n',
        '        # Recall@k (before reranking)\n',
        "        hit_at_5  = 1 if gold_chunk_id in retrieved_ids[:5]  else 0\n",
        "        hit_at_10 = 1 if gold_chunk_id in retrieved_ids[:10] else 0\n",
        "        hit_at_20 = 1 if gold_chunk_id in retrieved_ids[:20] else 0\n",
        '\n',
        '        # MRR (before reranking)\n',
        '        if gold_chunk_id in retrieved_ids:\n',
        '            mrr = 1.0 / (retrieved_ids.index(gold_chunk_id) + 1)\n',
        '        else:\n',
        '            mrr = 0.0\n',
        '\n',
        '        # Recall@5 after reranking\n',
        '        hit_at_5_reranked = 1 if gold_chunk_id in reranked_ids else 0\n',
        '\n',
        '        # MRR after reranking\n',
        '        if gold_chunk_id in reranked_ids:\n',
        '            mrr_reranked = 1.0 / (reranked_ids.index(gold_chunk_id) + 1)\n',
        '        else:\n',
        '            mrr_reranked = 0.0\n',
        '\n',
        '        results_per_row.append({\n',
        "            'qa_id':              row.get('qa_id', ''),\n",
        "            'question_type':      row.get('question_type', ''),\n",
        "            'doc_type':           row.get('doc_type', ''),\n",
        "            'coverage_status':    row.get('coverage_status', ''),\n",
        "            'hit_at_5':           hit_at_5,\n",
        "            'hit_at_10':          hit_at_10,\n",
        "            'hit_at_20':          hit_at_20,\n",
        "            'mrr':                mrr,\n",
        "            'hit_at_5_reranked':  hit_at_5_reranked,\n",
        "            'mrr_reranked':       mrr_reranked,\n",
        '        })\n',
        '\n',
        '    results_df = pd.DataFrame(results_per_row)\n',
        '\n',
        '    summary = {\n',
        "        'Recall@5 (FAISS)':    results_df['hit_at_5'].mean(),\n",
        "        'Recall@10 (FAISS)':   results_df['hit_at_10'].mean(),\n",
        "        'Recall@20 (FAISS)':   results_df['hit_at_20'].mean(),\n",
        "        'MRR (FAISS)':         results_df['mrr'].mean(),\n",
        "        'Recall@5 (reranked)': results_df['hit_at_5_reranked'].mean(),\n",
        "        'MRR (reranked)':      results_df['mrr_reranked'].mean(),\n",
        '    }\n',
        '\n',
        '    return summary, results_df\n',
        '\n',
        '\n',
        'print("Evaluation function defined.")',
    ]


def clear_cell(cell):
    cell['outputs'] = []
    cell['execution_count'] = None


# ── Fix 4_Fine_Tuning.ipynb: cell 11 (helpers) and cell 55 (evaluate_retrieval) ──
nb_file = '4_Fine_Tuning.ipynb'
with open(nb_file, encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][11]['source'] = build_helpers_source(has_evaluate_fn=False)
clear_cell(nb['cells'][11])

nb['cells'][55]['source'] = build_eval_fn_source()
clear_cell(nb['cells'][55])

with open(nb_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f'Fixed {nb_file}: cells 11 and 55')


# ── Fix 5_RAG_Evaluation.ipynb: cell 9 (helpers + evaluate_retrieval combined) ──
nb_file = '5_RAG_Evaluation.ipynb'
with open(nb_file, encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][9]['source'] = build_helpers_source(has_evaluate_fn=True)
clear_cell(nb['cells'][9])

with open(nb_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f'Fixed {nb_file}: cell 9')
