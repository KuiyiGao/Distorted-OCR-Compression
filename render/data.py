import json

def prepare_cuad_mrc_data(json_path, tokenizer, target_docs=1, max_length=512, stride=128):
    with open(json_path, 'r', encoding='utf-8') as f:
        cuad = json.load(f)['data'][:target_docs]

    questions, contexts, qa_ids = [], [], []
    for doc in cuad:
        ctx = ""
        for para in doc['paragraphs']:
            ctx += para['context'] + "\n\n"
        for para in doc['paragraphs']:
            for qa in para['qas']:
                questions.append(qa['question'])
                contexts.append(ctx)
                qa_ids.append(qa['id'])

    all_ids, all_masks, all_offsets = [], [], []
    all_seqids, all_sample_idx = [], []

    bs = 100
    for start in range(0, len(questions), bs):
        end = min(start + bs, len(questions))
        tok = tokenizer(
            questions[start:end], contexts[start:end],
            max_length=max_length, stride=stride,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        all_ids.extend(tok["input_ids"])
        all_masks.extend(tok["attention_mask"])
        all_offsets.extend(tok["offset_mapping"])
        for i in range(len(tok["input_ids"])):
            all_seqids.append(tok.sequence_ids(i))
        for idx in tok["overflow_to_sample_mapping"]:
            all_sample_idx.append(idx + start)

    dataset = []
    for i in range(len(all_ids)):
        si = all_sample_idx[i]
        dataset.append({
            'qa_id': qa_ids[si],
            'question': questions[si],
            'context': contexts[si],
            'input_ids': all_ids[i],
            'attention_mask': all_masks[i],
            'sequence_ids': [s if s is not None else -1 for s in all_seqids[i]],
            'offsets': all_offsets[i]
        })
    return dataset
