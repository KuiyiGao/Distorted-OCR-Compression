import json
import torch
import random
from torch.utils.data import Dataset

def prepare_cuad_mrc_data(json_path, tokenizer, max_length=512, stride=None, n_docs=None):
    print(f"Loading MRC data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        cuad_data = json.load(f)['data']

    # Q3: stride derived from max_length if not given; n_docs replaces the 50% hardcode
    if stride is None:
        stride = max_length // 4
    if n_docs is not None:
        cuad_data = cuad_data[:n_docs]
    print(f"Using {len(cuad_data)} documents (stride={stride})")

    questions, contexts, char_labels_list, qa_ids = [], [], [], []
    for doc_idx, document in enumerate(cuad_data):
        for para_idx, paragraph in enumerate(document['paragraphs']):
            context = paragraph['context']
            for qa_idx, qa in enumerate(paragraph['qas']):
                question = qa['question']
                is_impossible = qa.get('is_impossible', False)
                # ACCELERATION LOGIC: Drop 95% of completely empty negative context questions!
                if is_impossible and random.random() > 0.05:
                    continue
                
                qa_id = qa.get('id', f"{doc_idx}_{para_idx}_{qa_idx}")
                
                char_labels = [0] * (len(context) + 1) 
                if not is_impossible:
                    for ans in qa.get('answers', []):
                        start = ans.get('answer_start', 0)
                        end = start + len(ans.get('text', ''))
                        for i in range(start, end):
                            if i < len(char_labels):
                                char_labels[i] = 1
                
                questions.append(question)
                contexts.append(context)
                char_labels_list.append(char_labels)
                qa_ids.append(qa_id)
                
    print(f"Batch tokenizing {len(questions)} Query-Context pairs via Chunked Rust backed mapping (Safe Memory Mode)...")
    
    input_ids_batch = []
    attention_mask_batch = []
    offset_mapping_batch = []
    overflow_to_sample_mapping = []
    sequence_ids_batch = []
    
    batch_size = 100
    for chunk_start in range(0, len(questions), batch_size):
        chunk_end = min(chunk_start + batch_size, len(questions))
        tokenized = tokenizer(
            questions[chunk_start:chunk_end], 
            contexts[chunk_start:chunk_end],
            max_length=max_length, stride=stride, truncation="only_second",
            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length"
        )
        
        input_ids_batch.extend(tokenized["input_ids"])
        attention_mask_batch.extend(tokenized["attention_mask"])
        offset_mapping_batch.extend(tokenized["offset_mapping"])
        
        for i in range(len(tokenized["input_ids"])):
            sequence_ids_batch.append(tokenized.sequence_ids(i))
            
        for idx in tokenized["overflow_to_sample_mapping"]:
            overflow_to_sample_mapping.append(idx + chunk_start)
            
        print(f"--> Tokenized ({chunk_end}/{len(questions)}) original document pairs successfully...")
    
    print("Mapping native arrays mapping to global indices...")
    dataset = []
    
    for i in range(len(input_ids_batch)):
        sample_idx = overflow_to_sample_mapping[i]
        qa_id = qa_ids[sample_idx]
        question = questions[sample_idx]
        context = contexts[sample_idx]
        char_labels = char_labels_list[sample_idx]
        
        sequence_ids = sequence_ids_batch[i]
        offsets = offset_mapping_batch[i]
        labels = []
        
        for idx, (start_char, end_char) in enumerate(offsets):
            if sequence_ids[idx] != 1:
                labels.append(-100) 
            else:
                if start_char < len(char_labels) and end_char <= len(char_labels):
                    if sum(char_labels[start_char:end_char]) > 0:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    labels.append(0)
        
        dataset.append({
            'qa_id': qa_id,
            'question': question,
            'context': context,
            'input_ids': input_ids_batch[i],
            'attention_mask': attention_mask_batch[i],
            'labels': labels,
            'sequence_ids': [s if s is not None else -1 for s in sequence_ids],
            'offsets': offsets
        })
                    
    print(f"Successfully bridged into {len(dataset)} chunk pairs globally.")
    return dataset

class CUADQADataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512, n_docs=None):
        self.data = prepare_cuad_mrc_data(json_path, tokenizer, max_length=max_length, n_docs=n_docs)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]