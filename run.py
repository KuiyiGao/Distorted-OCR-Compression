import os
import torch
import collections
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from render.data import CUADQADataset
from render.attention import ImbalancedMRCTrainer, extract_attention_saliency
from render.utils import stitch_and_smooth_saliency
from render.render import render_tsvr_image, visualize_single_attention

def main():
    print("=== Starting Q/A LegalBERT Full-Contract TSVR Pipeline ===")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    json_path = "render/data/CUADv1.json"
    
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    print("Parsing roughly 50% of CUAD corpus simultaneously leveraging native arrays...")
    dataset = CUADQADataset(json_path, tokenizer, max_length=512)
    
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    print(f"Loading underlying Model: {model_name} (Imbalanced CrossEntropy 2-class mapping)")
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    print("Accelerating Procedure: Freezing Lower Encoder Matrices...")
    for name, param in model.bert.named_parameters():
        if not name.startswith("encoder.layer.10") and not name.startswith("encoder.layer.11"):
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="./tsvr_model_checkpoints",
        evaluation_strategy="steps",  
        eval_steps=100, 
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-4, 
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    # Note: Hugging Face Trainer + PyTorch dataloader dynamically collates raw lists to tensors!
    trainer = ImbalancedMRCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Stage 1a: Commencing Accelerated Training hook to track BEST Eval_LOSS Model...")
    trainer.train()
    
    print("Correlating Output predictions leveraging Best Model weights checkpointed...")
    qa_grouped_data = collections.defaultdict(list)
    for sample in dataset.data:
        qa_grouped_data[sample['qa_id']].append(sample)
        
    target_qas = []
    # Identify 1 positive sample to render specifically highlighting attention efficacy
    for qa_id, chunks in qa_grouped_data.items():
        if any(1 in c['labels'] for c in chunks): # Check via list element matching
            target_qas.append((qa_id, chunks))
            break
            
    output_dir = "cuad_tsvr_results_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (qa_id, chunks) in enumerate(target_qas):
        question = chunks[0]["question"]
        full_context = chunks[0]["context"]
        
        chunks_extracted = []
        for chunk in chunks:
            # We construct Tensors securely right at the point of Model extraction
            input_ids = torch.tensor(chunk["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(chunk["attention_mask"], dtype=torch.long)
            
            probs = extract_attention_saliency(trainer.model, input_ids, attention_mask, device)
            
            chunks_extracted.append({
                'probs': probs,
                'offsets': chunk['offsets'],
                'sequence_ids': chunk['sequence_ids']
            })
            
        print(f"Stitching all {len(chunks)} overlapping windows covering the entire contract smoothly for Best Checkpoint Model...")
        word_weights = stitch_and_smooth_saliency(full_context, chunks_extracted, sigma=2.0)
        
        if not word_weights:
            continue
            
        att_path = os.path.join(output_dir, f"sample_best_model_attention.png")
        visualize_single_attention(word_weights, att_path)
        
        tsvr_img = render_tsvr_image(word_weights, image_width=800, max_font_size=50)
        tsvr_path = os.path.join(output_dir, f"sample_best_model_tsvr.png")
        tsvr_img.save(tsvr_path)
        
        with open(os.path.join(output_dir, f"sample_best_model_meta.txt"), "w") as f:
            f.write(f"Question: {question}\n")
            f.write(f"QA ID: {qa_id}\n")
            f.write(f"Total Combined Window Chunks Overlapped: {len(chunks)}\n")
            high_att = [w[0] for w in word_weights if w[1] > 0.8]
            f.write(f"High Saliency Terms Detected by BEST Eval Model: {high_att}\n")
            
    print(f"\nSuccessfully generated Best Model single evaluation TSVR into '{output_dir}/'.")

if __name__ == "__main__":
    main()
