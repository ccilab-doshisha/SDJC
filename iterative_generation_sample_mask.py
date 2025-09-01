import spacy
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import os
import random

# --- 1. Setup: Load models, tokenizer, and data ---

print("Loading SpaCy model 'ja_ginza'...")
nlp = spacy.load('ja_ginza')

# --- Load Sentences ---
sentences = []
corpus_path = '/workspace/clinic_corpus.txt'
try:
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sentences.append(line.strip())
    print(f"Successfully loaded {len(sentences)} sentences from {corpus_path}")
    if not sentences:
        print("Error: The corpus file is empty. Exiting.")
        exit()
except FileNotFoundError:
    print(f"FATAL ERROR: Corpus file not found at '{corpus_path}'.")
    exit()

# --- Load T5 Model ---
T5_PATH = '/workspace/trained_model/T5-clinic-from-processed-data/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

print(f"Loading T5 model and tokenizer from '{T5_PATH}'...")
t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)
t5_mlm.eval()

# --- 2. Helper Function ---

def reconstruct_sentence(masked_text: str, generated_output_tokens: torch.Tensor) -> str:
    filler_text = t5_tokenizer.decode(generated_output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    mask_token = '<extra_id_0>'
    if mask_token in masked_text:
        return masked_text.replace(mask_token, filler_text, 1)
    return masked_text

# --- 3. Main Generation Logic with Configurable Sampling ---

# ####################################################################
# ### ABLATION STUDY PARAMETERS: Modify these for each experiment ###
# ####################################################################
PROPORTION_TO_MODIFY = 0.25
MIN_TO_MODIFY = 1
MAX_TO_MODIFY = 99 # Safety cap. Set high (e.g., 99) to disable.
# ####################################################################

file_suffix = f"prop{PROPORTION_TO_MODIFY}_min{MIN_TO_MODIFY}_max{MAX_TO_MODIFY}"
file_out_top2 = f'/workspace/clinic_data_generation/gen_iter_{file_suffix}_top2.txt'
file_out_top3 = f'/workspace/clinic_data_generation/gen_iter_{file_suffix}_top3.txt'
file_out_top4 = f'/workspace/clinic_data_generation/gen_iter_{file_suffix}_top4.txt'

output_directory = os.path.dirname(file_out_top2)
os.makedirs(output_directory, exist_ok=True)
print(f"Output files will be saved to: {output_directory}")
print(f"File suffix for this run: {file_suffix}")


# --- 4. Main Loop with Generation, Saving, and Printing ---

try:
    with open(file_out_top2, 'w', encoding='utf-8') as f_top2, \
         open(file_out_top3, 'w', encoding='utf-8') as f_top3, \
         open(file_out_top4, 'w', encoding='utf-8') as f_top4:

        for anchor_sentence in tqdm(sentences, desc=f"Processing & Printing ({file_suffix})"):
            doc = nlp(anchor_sentence)
            all_noun_chunks = [nc.text for nc in doc.noun_chunks]
            
            chunks_to_modify = []
            if all_noun_chunks:
                num_chunks = len(all_noun_chunks)
                num_by_proportion = int(num_chunks * PROPORTION_TO_MODIFY)
                num_to_modify = max(MIN_TO_MODIFY, num_by_proportion)
                num_to_modify = min(MAX_TO_MODIFY, num_to_modify)
                num_to_modify = min(num_to_modify, num_chunks)
                chunks_to_modify = random.sample(all_noun_chunks, num_to_modify)
            
            sentence_paths = {'top2': anchor_sentence, 'top3': anchor_sentence, 'top4': anchor_sentence}
            if chunks_to_modify:
                for chunk_text in chunks_to_modify:
                    next_sentence_paths = {}
                    for path_key, current_sentence in sentence_paths.items():
                        masked_text = current_sentence.replace(chunk_text, '<extra_id_0>', 1)
                        if '<extra_id_0>' not in masked_text:
                            next_sentence_paths[path_key] = current_sentence
                            continue

                        encoded = t5_tokenizer(masked_text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
                        input_ids = encoded['input_ids'].to(DEVICE)
                        
                        with torch.no_grad():
                            outputs = t5_mlm.generate(
                                input_ids=input_ids, 
                                num_beams=200, 
                                num_return_sequences=4,
                                max_length=15
                            )
                        
                        rank_index = int(path_key.replace('top', '')) - 1 
                        if len(outputs) > rank_index:
                            chosen_output_tokens = outputs[rank_index]
                            next_sentence_paths[path_key] = reconstruct_sentence(masked_text, chosen_output_tokens)
                        else:
                            next_sentence_paths[path_key] = current_sentence
                    sentence_paths = next_sentence_paths

            final_top2 = sentence_paths['top2']
            final_top3 = sentence_paths['top3']
            final_top4 = sentence_paths['top4']
            
            # --- START: MODIFIED BLOCK for Saving, Printing, and Flushing ---

            # 1. Write results to the files
            f_top2.write(f"{anchor_sentence.strip()}\t{final_top2.strip()}\n")
            f_top3.write(f"{anchor_sentence.strip()}\t{final_top3.strip()}\n")
            f_top4.write(f"{anchor_sentence.strip()}\t{final_top4.strip()}\n")

            # 2. Print results to the console for immediate checking
            # The initial '\n' prevents the tqdm bar from messing up the first line
            print("\n" + "="*80)
            print(f"ANCHOR: {anchor_sentence.strip()}")
            if chunks_to_modify:
                print(f"MODIFIED CHUNKS: {chunks_to_modify}")
            else:
                print("MODIFIED CHUNKS: [] (No nouns found)")
            print("-" * 40)
            print(f"  -> GEN TOP 2: {final_top2.strip()}")
            print(f"  -> GEN TOP 3: {final_top3.strip()}")
            print(f"  -> GEN TOP 4: {final_top4.strip()}")
            print("="*80)

            # 3. Flush the file buffers to disk to save progress
            f_top2.flush()
            f_top3.flush()
            f_top4.flush()
            
            # --- END: MODIFIED BLOCK ---

    print(f"\nProcessing complete. Results saved.")

except Exception as e:
    print(f"\nAn error occurred during processing: {e}")