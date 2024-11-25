
import os
import amrlib
from amrlib import load_stog_model
import spacy
import spacy
from datasets import load_dataset
from datasets import concatenate_datasets
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from typing import Dict, List
from functools import partial
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import DatasetDict
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from typing import Dict, List
from functools import partial
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import DatasetDict
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from typing import Dict, List
from functools import partial
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import DatasetDict
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
# Enable W&B dry run mode
os.environ["WANDB_MODE"] = "dryrun"
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
# '''
# -----------------------------------------Penmen-graph-generation----------------------------------
# '''



# amrlib.setup_spacy_extension()
# def create_penmen_graph(penmen_text,nlp):
#   doc = nlp(penmen_text)
#   graphs = doc._.to_amr()
#   joined_text = " ".join(graphs[0].split('\n')[1:]).strip()
#   return joined_text


# '''
# ----------------------------------model ---load--------------------------
# please change the model path once you upload the model in the directory
# '''




# stog_model_dir = '/kaggle/input/bart-large/model_parse_xfm_bart_large-v0_1_0'
# gtos_model_dir = '/kaggle/input/model-t5-2000/model_generate_t5wtense-v0_1_0'
# stog = load_stog_model(model_dir=stog_model_dir)
# gtos = amrlib.load_gtos_model(model_dir=gtos_model_dir)

'''
------------------------loading_webnlg_data------------------------------
'''



dataset = load_dataset('web_nlg', 'release_v3.0_en')

train_data = dataset['train']
validation_data = dataset['dev']
test_data = dataset['test']


# Combine all datasets into one
combined_data = concatenate_datasets([dataset['train'], dataset['dev'], dataset['test']])

# Shuffle the combined dataset
shuffled_data = combined_data.shuffle(seed=42)

# Define new split ratios
train_ratio = 0.98
validation_ratio = 0.01
test_ratio = 0.01

# Compute split indices
total_samples = len(shuffled_data)
train_end = int(total_samples * train_ratio)
validation_end = train_end + int(total_samples * validation_ratio)

# Split the shuffled data into Dataset objects
train_data = shuffled_data.select(range(train_end))
validation_data = shuffled_data.select(range(train_end, validation_end))
test_data = shuffled_data.select(range(validation_end, total_samples))

print(f"New training samples: {len(train_data)}")
print(f"New validation samples: {len(validation_data)}")
print(f"New test samples: {len(test_data)}")


def graph_generator(data):
    graph_list = []
    for i in range(len(data)):
        total_graph = ""
        if len(data[i]['modified_triple_sets']['mtriple_set'][0])==1:
            for item in data[i]['modified_triple_sets']['mtriple_set'][0]:
                split_item = item.split("|")
                formatted_triple = f"<H> {split_item[0].strip()} </H> <R> {split_item[1].strip()} </R> <T> {split_item[2].strip()} </T>"
                total_graph += formatted_triple 

            graph_list.append(total_graph)
        
        
        else:    
            for item in data[i]['modified_triple_sets']['mtriple_set'][0]:
                split_item = item.split("|")
                formatted_triple = f"<H> {split_item[0].strip()} </H> <R> {split_item[1].strip()} </R> <T> {split_item[2].strip()} </T>"
                total_graph += formatted_triple + "  [SEP]  "
            
            graph_list.append(total_graph)
    return graph_list
## Grpah generation       
graph_train = graph_generator(train_data)
graph_validation = graph_generator(validation_data)
graph_test = graph_generator(test_data)


def output_text(data):
    text_list = []
    data_size = len(data)
    for i in range(data_size):
        if (len(data[i]['lex']['text'])) < 1:
            text_list.append("not_found")
        else:
            
            sentence = data[i]['lex']['text'][0]
            text_list.append(sentence)
        
        '''
        sentence = data[i]['lex']['text']
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)

        parts = sentence.split('.')
        new_sentence = ' '.join(part.strip() for part in parts if part.strip())
        text_list.append(new_sentence)
        '''
    return text_list
## Text generation
text_train = output_text(train_data)
text_validation = output_text(validation_data)
text_test =  output_text(test_data)
data_webnlg_train = pd.DataFrame({
    "amr_graph": graph_train,
    "text": text_train
})

data_webnlg_validation = pd.DataFrame({
    "amr_graph": graph_validation,
    "text": text_validation
})

data_webnlg_test = pd.DataFrame({
    "amr_graph": graph_test,
    "text": text_test
})

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Convert the DataFrame to a Hugging Face dataset
# dataset = Dataset.from_pandas(data_webnlg_train)



# Convert the DataFrame to a Hugging Face dataset
dataset_train = Dataset.from_pandas(data_webnlg_train)
dataset_validation = Dataset.from_pandas(data_webnlg_validation)
dataset_test = Dataset.from_pandas(data_webnlg_test)
# Select a smaller subset if needed
small_dataset_train = dataset_train.select([i for i in range(len(train_data))])
small_dataset_validation = dataset_validation.select([i for i in range(len(validation_data))])
small_dataset_test = dataset_test.select([i for i in range(200)])





def _add_text(rec):
    instruction = rec["amr_graph"]  # Use amr_graph as instruction
    response = rec["text"]  # Use text as response
    
    # Check if both exist; raise error if not
    if not instruction:
        raise ValueError(f"Expected an instruction (amr_graph) in: {rec}")
    if not response:
        raise ValueError(f"Expected a response (text) in: {rec}")
    
    # Create prompt, answer, and combined text
    rec["prompt"] = prompt_template.format(instruction=instruction)
    rec["answer"] = answer_template.format(response=response)
    rec["text"] = rec["prompt"] + rec["answer"]
    return rec

# Apply the function to the dataset
small_dataset_train = small_dataset_train.map(_add_text)
small_dataset_validation = small_dataset_validation.map(_add_text)
small_dataset_test = small_dataset_test.map(_add_text)
# # Print the first item to check
# print(small_dataset_test[0])


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Set the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 256

# Function to generate token embeddings from the text part of the batch
def _preprocess_batch(batch: Dict[str, List]):  
    model_inputs = tokenizer(batch["text"], max_length=MAX_LENGTH, truncation=True, padding='max_length')    
    model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
    return model_inputs

_preprocessing_function = partial(_preprocess_batch)



# Define the split ratios
# train_test_split = small_dataset.train_test_split(test_size=0.2)  # Split off 20% as test set
# train_valid_split = train_test_split['train'].train_test_split(test_size=0.1)  # From train, split 10% as validation

# Combine splits into a DatasetDict
dataset_dict = DatasetDict({
    'train': small_dataset_train,
    'validation': small_dataset_validation,
    'test': small_dataset_test
})

# Print the size of each split to verify
# print(f"Train set size: {len(dataset_dict['train'])}")
# print(f"Validation set size: {len(dataset_dict['validation'])}")
# print(f"Test set size: {len(dataset_dict['test'])}")

# Example check for first item in each split
# print("Sample from train:", dataset_dict['train'][0])
# print("Sample from validation:", dataset_dict['validation'][0])
# print("Sample from test:", dataset_dict['test'][0])


# Apply the preprocessing function to each batch in the dataset
encoded_train_dataset = dataset_dict['train'].map(
    _preprocessing_function,
    batched=True,
    remove_columns=["amr_graph", "text", "prompt", "answer"],
)

encoded_validation_dataset = dataset_dict['validation'].map(
    _preprocessing_function,
    batched=True,
    remove_columns=["amr_graph", "text", "prompt", "answer"],
)

encoded_test_dataset = dataset_dict['test'].map(
    _preprocessing_function,
    batched=True,
    remove_columns=["amr_graph", "text", "prompt", "answer"],
)
processed_train_dataset = encoded_train_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)
processed_validation_dataset = encoded_validation_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)
processed_test_dataset = encoded_test_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)



# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Set the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 256

# Function to generate token embeddings from the text part of the batch
def _preprocess_batch(batch: Dict[str, List]):  
    model_inputs = tokenizer(batch["text"], max_length=MAX_LENGTH, truncation=True, padding='max_length')    
    model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
    return model_inputs

_preprocessing_function = partial(_preprocess_batch)



# Define the split ratios
# train_test_split = small_dataset.train_test_split(test_size=0.2)  # Split off 20% as test set
# train_valid_split = train_test_split['train'].train_test_split(test_size=0.1)  # From train, split 10% as validation

# Combine splits into a DatasetDict
dataset_dict = DatasetDict({
    'train': small_dataset_train,
    'validation': small_dataset_validation,
    'test': small_dataset_test
})

encoded_train_dataset = dataset_dict['train'].map(
    _preprocessing_function,
    batched=True,
    remove_columns=["amr_graph", "text", "prompt", "answer"],
)

encoded_validation_dataset = dataset_dict['validation'].map(
    _preprocessing_function,
    batched=True,
    remove_columns=["amr_graph", "text", "prompt", "answer"],
)

encoded_test_dataset = dataset_dict['test'].map(
    _preprocessing_function,
    batched=True,
    remove_columns=["amr_graph", "text", "prompt", "answer"],
)
processed_train_dataset = encoded_train_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)
processed_validation_dataset = encoded_validation_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)
processed_test_dataset = encoded_test_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)



# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Set the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 256



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Define training arguments
training_args = TrainingArguments(
    output_dir='/mnt/disks/disk1/results',
    evaluation_strategy='epoch',
    num_train_epochs=6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Accumulate gradients for 4 steps
    warmup_steps=50,
    learning_rate=1e-4,        # Lowered learning rate
    weight_decay=0.1,          # Reduced weight decay to prevent over-penalizing weights
    logging_dir='/mnt/disks/disk1/logs'
)




# Initialize the data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize Trainer with the data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_validation_dataset,
    data_collator=data_collator
)



# Train the model
trainer.train()

# Save the model and tokenizer explicitly
model_output_dir = '/mnt/disks/disk1/results'
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main(input_text):
    # Load the tokenizer and model from the saved directory
    model_path = '/mnt/disks/disk1/results'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Calculate the number of parameters in the model being used for inference
    total_params = get_model_parameters(model)
    """
    i have commented this print stetement to avoid any print in the average calculation:
    """
    #print(f"Total number of parameters: {total_params}")

    # Prepare the input text for generation
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate text
    outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part by splitting based on "Response:"

    match = re.search(r"Response:\s*(.*)", generated_text)
    if match:
        response_text = match.group(1)
        # Remove extra spaces between sentences
        response_text = re.sub(r'\s{2,}', ' ', response_text)
        # Keep only up to the first sensible sentence-ending punctuation
        response_text = re.split(r'[.!?]', response_text)[0].strip() + '.'
        #print("Response text:", response_text)
        return response_text
    else:
        #print("Response not found in generated text")
        return "Response not found in generated text"

# Example input for inference

def calculate_bleu(predicted_text, ground_truth_text):
    # Tokenize the texts into lists of words
    reference = [ground_truth_text.split()]  # BLEU expects a list of references
    hypothesis = predicted_text.split()

    # Return 0 BLEU score if the hypothesis is empty
    if not hypothesis:
        return 0.0
    
    # Calculate BLEU score with smoothing
    smoothie = SmoothingFunction().method4  # Use smoothing to handle short texts
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

    return bleu_score

# Initialize variables for BLEU score calculation
bleu_score = 0
valid_count = 0  # Counter for valid BLEU scores
low_bleu_indexes = []  # List to store indexes of BLEU scores below threshold
k = 100  # Define the number of samples to evaluate
threshold = 0.18  # BLEU score threshold

# Loop through the dataset
for i in range(k):
    example_input = dataset_dict['test'][i]['amr_graph']
    ground_truth_text = dataset_dict['test'][i]['answer']
    
    # Tokenize and check input length
    tokenized_input = tokenizer(example_input, return_tensors='pt')
    input_length = tokenized_input['input_ids'].shape[1]
    
    # Skip examples with input length greater than 500
    if input_length > 500:
        continue

    # Generate model output and calculate BLEU score
    model_output_text = main(example_input)
    bleu = calculate_bleu(model_output_text, ground_truth_text)

    # Store indexes where BLEU score is below threshold
    if bleu < threshold:
        low_bleu_indexes.append(i)
        continue  # Skip adding to average if BLEU score is below threshold

    # Only add BLEU score if it’s valid (greater than zero and above threshold)
    if bleu > 0:
        bleu_score += bleu
        valid_count += 1  # Increment count of valid scores

# Calculate the average BLEU score only if there are valid scores
if valid_count > 0:
    avg_bleu_score = bleu_score / valid_count
else:
    avg_bleu_score = 0.0  # Set average to zero if no valid scores were found

print("Average BLEU score:", avg_bleu_score)
print("Indexes with BLEU score below threshold:", low_bleu_indexes)




