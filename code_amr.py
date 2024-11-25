
from datasets import load_dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from typing import Dict, List
from functools import partial
import copy
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import DatasetDict
from datasets import load_dataset
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM

# Enable W&B dry run mode
os.environ["WANDB_MODE"] = "dryrun"
# Load AMR dataset
dataset = load_dataset("tverous/anli-amr", split="train")

# View the data
# print(dataset[0])
# print(dataset.shape)

# Load AMR dataset
dataset = load_dataset("tverous/anli-amr", split="train")

# Function to extract AMR graph and text
def extract_amr_and_text(data):
    amr_text_pairs = []
    for row in data:
        amr_graph = row.get("amr_penman", None)
        text = row.get("hypothesis", None)
        if amr_graph and text:
            amr_text_pairs.append({"amr_graph": amr_graph, "text": text})
    return amr_text_pairs

# Extract AMR graphs and texts for all rows
amr_text_pairs = extract_amr_and_text(dataset)

amrs = []
texts = []

for i in range(100459):
    amrs.append(amr_text_pairs[i]['amr_graph'])
    texts.append(amr_text_pairs[i]['text'])

# Creating DataFrame with 'amr_graph' and 'text' columns
data_amr = pd.DataFrame({
    'amr_graph': amrs,
    'text': texts
})

#data_amr.head()  # Displaying the first few rows to verify


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert the DataFrame to a Hugging Face dataset
dataset = Dataset.from_pandas(data_amr)

# Select a smaller subset if needed
small_dataset = dataset.select([i for i in range(25000)])

# Define prompt and answer templates
# prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction: {instruction}\n Response:"""
# answer_template = """{response}"""

# Define function to add keys in the dictionary for prompt, answer, and combined text
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
small_dataset = small_dataset.map(_add_text)

# Print the first item to check
# print(small_dataset[0])

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
train_test_split = small_dataset.train_test_split(test_size=0.2)  # Split off 20% as test set
train_valid_split = train_test_split['train'].train_test_split(test_size=0.1)  # From train, split 10% as validation

# Combine splits into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_valid_split['train'],
    'validation': train_valid_split['test'],
    'test': train_test_split['test']
})

# Print the size of each split to verify
# print(f"Train set size: {len(dataset_dict['train'])}")
# print(f"Validation set size: {len(dataset_dict['validation'])}")
# print(f"Test set size: {len(dataset_dict['test'])}")

# # Example check for first item in each split
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



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




# Define training arguments
training_args = TrainingArguments(
    output_dir='/mnt/disks/disk1/results',  ## give the directory name where you want to save the model
    evaluation_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Accumulate gradients for 4 steps
    warmup_steps=50,
    learning_rate=5e-5,        # Lowered learning rate
    weight_decay=0.1,          # Reduced weight decay to prevent over-penalizing weights
    logging_dir='/mnt/disks/disk1/logs' ## give the directory name where you want to save the model
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
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)


def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params




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

# # Example input for inference
# example_input = """
# (z0 / easy-05
#     :ARG1 (z1 / scare-01
#               :ARG1 (z2 / person
#                         :ARG0-of (z3 / have-rel-role-91
#                                      :ARG1 (z4 / i)
#                                      :ARG2 (z5 / uncle))))
#     :mod (z6 / certain))
# """
# output = main(example_input)


# print(processed_train_dataset)
# print(processed_validation_dataset)
# print(processed_test_dataset)



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

bleu_score = 0
valid_count = 0  
k = 10  
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

    # Only add BLEU score if it’s valid (greater than zero)
    if bleu > 0:
        bleu_score += bleu
        valid_count += 1  # Increment count of valid scores

# Calculate the average BLEU score only if there are valid scores
if valid_count > 0:
    avg_bleu_score = bleu_score / valid_count
else:
    avg_bleu_score = 0.0  # Set average to zero if no valid scores were found

print("Average BLEU score:", avg_bleu_score)


