
import os
import amrlib
from amrlib import load_stog_model
import spacy
import spacy
from datasets import load_dataset
from datasets import concatenate_datasets
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
import pandas as pd
from sklearn.model_selection import train_test_split
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
from transformers import AutoTokenizer, AutoModelForCausalLM

# Enable W&B dry run mode
os.environ["WANDB_MODE"] = "dryrun"
# Load the CSV file
file_path = "/kaggle/input/adegethytktk/data_matrix_1.csv"
df = pd.read_csv(file_path)

# Step 1: Define Features (X) and Target (y)
X = df.iloc[:, :-1]  # All columns except the last
y = df.iloc[:, -1]   # The last column as target

# Step 2: Train-Test-Validation Split
# First, split into train and temp (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Then, split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Store splits in a dictionary
data_splits = {
    "train": {"X": X_train, "y": y_train},
    "validation": {"X": X_val, "y": y_val},
    "test": {"X": X_test, "y": y_test}
}
dataset_dict = DatasetDict({
    'train': small_dataset_train,
    'validation': small_dataset_validation,
    'test': small_dataset_test
})

# Check if GPU is available


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Set the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 256


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


# Print the size of each split to verify
print(f"Train set size: {len(dataset_dict['train'])}")
print(f"Validation set size: {len(dataset_dict['validation'])}")
print(f"Test set size: {len(dataset_dict['test'])}")

# Example check for first item in each split
# print("Sample from train:", dataset_dict['train'][0])
# print("Sample from validation:", dataset_dict['validation'][0])
# print("Sample from test:", dataset_dict['test'][0])


# Apply the preprocessing function to each batch in the dataset
encoded_train_dataset = dataset_dict['train'].map(
    _preprocessing_function,
    batched=True
)

encoded_validation_dataset = dataset_dict['validation'].map(
    _preprocessing_function,
    batched=True
)

encoded_test_dataset = dataset_dict['test'].map(
    _preprocessing_function,
    batched=True
)

processed_train_dataset = encoded_train_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)
processed_validation_dataset = encoded_validation_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)
processed_test_dataset = encoded_test_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)






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
    model_path = model_output_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Calculate the number of parameters in the model being used for inference
    total_params = get_model_parameters(model)
    print(f"Total number of parameters: {total_params}")

    # Prepare the input text for generation
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate text
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

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
    example_input = dataset_dict['test'][i]['Linearize graph']
    ground_truth_text = dataset_dict['test'][i]['Abstract']
    
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




