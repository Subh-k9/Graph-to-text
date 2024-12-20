# -*- coding: utf-8 -*-
"""ddpms-gpt2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ibUuRIipTr7zSRYfsL4kL127S3GnBb04
"""

!pip install datasets
!pip install penman
!wget -O amr3.0.tar.gz https://amr.isi.edu/download/amr-bank-3.0.txt
!mkdir amr_data
!tar -xvzf amr3.0.tar.gz -C amr_data
!pip install transformers datasets torch

from datasets import load_dataset

# Load AMR dataset
dataset = load_dataset("tverous/anli-amr", split="train")

# View the data
print(dataset[0])
print(dataset.shape)

from datasets import load_dataset
import pandas as pd

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

data_amr.head()  # Displaying the first few rows to verify

"""### Denoising Diffusion Probabilistic Models (DDPM) with GPT2 as the underlying backbone

#### Step1: Forward Diffusion process
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bar

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer and GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)  # Move GPT-2 model to GPU

# Define beta schedule
T = 1000


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)
betas = linear_beta_schedule(timesteps=T)


# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cuda"):
    """
    Takes text and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0).to(device)  # Move noise to device
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def tokenize_input(text, device="cuda"):
    """ Tokenize input text into token IDs and return embeddings """
    tokens = tokenizer(text, return_tensors="pt").to(device)
    input_ids = tokens['input_ids']
    embeddings = gpt2_model.transformer.wte(input_ids)  # Embeddings layer of GPT-2
    return embeddings, input_ids

def diffusion_on_text(text, T, device="cuda", interval=100):
    """
    Performs the forward diffusion process on token embeddings for all timesteps
    but only applies noise at intervals of 100 steps.
    """
    # Tokenize and get token embeddings
    embeddings, input_ids = tokenize_input(text, device)

    noisy_embeddings = embeddings
    for t in range(T):
        if t % interval == 0:  # Apply noise at every 100th timestep
            timestep = torch.tensor([t]).to(device)  # timestep for current step
            # Apply forward diffusion to embeddings
            noisy_embeddings, noise = forward_diffusion_sample(noisy_embeddings, timestep, device)

    return noisy_embeddings, input_ids

"""#### Step2: Reverse Diffusion"""

def reverse_diffusion_step(model, x_t, t, device="cuda"):
    """ Perform the reverse diffusion step """
    # Predict noise using the model
    predicted_noise = model(x_t)  # The model predicts the noise (epsilon_t)

    # Reverse step formula
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    # Apply reverse diffusion: x_{t-1} = (x_t - predicted_noise * sqrt(1 - alpha_t)) / sqrt(alpha_t)
    x_t_minus_1 = (x_t - predicted_noise * sqrt_one_minus_alphas_cumprod_t) / sqrt_alphas_cumprod_t
    return x_t_minus_1


def reverse_diffusion_on_text(noisy_embeddings, input_ids, T, device="cuda"):
    """
    Performs the reverse diffusion process on noisy token embeddings
    to recover the clean embeddings.
    """
    x_t = noisy_embeddings
    for t in reversed(range(T)):
        timestep = torch.tensor([t]).to(device)  # timestep for current step
        # Perform reverse diffusion step
        x_t = reverse_diffusion_step(gpt2_model, x_t, timestep, device)

    return x_t

"""##### Step1 implementation: Corrupt the text embeddings (choose 15k samples), print the corrupted text for one example"""

from tqdm import tqdm  # Import tqdm for progress bar

# Number of data points you want to corrupt (e.g., first 100 texts)
num_data_points_to_corrupt = 15000  # You can adjust this number as needed

# Iterate over the entire 'text' column with tqdm progress bar, corrupt each text, and store the noisy embeddings
corrupted_texts = []

# Use tqdm to track progress and limit the number of data points processed
for i, example_text in tqdm(enumerate(data_amr['text']), total=num_data_points_to_corrupt, desc="Corrupting texts"):
    if i >= num_data_points_to_corrupt:
        break  # Stop after processing the specified number of data points

    # Corrupt the text embeddings
    noisy_embeddings, input_ids = diffusion_on_text(example_text, T, device, interval=100)

    # If you want to only display one example, you can break after the first one
    if i == 2:  # Change this condition to display any other example, if needed
        # Generate text from noisy embeddings
        if noisy_embeddings.size(-1) == 768:  # Ensure embedding size matches GPT2's hidden size
            outputs = gpt2_model(inputs_embeds=noisy_embeddings, output_hidden_states=True)

            # Decode logits to text
            logits = outputs.logits  # Output logits
            generated_ids = torch.argmax(logits, dim=-1)  # Greedy decoding
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            print("Original Text:", example_text)
            print("Generated Text (from noisy embeddings):", generated_text)
        else:
            print("Error: Embedding size does not match the model's hidden size.")

    # Optionally, store corrupted text embeddings (you can store in a list or process further)
    corrupted_texts.append(noisy_embeddings)

# You can also save or further process the `corrupted_texts` list if needed.

type(input_ids)

"""##### Step2 Implementation: Reconstructing the text embeddings"""

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # For progress bar
import os



# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer and GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Add a custom pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)  # Move GPT-2 model to GPU

# Define beta schedule
T = 1000


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cuda"):
    """
    Takes text and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0).to(device)  # Move noise to device
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


def tokenize_input(texts, device="cuda"):
    """
    Tokenize input text(s) into token IDs and return padded embeddings.
    """
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    embeddings = gpt2_model.transformer.wte(input_ids)  # Embeddings layer of GPT-2
    return embeddings, input_ids, attention_mask



def reverse_diffusion_step(model, x_t, t, device="cuda"):
    """ Perform the reverse diffusion step """
    # Predict noise using the model conditioned on timestep
    predicted_noise = model(x_t, t)

    # Reverse step formula
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    # Apply reverse diffusion: x_{t-1} = (x_t - predicted_noise * sqrt(1 - alpha_t)) / sqrt(alpha_t)
    x_t_minus_1 = (x_t - predicted_noise * sqrt_one_minus_alphas_cumprod_t) / sqrt_alphas_cumprod_t
    return x_t_minus_1


def reverse_diffusion_on_text(model, noisy_embeddings, T, device="cuda"):
    """
    Performs the reverse diffusion process on noisy token embeddings
    to recover the clean embeddings.
    """
    x_t = noisy_embeddings
    for t in reversed(range(T)):
        timestep = torch.tensor([t]).to(device)  # timestep for current step
        # Perform reverse diffusion step
        x_t = reverse_diffusion_step(model, x_t, timestep, device)

    return x_t


# Define the noise prediction model
class NoisePredictor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x, t):
        # Combine input and timestep as a conditioning mechanism
        t_embed = torch.ones_like(x) * t.view(-1, 1, 1)  # Shape match
        return self.mlp(x + t_embed)

"""#### Graph Embeddings"""

!pip install torch_geometric

import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


# Define a sample GAT model (this should be replaced with your pretrained model)
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=heads, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x



pretrained_gat = GAT(input_dim, hidden_dim, output_dim)
pretrained_gat.load_state_dict(torch.load("pretrained_gat_model.pth"))  # Load pretrained weights
pretrained_gat.eval()  # Set model to evaluation mode

# Obtain embeddings for the graph nodes
with torch.no_grad():
    node_embeddings = pretrained_gat(graph_data.x, graph_data.edge_index)

# print("Node embeddings:")
# print(node_embeddings)

"""#### Contrastive Learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoCNELoss(nn.Module):
    def __init__(self, temperature=0.1):
        """
        InfoCNE (Information Contrastive Noise Estimation) loss function.
        :param temperature: A temperature scaling factor for similarity calculation.
        """
        super(InfoCNELoss, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, graph_embeddings):
        """
        Forward pass for InfoCNE loss calculation.
        :param text_embeddings: Noisy text embeddings [batch_size, embed_dim].
        :param graph_embeddings: Graph node embeddings [batch_size, embed_dim].
        :return: InfoCNE loss value.
        """
        # Normalize the embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(text_embeddings, graph_embeddings.T)  # [batch_size, batch_size]

        # Apply temperature scaling
        similarity_matrix /= self.temperature

        # Create labels for positive pairs (diagonal elements are positive)
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

        # Cross-entropy loss based on similarity scores
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


# Instantiate the InfoCNE loss
info_cne_loss_fn = InfoCNELoss(temperature=0.1)

def train_diffusion_model_with_infoCNE(model, optimizer, text_data, epochs=5, batch_size=2, T=1000, device="cuda", embed_dim=256):
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i + batch_size]

            # Tokenize and get embeddings
            batch_embeddings, input_ids, attention_mask = tokenize_input(batch, device)

            # Randomly sample timesteps
            batch_size = batch_embeddings.size(0)
            t = torch.randint(0, T, (batch_size,), device=device)

            # Forward diffusion
            noisy_embeddings, noise = forward_diffusion_sample(batch_embeddings, t, device)

            # Predict noise using the model
            predicted_noise = model(noisy_embeddings)

            # Mask padding tokens during loss computation
            valid_tokens = attention_mask.unsqueeze(-1).expand_as(predicted_noise)
            diffusion_loss = F.mse_loss(predicted_noise[valid_tokens], noise[valid_tokens])

            # Compute the InfoCNE loss between noisy embeddings and node embeddings (text vs. graph)
            info_cne_loss = info_cne_loss_fn(noisy_embeddings, batch_embeddings)

            # Total loss: combination of diffusion loss and InfoCNE loss
            total_loss = diffusion_loss + 0.1 * info_cne_loss  # Adjust the weight of the InfoCNE loss

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Print loss
            print(f"Batch Loss: {total_loss.item():.4f} (Diffusion Loss: {diffusion_loss.item():.4f}, InfoCNE Loss: {info_cne_loss.item():.4f})")

# Main training execution
embedding_dim = gpt2_model.transformer.wte.weight.shape[1]
noise_predictor = NoisePredictor(embedding_dim).to(device)
optimizer = torch.optim.Adam(noise_predictor.parameters(), lr=1e-4)


# Train the model with diffusion + InfoCNE loss
train_diffusion_model_with_infoCNE(noise_predictor, optimizer, text_data, epochs=5, batch_size=32, device="cuda", embed_dim=embedding_dim)