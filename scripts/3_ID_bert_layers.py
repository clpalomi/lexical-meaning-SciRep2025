# -*- coding: utf-8 -*-
"""
@author: Claudio
"""
# INSTALL

# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0.tar.gz (THIS ONE)
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz
# pip install fasttext-wheel
# pip install sentence-transformers
# pip install --upgrade numpy==1.26.4  # this worked to address problems with spacy
# pip install scikit-dimension


# IMPORT PACKAGES
import os
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import skdim
import torch
from transformers import AutoModel, AutoTokenizer


# PICTURES
# German:  pict 1: boy, pict 2: countryside, pict 4: couple, pict 6: old woman
# Spanish: pict 1: countryside, pict2: couple, pict3: bridge


# PATH INPUT
path_input     = '/PATH/TO/PRIVATE_DATA/german/texts/'
path_demo      = '/PATH/TO/PRIVATE_DATA/german/'
path_results   = '/PATH/TO/RESULTS/'
path_santander = '/PATH/TO/PRIVATE_DATA/spanish/'
path_london    = '/PATH/TO/PRIVATE_DATA/english/'


""" FUNCTIONS"""

# CREATE A DICTIONARY OF TEXT FROM FOLDER
def read_files_sent(folder_path):
    files_dict = {}
    sentence_pattern = r'[^.!?]*[.!?]'  # Pattern to identify sentences ending with '.', '!', or '?'
    filename_pattern = r'^.*bild_\d+'
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if re.search(filename_pattern, filename) and filename.endswith(".txt"):
                print(filename)
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read().strip()
                    # Add '.' at the end if not present
                    if text[-1] not in '.!?':
                        text += '.'
                    # Replace newline characters with '.' after the last word
                    text = re.sub(r'(\S)\s*\n', r'\1.', text)
                    sentences = re.findall(sentence_pattern, text)
                    # Remove leading whitespaces from each sentence
                    sentences = [sentence.strip() for sentence in sentences]
                    files_dict[filename] = sentences
    return files_dict


# GET NUMBER OF BERT MODEL LAYERS
def get_bert_layers_count(model):
    # Run a dummy input through the model to extract the hidden states
    with torch.no_grad():
        dummy_input = torch.tensor([[101, 102]])  # Minimal input (CLS + SEP)
        outputs = model(input_ids=dummy_input, output_hidden_states=True)
    num_layers = len(outputs.hidden_states)  # Includes word embedding layer
    print(f"The model has {num_layers - 1} transformer layers (+1 embedding layer).")
    return num_layers - 1  # Excluding the embedding layer


# GET BERT EMBEDDINGS 1
def get_bert_embeddings(text, max_length=512, stride=256):
    # Tokenize full text without truncation
    encoded = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    input_ids = encoded["input_ids"][0]  # Extract token IDs
    # Split into overlapping chunks
    chunks = [input_ids[i : i + max_length] for i in range(0, len(input_ids), stride)]
    # Store results
    all_tokens = []
    all_embeddings = []
    for chunk in chunks:
        chunk = chunk[:max_length]  # Ensure chunk is within max_length
        # Convert to tensor and add batch dimension
        chunk_input = {"input_ids": chunk.unsqueeze(0)}
        with torch.no_grad():
            outputs = model(**chunk_input)
        # Extract last hidden state
        embeddings = outputs.last_hidden_state[0]  # (seq_length, hidden_dim)
        tokens = tokenizer.convert_ids_to_tokens(chunk)  # Convert token IDs to readable tokens
        # Store results
        all_tokens.extend(tokens)
        all_embeddings.append(embeddings)
    # Concatenate all embeddings along the token axis
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (total_tokens, hidden_dim)
    return all_embeddings


# GET BERT EMBEDDINGS 2
def get_bert_embeddings_2(text, model, tokenizer, max_length=512, stride=256, hidden_layer=-1):
    """
    Extracts BERT embeddings from a specified hidden layer for a given text.
    
    Parameters:
    - text (str): Input text.
    - model (transformers.PreTrainedModel): Pretrained BERT model.
    - tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    - max_length (int): Maximum sequence length for a chunk.
    - stride (int): Overlapping stride between chunks.
    - hidden_layer (int): Hidden layer index to extract embeddings from (-1 for last layer).
    
    Returns:
    - torch.Tensor: Concatenated embeddings of shape (total_tokens, hidden_dim).
    """
    # Tokenize full text without truncation
    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"][0]  # Extract token IDs
    # Split into overlapping chunks
    chunks = [input_ids[i : i + max_length] for i in range(0, len(input_ids), stride)]
    all_tokens = []
    all_embeddings = []
    for chunk in chunks:
        chunk = chunk[:max_length]  # Ensure chunk is within max_length
        chunk_tensor = torch.tensor(chunk).unsqueeze(0)  # Convert list to tensor and add batch dimension
        attention_mask = torch.ones_like(chunk_tensor)  # Ensure correct attention mask
        with torch.no_grad():
            outputs = model(input_ids=chunk_tensor, attention_mask=attention_mask, output_hidden_states=True)
        # Extract embeddings from the specified hidden layer
        embeddings = outputs.hidden_states[hidden_layer][0]  # Shape: (seq_length, hidden_dim)
        tokens = tokenizer.convert_ids_to_tokens(chunk.tolist())  # Convert to readable tokens
        all_tokens.extend(tokens)
        all_embeddings.append(embeddings)
    # Concatenate all embeddings along the token axis
    all_embeddings = torch.cat(all_embeddings, dim=0)  # Shape: (total_tokens, hidden_dim)
    return all_embeddings


# PCA OF WORD EMBEDDINGS
def perform_pca(data, n_components=0.9, scale_data=True):
    """
    Returns:
    - A dictionary containing the principal components, explained variance, 
      and cumulative variance explained.
    """
    if scale_data:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance_explained = np.cumsum(pca.explained_variance_ratio_)
    return transformed_data, explained_variance, cumulative_variance_explained


# COSINE SIMILARITY MATRIX
def cosine_similarity_matrix(vectors):
    """
    Compute the cosine distance matrix for a set of vectors.
    """
    # Normalize the vectors to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    # Compute the cosine similarity matrix
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    return similarity_matrix


# CALCULATE INTRINSIC DIMENSIONS
def intrinsic_dim(word_embeddings):
    embeddings = np.vstack(word_embeddings)
    # Estimate ID
    dim_raw = skdim.id.MLE().fit(embeddings).dimension_
    embeddings = pd.DataFrame(embeddings)
    # Compute the correlation matrix
    correlation_matrix = embeddings.corr()
    dim_correlation_matrix = skdim.id.MLE().fit(correlation_matrix).dimension_
    # Compute similarity matrix
    sim_matrix = cosine_similarity_matrix(embeddings)
    dim_sim_matrix = skdim.id.MLE().fit(sim_matrix).dimension_
    return [dim_raw, dim_correlation_matrix, dim_sim_matrix]


""" GERMAN PROCESSING: SPLIT SPEECH INTO SENTENCES """

# SAVE EACH LIST OF WORDS
text_files = read_files_sent(path_input)
text_files = {key[:-4]: value for key, value in text_files.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_files = {key: value for key, value in text_files.items() if key[:-7] in names}


""" SPANISH - SANTANDER DATA """

df_sant = pd.read_excel(path_santander+'new_merged_sentences.xlsx')
# COLUMN NAMES
column_names = {
    'Picture': 'picture',
    'Group': 'group',
    'PAR': 'name'
}
df_sant = df_sant.rename(columns=column_names)
# VALUES
column_values = {
    'picture': {'picture_1': '1', 'picture_2': '2', 'picture_3': '3'},
    'group': {'controles': 'HC', 'pacientes': 'SZ'},
}
df_sant.replace(column_values, inplace=True)
# UNIQUE NAMES
names_sant = df_sant[['name', 'group']].drop_duplicates()
names_sant.reset_index(drop=True, inplace=True)
# GROUPS
groups_spa = df_sant[['name','group']]
groups_spa = groups_spa.drop_duplicates().reset_index(drop=True)
# TEXTS TO SANTANDER TEXTS DICTIONARY
text_sant = {}
for (name, picture), group_df in df_sant.groupby(['name', 'picture']):
    print(name, picture)
    key = f'{name}_{picture}'
    utterances = group_df['utterance'].tolist()
    text_sant[key] = utterances


""" ENGLISH - LONDON ONTARIO DATA """

# IMPORT DATA
# Import the .xlsx file into a DataFrame
df_lon = pd.read_excel(path_london+'summary.xlsx')
# Filter rows based on conditions
df_lon = df_lon[(df_lon['Role'] == 'interviewee') & (df_lon['Meta'] != 'Off-task')]
# Drop the columns 'Role', 'Meta', and 'Time'
df_lon = df_lon.drop(['Role', 'Meta', 'Time'], axis=1)
# Convert column names to lowercase
df_lon.columns = df_lon.columns.str.lower()
# GROUPS
# import the .xlsx of groups
df_g = pd.read_excel(path_london+'groups.xlsx')
# MERGE THE DATA WITH GROUPS
df_lon = df_lon.merge(df_g, on='par', how='inner')
df_lon = df_lon[['par', 'group', 'picture', 'utterance']]
df_lon = df_lon.rename(columns = {'par': 'name'})
df_g   = df_g.rename(columns = {'par': 'name'})
# TEXTS TO LONDON TEXTS DICTIONARY
text_london = {}
for (name, picture), group_df in df_lon.groupby(['name', 'picture']):
    print(name, picture)
    key = f'{name}_{picture}'
    utterances = group_df['utterance'].tolist()
    text_london[key] = utterances



""" GERMAN - INTRINSIC DIMENSIONALITY """

# LOAD PRE-TRAINED GERMAN BERT MODEL AND TOKENIZER
model_name = "dbmdz/bert-base-german-cased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name, output_hidden_states=True)
num_layers = get_bert_layers_count(model)  # print number of layers

# LOOP FOR EACH SAMPLE
results = []
for key, lst in text_files.items():
    print(key)
    joined_text = ' '.join(lst)
    # Dictionary for each ID and layer
    id_bert = {}
    id_bert_corr_matrix = {}
    id_bert_sim_matrix = {}
    # Loop thorugh layers
    for layer in range(1, 13):
        emb = get_bert_embeddings_2(joined_text, model=model, tokenizer=tokenizer, hidden_layer=layer)
        dim_raw, dim_corr, dim_sim = intrinsic_dim(emb)
        id_bert[layer] = dim_raw
        id_bert_corr_matrix[layer] = dim_corr
        id_bert_sim_matrix[layer] = dim_sim
    n_tokens = len(get_bert_embeddings_2(joined_text, model=model, tokenizer=tokenizer, hidden_layer=12))
    results.append({
        'name': key,
        'n_tokens': n_tokens,
        **{f'id_bert_{layer}': id_bert[layer] for layer in range(1, 13)},
        **{f'id_bert_corr_matrix_{layer}': id_bert_corr_matrix[layer] for layer in range(1, 13)},
        **{f'id_bert_sim_matrix_{layer}': id_bert_sim_matrix[layer] for layer in range(1, 13)},
    })

# RESULTS DATAFRAME
df = pd.DataFrame(results)
# Extract 'name' and '_bild'
df['bild'] = df['name'].str.extract(r'_bild_(\d+)')
df['name'] = df['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df = df.merge(groups, on='name', how='left')
# Remove columns
df = df.drop(columns=['s_group', 'type'])
# change positions
column_data = df.pop('group')
df.insert(1, 'group', column_data)
column_data = df.pop('bild')
df.insert(2, 'bild', column_data)
df = df.rename(columns={'bild': 'picture'})
df.to_pickle(path_results+'df_intrinsic_dim_bert_german.pkl')
#df = pd.read_pickle(path_results+'df_intrinsic_dim_bert_german.pkl')



""" SPANISH - INTRINSIC DIMENSIONALITY """

# LOAD PRE-TRAINED SPANISH BERT MODEL AND TOKENIZER
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
num_layers = get_bert_layers_count(model)  # print number of layers

# LOOP FOR EACH SAMPLE
results = []
for key, lst in text_sant.items():
    print(key)
    joined_text = ' '.join(lst)
    # Dictionary for each ID and layer
    id_bert = {}
    id_bert_corr_matrix = {}
    id_bert_sim_matrix = {}
    # Loop thorugh layers
    for layer in range(1, 13):
        emb = get_bert_embeddings_2(joined_text, model=model, tokenizer=tokenizer, hidden_layer=layer)
        dim_raw, dim_corr, dim_sim = intrinsic_dim(emb)
        id_bert[layer] = dim_raw
        id_bert_corr_matrix[layer] = dim_corr
        id_bert_sim_matrix[layer] = dim_sim
    n_tokens = len(get_bert_embeddings_2(joined_text, model=model, tokenizer=tokenizer, hidden_layer=12))
    results.append({
        'name': key,
        'n_tokens': n_tokens,
        **{f'id_bert_{layer}': id_bert[layer] for layer in range(1, 13)},
        **{f'id_bert_corr_matrix_{layer}': id_bert_corr_matrix[layer] for layer in range(1, 13)},
        **{f'id_bert_sim_matrix_{layer}': id_bert_sim_matrix[layer] for layer in range(1, 13)},
    })

# RESULTS DATAFRAME
df_spa = pd.DataFrame(results)
# Extract 'name' and '_bild'
df_spa['bild'] = df_spa['name'].str.extract(r'DS_\d+_(\d+)')
df_spa['name'] = df_spa['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_spa = df_spa.merge(groups_spa, on='name', how='left')
# change positions
column_data = df_spa.pop('group')
df_spa.insert(1, 'group', column_data)
column_data = df_spa.pop('bild')
df_spa.insert(2, 'bild', column_data)
df_spa = df_spa.rename(columns={'bild': 'picture'})
df_spa.to_pickle(path_results+'df_intrinsic_dim_bert_spanish.pkl')
# df_spa = pd.read_pickle(path_results+'df_intrinsic_dim_bert_spanish.pkl')



""" ENGLISH - - INTRINSIC DIMENSIONALITY """

# LOAD PRE-TRAINED ENGLISH BERT MODEL AND TOKENIZER
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
num_layers = get_bert_layers_count(model)  # print number of layers

# LOOP FOR EACH SAMPLE
results = []
for key, lst in text_london.items():
    print(key)
    joined_text = ' '.join(lst)
    # Dictionary for each ID and layer
    id_bert = {}
    id_bert_corr_matrix = {}
    id_bert_sim_matrix = {}
    # Loop thorugh layers
    for layer in range(1, 13):
        emb = get_bert_embeddings_2(joined_text, model=model, tokenizer=tokenizer, hidden_layer=layer)
        dim_raw, dim_corr, dim_sim = intrinsic_dim(emb)
        id_bert[layer] = dim_raw
        id_bert_corr_matrix[layer] = dim_corr
        id_bert_sim_matrix[layer] = dim_sim
    n_tokens = len(get_bert_embeddings_2(joined_text, model=model, tokenizer=tokenizer, hidden_layer=12))
    results.append({
        'name': key,
        'n_tokens': n_tokens,
        **{f'id_bert_{layer}': id_bert[layer] for layer in range(1, 13)},
        **{f'id_bert_corr_matrix_{layer}': id_bert_corr_matrix[layer] for layer in range(1, 13)},
        **{f'id_bert_sim_matrix_{layer}': id_bert_sim_matrix[layer] for layer in range(1, 13)},
    })

# RESULTS DATAFRAME
df_eng = pd.DataFrame(results)
# Extract 'name' and '_bild'
df_eng['bild'] = df_eng['name'].str.extract(r'.+_(\d+)')
df_eng['name'] = df_eng['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_eng = df_eng.merge(df_g, on='name', how='left')
# change positions
column_data = df_eng.pop('group')
df_eng.insert(1, 'group', column_data)
column_data = df_eng.pop('bild')
df_eng.insert(2, 'bild', column_data)
df_eng = df_eng.rename(columns={'bild': 'picture'})
df_eng.to_pickle(path_results+'df_intrinsic_dim_bert_english.pkl')
