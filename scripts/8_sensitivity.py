# -*- coding: utf-8 -*-
"""
@author: Claudio
"""
# INSTALL

# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0.tar.gz (THIS ONE)
# pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.8.0/es_core_news_lg-3.8.0.tar.gz
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
import spacy
import fasttext.util
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import skdim
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.neighbors import NearestNeighbors


# PICTURES
# German:  pict 1: boy, pict 2: countryside, pict 4: couple, pict 6: old woman
# Spanish: pict 1: countryside, pict2: couple, pict3: bridge


# PATHS
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


# JOIN SENTENCES OF TEXT_FILES
def join_sentences(sentences):
    joined_text = ' '.join(sentences)
    punctuation_marks = r"[.,:;!?]"
    joined_text = re.sub(punctuation_marks, '', joined_text)
    joined_text = re.sub(r"(\w+)'[\w]+",r'\1', joined_text)
    # Split the joined text into a list of words
    word_list = re.findall(r'\b\w+\b', joined_text)
    return word_list


# FILTER BY CONTENT WORDS
def filter_content(word_list):
    # Join the list of words into a string
    text = ' '.join(word_list)
    # Process the text with spaCy
    doc = nlp(text)
    # Define content POS tags
    content_pos_tags = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}
    # Extract content words (tokens that are not stop words or punctuation)
    content_words_uni = {token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.pos_ in content_pos_tags}
    content_words_all = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in content_pos_tags]
    return [list(content_words_uni), content_words_all]


# GET BERT EMBEDDINGS
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
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (total_tokens, hidden_dim)
    return all_embeddings


# SEMANTIC SIMILARITY (CONSECUTIVE = order 1) WORDS FASTTEXT
def sem_sim_fasttext(text, order=1):
    similarities = []
    # Iterate through the words in the list
    for i in range(len(text) - order):
        # Get the word embeddings for the current pair
        word1_embedding = ft.get_word_vector(text[i])
        word2_embedding = ft.get_word_vector(text[i + order])
    # Concatenate all embeddings along the token axis])
        # Calculate cosine similarity between word embeddings
        similarity = np.dot(word1_embedding, word2_embedding) / (np.linalg.norm(word1_embedding) * np.linalg.norm(word2_embedding))
        # Append the similarity score to the list
        if not np.isnan(similarity):
            similarities.append(similarity)
    return similarities


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
def intrinsic_dim(word_embeddings, k=5):
    embeddings = np.vstack(word_embeddings)
    # Estimate ID
    dim_raw = skdim.id.MLE().fit(embeddings, n_neighbors=k).dimension_
    embeddings = pd.DataFrame(embeddings)
    # Compute the correlation matrix
    correlation_matrix = embeddings.corr()
    dim_correlation_matrix = skdim.id.MLE().fit(correlation_matrix).dimension_
    # Compute similarity matrix
    sim_matrix = cosine_similarity_matrix(embeddings)
    dim_sim_matrix = skdim.id.MLE().fit(sim_matrix).dimension_
    return [dim_raw, dim_correlation_matrix, dim_sim_matrix]


# SENSITIVITY

# LOCAL ID
def local_mle_id(X, k=5, metric='euclidean', eps=1e-12):
    """
    Levina–Bickel local MLE for intrinsic dimension.
    - Caps k at n-1.
    - Avoids log(0) via eps.
    - Returns NaNs for degenerate points instead of raising.
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points to compute neighbors.")
    # cap k safely
    k = int(min(max(1, k), max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dists, _ = nn.kneighbors(X, n_neighbors=k + 1)
    # drop self distance
    d = dists[:, 1:]
    # guard against zeros / ties
    kth = d[:, [-1]]
    ratio = np.maximum(d, eps) / np.maximum(kth, eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(ratio)
        denom = -np.mean(logs, axis=1)          # negative mean(log r_i / r_k)
        denom = np.where(denom <= eps, np.nan, denom)  # avoid 1/0 or tiny denominator
        local_id = 1.0 / denom
    return local_id


# ID SWEEP
def id_sweep(X, ks, metric='euclidean', stop_on_first_invalid=True, agg='median'):
    """
    Sweep over k values. If a requested k is not feasible (k > n-1),
    agg: 'median' or 'mean' for summarizing local IDs.
    """
    X = np.asarray(X)
    n = X.shape[0]
    max_k = max(1, n - 1)
    # ensure ascending, unique ks so we can stop early
    ks_sorted = sorted(set(int(k) for k in ks))
    out = []
    for k in ks_sorted:
        if k > max_k:
            # cannot compute this or any larger k with current n
            if stop_on_first_invalid:
                break
            else:
                continue
        lid = local_mle_id(X, k=k, metric=metric)
        # robust summary ignoring NaNs from degenerate neighborhoods
        if agg == 'median':
            center = np.nanmedian(lid)
        elif agg == 'mean':
            center = np.nanmean(lid)
        else:
            raise ValueError("agg must be 'median' or 'mean'")
        q75 = np.nanpercentile(lid, 75)
        q25 = np.nanpercentile(lid, 25)
        iqr  = q75 - q25
        sd   = np.nanstd(lid, ddof=1)
        n_eff = np.sum(~np.isnan(lid))
        out.append({
            'k': k,
            'center': center,
            'iqr': iqr,
            'sd': sd,
            'n_points_used': int(n_eff),
            'max_k_supported': max_k
        })
    return out



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



""" GERMAN - SENSITIVITY ANALSYS OF ID ESTIMATION """

# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")

# LOAD FASTTEXT
fasttext.util.download_model('de', if_exists='ignore')  # German
ft = fasttext.load_model('cc.de.300.bin')

# LOAD PRE-TRAINED GERMAN BERT MODEL AND TOKENIZER
model_name = "dbmdz/bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# LOOP FOR EACH SAMPLE
results   = []
for key, lst in text_files.items():
    print(key)
    joined_text  = ' '.join(lst)
    # BERT EMBEDDINGS
    bert_emb = get_bert_embeddings(joined_text)
    n_tokens = len(bert_emb)
    # JOINED WORDS
    joined_list  = join_sentences(lst)
    c_words_u    = filter_content(joined_list)[0]
    c_words_a    = filter_content(joined_list)[1]
    n_c_words_u  = len(c_words_u)
    n_c_words_a  = len(c_words_a)
    # FASTTEXT EMBEDDINGS
    word_vectors = [ft.get_word_vector(word) for word in c_words_u]
    # SEMSIM FASTTEXT
    sem_list_1     = sem_sim_fasttext(c_words_a)
    sem_list_1     = [x for x in sem_list_1 if x < 0.9999]
    mean_semsim    = np.mean(sem_list_1).astype(float)
    # EMBEDDINGS TO NP.ARRAY POINTS
    points = np.array(word_vectors)
    points_bert = np.array(bert_emb)
    # CALCULATE INTRINSIC DIMENSIONALITY SWEEP
    range_id_fast  = id_sweep(X=points, ks = tuple(range(2, 41)))
    values_id_fast = [d["center"] for d in range_id_fast]
    range_id_bert  = id_sweep(X=points_bert, ks = tuple(range(2, 51)))
    values_id_bert = [d["center"] for d in range_id_bert]
    # INTRINSIC DIMENSIONALITY USING METHOD
    #values_id_fast_first = [intrinsic_dim(points, k=5) for k in range(2, 41, 1)]
    #values_id_bert_first = [intrinsic_dim(points_bert, k=5) for k in range(2, 51, 1)]
    results.append({'name': key,
                    'n_content_u': n_c_words_u,
                    'n_content_a': n_c_words_a,
                    'n_tokens': n_tokens,
                    'mean_semsim': mean_semsim,
                    'values_id_fast': values_id_fast,
                    'values_id_bert': values_id_bert
                    #'values_id_fast_first': values_id_fast_first,
                    #,'values_id_bert_first': values_id_bert_first
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
df.to_pickle(path_results+'df_words_dim_sensitivity_german.pkl')



""" SPANISH - SENSITIVITY ANALSYS OF ID ESTIMATION """

# LOAD NLP SPACY
nlp = spacy.load("es_core_news_lg")

# LOAD FASTTEXT
fasttext.util.download_model('es', if_exists='ignore')  # Spanish
ft = fasttext.load_model('cc.es.300.bin')

# LOAD PRE-TRAINED SPANISH BERT MODEL AND TOKENIZER
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# LOOP FOR EACH SAMPLE
results   = []
for key, lst in text_sant.items():
    print(key)
    joined_text  = ' '.join(lst)
    # BERT EMBEDDINGS
    bert_emb = get_bert_embeddings(joined_text)
    n_tokens = len(bert_emb)
    # JOINED WORDS
    joined_list  = join_sentences(lst)
    c_words_u    = filter_content(joined_list)[0]
    c_words_a    = filter_content(joined_list)[1]
    n_c_words_u  = len(c_words_u)
    n_c_words_a  = len(c_words_a)
    # FASTTEXT EMBEDDINGS
    word_vectors = [ft.get_word_vector(word) for word in c_words_u]
    # SEMSIM FASTTEXT
    sem_list_1     = sem_sim_fasttext(c_words_a)
    sem_list_1     = [x for x in sem_list_1 if x < 0.9999]
    mean_semsim    = np.mean(sem_list_1).astype(float)
    # EMBEDDINGS TO NP.ARRAY POINTS
    points = np.array(word_vectors)
    points_bert = np.array(bert_emb)
    # CALCULATE INTRINSIC DIMENSIONALITY SWEEP
    range_id_fast  = id_sweep(X=points, ks = tuple(range(2, 41)))
    values_id_fast = [d["center"] for d in range_id_fast]
    range_id_bert  = id_sweep(X=points_bert, ks = tuple(range(2, 51)))
    values_id_bert = [d["center"] for d in range_id_bert]
    results.append({'name': key,
                    'n_content_u': n_c_words_u,
                    'n_content_a': n_c_words_a,
                    'n_tokens': n_tokens,
                    'mean_semsim': mean_semsim,
                    'values_id_fast': values_id_fast,
                    'values_id_bert': values_id_bert
                    })

# RESULTS DATAFRAME
df_sp = pd.DataFrame(results)
# Extract 'name' and '_bild'
df_sp['bild'] = df_sp['name'].str.extract(r'DS_\d+_(\d+)')
df_sp['name'] = df_sp['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_sp = df_sp.merge(groups_spa, on='name', how='left')
# change positions
column_data = df_sp.pop('group')
df_sp.insert(1, 'group', column_data)
column_data = df_sp.pop('bild')
df_sp.insert(2, 'bild', column_data)
df_sp.to_pickle(path_results+'df_words_dim_sensitivity_spanish.pkl')



""" ENGLISH - SENSITIVITY ANALSYS OF ID ESTIMATION """

# LOAD NLP SPACY
nlp = spacy.load("en_core_web_lg")

# LOAD FASTTEXT
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

# LOAD PRE-TRAINED ENGLISH BERT MODEL AND TOKENIZER
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# LOOP FOR EACH SAMPLE
results   = []
for key, lst in text_london.items():
    print(key)
    joined_text  = ' '.join(lst)
    # BERT EMBEDDINGS
    bert_emb = get_bert_embeddings(joined_text)
    n_tokens = len(bert_emb)
    # JOINED WORDS
    joined_list  = join_sentences(lst)
    c_words_u    = filter_content(joined_list)[0]
    c_words_a    = filter_content(joined_list)[1]
    n_c_words_u  = len(c_words_u)
    n_c_words_a  = len(c_words_a)
    # FASTTEXT EMBEDDINGS
    word_vectors = [ft.get_word_vector(word) for word in c_words_u]
    # SEMSIM FASTTEXT
    sem_list_1     = sem_sim_fasttext(c_words_a)
    sem_list_1     = [x for x in sem_list_1 if x < 0.9999]
    mean_semsim    = np.mean(sem_list_1).astype(float)
    # EMBEDDINGS TO NP.ARRAY POINTS
    points = np.array(word_vectors)
    points_bert = np.array(bert_emb)
    # CALCULATE INTRINSIC DIMENSIONALITY SWEEP
    range_id_fast  = id_sweep(X=points, ks = tuple(range(2, 41)))
    values_id_fast = [d["center"] for d in range_id_fast]
    range_id_bert  = id_sweep(X=points_bert, ks = tuple(range(2, 51)))
    values_id_bert = [d["center"] for d in range_id_bert]
    results.append({'name': key,
                    'n_content_u': n_c_words_u,
                    'n_content_a': n_c_words_a,
                    'n_tokens': n_tokens,
                    'mean_semsim': mean_semsim,
                    'values_id_fast': values_id_fast,
                    'values_id_bert': values_id_bert
                    })

# RESULTS DATAFRAME
df_en = pd.DataFrame(results)
# Extract 'name' and '_bild'
df_en['bild'] = df_en['name'].str.extract(r'.+_(\d+)')
df_en['name'] = df_en['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_en = df_en.merge(df_g, on='name', how='left')
# change positions
column_data = df_en.pop('group')
df_en.insert(1, 'group', column_data)
column_data = df_en.pop('bild')
df_en.insert(2, 'bild', column_data)
df_en.to_pickle(path_results+'df_words_dim_sensitivity_english.pkl')


