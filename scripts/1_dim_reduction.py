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
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import skdim
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.stats import linregress


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


# CENTROID AND DISPERSION OF WORDS FASTTEXT
def centroid_fasttext(text):
    embeddings = []
    for i in range(1, len(text)):
        embeddings.append(ft.get_word_vector(text[i]))
    # Calculate centroid
    centroid = np.mean(embeddings, axis=0)
    # Calculate dispersion
    dispersion = np.mean([cosine_similarity(centroid.reshape(1, -1), emb.reshape(1, -1))[0, 0] for emb in embeddings])
    # Calculate sum of euclidean distances between consecutive embeddings
    euclidean_distances = [np.linalg.norm(emb1 - emb2) for emb1, emb2 in zip(embeddings[:-1], embeddings[1:])]
    sum_distances = np.sum(euclidean_distances)
    return [centroid, dispersion, sum_distances]


# MOVING WINDOW CENTROID OF WORDS FASTTEXT
def centroid_window_fasttext(text, w):
    centroids = []
    dispersions = []
    sum_distances = []
    for i in range(len(text) - w + 1):
        window_text = text[i:i+w]
        centroid, dispersion, sum_distance = centroid_fasttext(window_text)
        centroids.append(centroid)
        dispersions.append(dispersion)
        sum_distances.append(sum_distance)
    return centroids, dispersions, sum_distances


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


# EIGENSPECTRUM
def compute_eigenspectrum(X, max_components=None, loglog=True):
    # Center data
    X_centered = X - np.mean(X, axis=0)
    # Covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    # Compute and sort eigenvalues
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    eigenvalues_sorted = np.flip(np.sort(eigenvalues))
    # Keep only positive eigenvalues
    eigvals = eigenvalues_sorted[eigenvalues_sorted > 0]
    # Limit to max_components
    if max_components is not None:
        eigvals = eigvals[:max_components]
    # Rank (1-based)
    ranks = np.arange(1, len(eigvals) + 1)
    # Estimate alpha decay
    if loglog:
        log_ranks = np.log(ranks)
        log_eigvals = np.log(eigvals)
        slope, _, _, _, _ = linregress(log_ranks, log_eigvals)
        alpha = -slope
    else:
        alpha = np.nan
    return {
        'eigvals': eigvals,
        'ranks': ranks,
        'alpha': alpha
    }


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



""" GERMAN - SEMANTIC SIMILARITTY BETWEEN WORDS AND REDUCTION """

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
spectrum  = []
centroids = []
pca_3d    = []
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
    # EIGENVALUES: NEW!
    spect = compute_eigenspectrum(word_vectors, max_components=50)
    alpha = spect['alpha']
    # EMBEDDINGS TO NP.ARRAY POINTS
    points = np.array(word_vectors)
    points_bert = np.array(bert_emb)
    # CALCULATE INTRINSIC DIMENSIONALITY
    id_fast             = intrinsic_dim(word_vectors)[0]
    id_fast_corr_matrix = intrinsic_dim(word_vectors)[1]
    id_fast_sim_matrix  = intrinsic_dim(word_vectors)[2]
    id_bert             = intrinsic_dim(bert_emb, k=20)[0]
    id_bert_corr_matrix = intrinsic_dim(bert_emb)[1]
    id_bert_sim_matrix  = intrinsic_dim(bert_emb)[2] 
    # PERFORM PCA
    res        =  perform_pca(points)
    res_bert   =  perform_pca(points_bert)
    var_exp      = []
    var_exp_bert = []
    # PRINCIPAL COMPONENTS
    pc1 = res[0][:,0]
    pc2 = res[0][:,1]
    pc3 = res[0][:,2]
    # FASTTEXT VAR
    for i in range(10):
        try:
            var_exp.append(float(res[2][i]))
        except IndexError:
            var_exp.append(np.nan)
    var_exp_1, var_exp_2, var_exp_3, var_exp_4, var_exp_5, var_exp_6, var_exp_7, var_exp_8, var_exp_9, var_exp_10 = var_exp
    # BERT VAR
    for i in range(10):
        try:
            var_exp_bert.append(float(res_bert[2][i]))
        except IndexError:
            var_exp_bert.append(np.nan)
    var_exp_1_bert, var_exp_2_bert, var_exp_3_bert, var_exp_4_bert, var_exp_5_bert, var_exp_6_bert, var_exp_7_bert, var_exp_8_bert, var_exp_9_bert, var_exp_10_bert = var_exp_bert
    # VARIANCE EXPLAINED FASTTEXT
    n_comp_90 = len(res[2])
    res       = perform_pca(points, n_components=0.5)
    n_comp_50 = len(res[2])
    # VARIANCE EXPLAINED BERT
    n_comp_90_bert = len(res_bert[2])
    res_bert       = perform_pca(points_bert, n_components=0.5)
    n_comp_50_bert = len(res_bert[2])
    # sum Euclidean distances
    sum_distances = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    sum_distances = sum_distances.astype(float)
    # dispersion from centroid
    dispersion = centroid_fasttext(c_words_u)[1].astype(float)
    # moving window centroids
    c, d, s = centroid_window_fasttext(c_words_a, 5)
    # average dispersion moving average
    m_dispersion = np.mean(d).astype(float)
    # semsim fasttext
    sem_list_1     = sem_sim_fasttext(c_words_a)
    sem_list_1     = [x for x in sem_list_1 if x < 0.9999]
    sem_list_2     = sem_sim_fasttext(c_words_a, order=2)
    sem_list_2     = [x for x in sem_list_2 if x < 0.9999]
    sem_list_3     = sem_sim_fasttext(c_words_a, order=3)
    sem_list_3     = [x for x in sem_list_3 if x < 0.9999]
    sem_list_4     = sem_sim_fasttext(c_words_a, order=4)
    sem_list_4     = [x for x in sem_list_4 if x < 0.9999]
    n_content      = len(sem_list_1)+1
    # mean semsim & distribution stats
    mean_semsim    = np.mean(sem_list_1).astype(float)
    max_semsim     = np.max(sem_list_1).astype(float)
    min_semsim     = np.min(sem_list_1).astype(float)
    mean_semsim_2  = np.mean(sem_list_2).astype(float)
    mean_semsim_3  = np.mean(sem_list_3).astype(float)
    mean_semsim_4  = np.mean(sem_list_4).astype(float)
    results.append({'name': key, 'n_content_u': n_c_words_u, 'n_content_a': n_c_words_a,
                    'n_tokens': n_tokens,
                    'alpha': alpha,
                    'id_fast': id_fast, 'id_fast_corr_matrix': id_fast_corr_matrix, 'id_fast_sim_matrix': id_fast_sim_matrix,
                    'id_bert': id_bert, 'id_bert_corr_matrix': id_bert_corr_matrix, 'id_bert_sim_matrix': id_bert_sim_matrix,
                    'n_comp_90': n_comp_90, 'n_comp_50': n_comp_50,
                    'var_exp_1': var_exp_1, 'var_exp_2': var_exp_2,
                    'var_exp_3': var_exp_3, 'var_exp_4': var_exp_4,
                    'var_exp_5': var_exp_5, 'var_exp_6': var_exp_6,
                    'var_exp_7': var_exp_7, 'var_exp_8': var_exp_8,
                    'var_exp_9': var_exp_9, 'var_exp_10': var_exp_10,
                    'n_comp_90_bert': n_comp_90_bert, 'n_comp_50_bert': n_comp_50_bert,
                    'var_exp_1_bert': var_exp_1_bert, 'var_exp_2_bert': var_exp_2_bert,
                    'var_exp_3_bert': var_exp_3_bert, 'var_exp_4_bert': var_exp_4_bert,
                    'var_exp_5_bert': var_exp_5_bert, 'var_exp_6_bert': var_exp_6_bert,
                    'var_exp_7_bert': var_exp_7_bert, 'var_exp_8_bert': var_exp_8_bert,
                    'var_exp_9_bert': var_exp_9_bert, 'var_exp_10_bert': var_exp_10_bert,
                    'sum_dist': sum_distances, 'dispersion': dispersion,
                    'm_dispersion': m_dispersion,
                    'mean_semsim': mean_semsim, 'max_semsim': max_semsim,
                    'min_semsim': min_semsim,
                    'mean_semsim_2': mean_semsim_2, 'mean_semsim_3': mean_semsim_3,
                    'mean_semsim_4': mean_semsim_4,
                    })
    centroids.append({'name': key, 'dispersion': d, 
                      'euclidean_sum': d})
    pca_3d.append({'name': key, 'n_content_u': n_c_words_u, 'n_content_a': n_c_words_a,
                  'id_fast': id_fast,
                  'var_exp_1': var_exp_1,
                  'var_exp_2': var_exp_2,
                  'var_exp_3': var_exp_3,
                  'PC1': pc1,
                  'PC2': pc2,
                  'PC3': pc3})
    spectrum.append({'name': key, 'n_content_u': n_c_words_u,
                    'spectrum': spect})


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
df.to_pickle(path_results+'df_words_dim_reduction_german.pkl')
#df = pd.read_pickle(path_results+'df_words_dim_reduction_german.pkl')


# RESULTS SPECTRUM
df_spec = pd.DataFrame(spectrum)
# Extract 'name' and '_bild'
df_spec['bild'] = df_spec['name'].str.extract(r'_bild_(\d+)')
df_spec['name'] = df_spec['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_spec = df_spec.merge(groups, on='name', how='left')
# Remove columns
df_spec = df_spec.drop(columns=['s_group', 'type'])
# change positions
column_data = df_spec.pop('group')
df_spec.insert(1, 'group', column_data)
column_data = df_spec.pop('bild')
df_spec.insert(2, 'bild', column_data)
df_spec.to_pickle(path_results+'df_words_dim_spectrum_german.pkl')


# PRINCIPAL COMPONENTS TO DATAFRAME
df_pc = pd.DataFrame(pca_3d)
# Extract 'name' and '_bild'
df_pc['bild'] = df_pc['name'].str.extract(r'_bild_(\d+)')
df_pc['name'] = df_pc['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_pc = df_pc.merge(groups, on='name', how='left')
# Remove columns
df_pc = df_pc.drop(columns=['s_group', 'type'])
# change positions
column_data = df_pc.pop('group')
df_pc.insert(1, 'group', column_data)
column_data = df_pc.pop('bild')
df_pc.insert(2, 'bild', column_data)
df_pc.to_pickle(path_results+'df_pca3d_reduction_german.pkl')
#df_pc = pd.read_pickle(path_results+'df_pca3d_reduction_german.pkl')

# MOVING CENTROIDS
df_c = pd.DataFrame(centroids)
# Extract 'name' and '_bild'
df_c['bild'] = df_c['name'].str.extract(r'_bild_(\d+)')
df_c['name'] = df_c['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_c = df_c.merge(groups, on='name', how='left')
# Remove columns
df_c = df_c.drop(columns=['s_group', 'type'])
# change positions
column_data = df_c.pop('group')
df_c.insert(1, 'group', column_data)
column_data = df_c.pop('bild')
df_c.insert(2, 'bild', column_data)
df_c.to_pickle(path_results+'df_words_moving_centroids.pkl')



""" SPANISH - SEMANTIC SIMILARITTY CONSECUTIVE WORDS AND REDUCTION """

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
spectrum  = []
centroids = []
pca_3d    = []
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
    # EIGENVALUES: NEW!
    spect = compute_eigenspectrum(word_vectors, max_components=50)
    alpha = spect['alpha']
    # EMBEDDINGS TO NP.ARRAY POINTS
    points = np.array(word_vectors)
    points_bert = np.array(bert_emb)
    # CALCULATE INTRINSIC DIMENSIONALITY
    id_fast             = intrinsic_dim(word_vectors)[0]
    id_fast_corr_matrix = intrinsic_dim(word_vectors)[1]
    id_fast_sim_matrix  = intrinsic_dim(word_vectors)[2]
    id_bert             = intrinsic_dim(bert_emb)[0]
    id_bert_corr_matrix = intrinsic_dim(bert_emb)[1]
    id_bert_sim_matrix  = intrinsic_dim(bert_emb)[2]
    # PERFORM PCA
    res        =  perform_pca(points)
    res_bert   =  perform_pca(points_bert)
    var_exp      = []
    var_exp_bert = []
    # PRINCIPAL COMPONENTS
    #pc1 = res[0][:,0]
    #pc2 = res[0][:,1]
    #pc3 = res[0][:,2]
    # FASTTEXT VAR
    for i in range(10):
        try:
            var_exp.append(float(res[2][i]))
        except IndexError:
            var_exp.append(np.nan)
    var_exp_1, var_exp_2, var_exp_3, var_exp_4, var_exp_5, var_exp_6, var_exp_7, var_exp_8, var_exp_9, var_exp_10 = var_exp
    # BERT VAR
    for i in range(10):
        try:
            var_exp_bert.append(float(res_bert[2][i]))
        except IndexError:
            var_exp_bert.append(np.nan)
    var_exp_1_bert, var_exp_2_bert, var_exp_3_bert, var_exp_4_bert, var_exp_5_bert, var_exp_6_bert, var_exp_7_bert, var_exp_8_bert, var_exp_9_bert, var_exp_10_bert = var_exp_bert
    # VARIANCE EXPLAINED FASTTEXT
    n_comp_90 = len(res[2])
    res       = perform_pca(points, n_components=0.5)
    n_comp_50 = len(res[2])
    # VARIANCE EXPLAINED BERT
    n_comp_90_bert = len(res_bert[2])
    res_bert       = perform_pca(points_bert, n_components=0.5)
    n_comp_50_bert = len(res_bert[2])
    # sum Euclidean distances
    sum_distances = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    sum_distances = sum_distances.astype(float)
    # dispersion from centroid
    dispersion = centroid_fasttext(c_words_u)[1].astype(float)
    # moving window centroids
    c, d, s = centroid_window_fasttext(c_words_a, 5)
    # average dispersion moving average
    m_dispersion = np.mean(d).astype(float)
    # semsim fasttext
    sem_list_1     = sem_sim_fasttext(c_words_a)
    sem_list_1     = [x for x in sem_list_1 if x < 0.9999]
    sem_list_2     = sem_sim_fasttext(c_words_a, order=2)
    sem_list_2     = [x for x in sem_list_2 if x < 0.9999]
    sem_list_3     = sem_sim_fasttext(c_words_a, order=3)
    sem_list_3     = [x for x in sem_list_3 if x < 0.9999]
    sem_list_4     = sem_sim_fasttext(c_words_a, order=4)
    sem_list_4     = [x for x in sem_list_4 if x < 0.9999]
    n_content      = len(sem_list_1)+1
    # mean semsim & distribution stats
    mean_semsim    = np.mean(sem_list_1).astype(float)
    max_semsim     = np.max(sem_list_1).astype(float)
    min_semsim     = np.min(sem_list_1).astype(float)
    mean_semsim_2  = np.mean(sem_list_2).astype(float)
    mean_semsim_3  = np.mean(sem_list_3).astype(float)
    mean_semsim_4  = np.mean(sem_list_4).astype(float)
    results.append({'name': key, 'n_content_u': n_c_words_u, 'n_content_a': n_c_words_a,
                    'n_tokens': n_tokens,
                    'alpha': alpha,
                    'id_fast': id_fast, 'id_fast_corr_matrix': id_fast_corr_matrix, 'id_fast_sim_matrix': id_fast_sim_matrix,
                    'id_bert': id_bert, 'id_bert_corr_matrix': id_bert_corr_matrix, 'id_bert_sim_matrix': id_bert_sim_matrix,
                    'n_comp_90': n_comp_90, 'n_comp_50': n_comp_50,
                    'var_exp_1': var_exp_1, 'var_exp_2': var_exp_2,
                    'var_exp_3': var_exp_3, 'var_exp_4': var_exp_4,
                    'var_exp_5': var_exp_5, 'var_exp_6': var_exp_6,
                    'var_exp_7': var_exp_7, 'var_exp_8': var_exp_8,
                    'var_exp_9': var_exp_9, 'var_exp_10': var_exp_10,
                    'n_comp_90_bert': n_comp_90_bert, 'n_comp_50_bert': n_comp_50_bert,
                    'var_exp_1_bert': var_exp_1_bert, 'var_exp_2_bert': var_exp_2_bert,
                    'var_exp_3_bert': var_exp_3_bert, 'var_exp_4_bert': var_exp_4_bert,
                    'var_exp_5_bert': var_exp_5_bert, 'var_exp_6_bert': var_exp_6_bert,
                    'var_exp_7_bert': var_exp_7_bert, 'var_exp_8_bert': var_exp_8_bert,
                    'var_exp_9_bert': var_exp_9_bert, 'var_exp_10_bert': var_exp_10_bert,
                    'sum_dist': sum_distances, 'dispersion': dispersion,
                    'm_dispersion': m_dispersion,
                    'mean_semsim': mean_semsim, 'max_semsim': max_semsim,
                    'min_semsim': min_semsim,
                    'mean_semsim_2': mean_semsim_2, 'mean_semsim_3': mean_semsim_3,
                    'mean_semsim_4': mean_semsim_4,
                    })
    centroids.append({'name': key, 'dispersion': d, 
                      'euclidean_sum': d})
    pca_3d.append({'name': key, 'n_content_u': n_c_words_u, 'n_content_a': n_c_words_a,
                   'id_fast': id_fast,
                   'var_exp_1': var_exp_1,
                   'var_exp_2': var_exp_2,
                   'var_exp_3': var_exp_3,
                   'PC1': pc1,
                   'PC2': pc2,
                   'PC3': pc3})
    spectrum.append({'name': key, 'n_content_u': n_c_words_u,
                    'spectrum': spect})


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
df_spa.to_pickle(path_results+'df_words_dim_reduction_spanish.pkl')
# df_spa = pd.read_pickle(path_results+'df_words_dim_reduction_spanish.pkl')


# SPECTRUM DATAFRAME
df_spa_spec = pd.DataFrame(spectrum)
# Extract 'name' and '_bild'
df_spa_spec['bild'] = df_spa_spec['name'].str.extract(r'DS_\d+_(\d+)')
df_spa_spec['name'] = df_spa_spec['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_spa_spec = df_spa_spec.merge(groups_spa, on='name', how='left')
# change positions
column_data = df_spa_spec.pop('group')
df_spa_spec.insert(1, 'group', column_data)
column_data = df_spa_spec.pop('bild')
df_spa_spec.insert(2, 'bild', column_data)
df_spa_spec.to_pickle(path_results+'df_words_dim_spectrum_spanish.pkl')
# df_spa_spec = pd.read_pickle(path_results+'df_words_dim_spectrum_spanish.pkl')


""" ENGLISH - SEMANTIC SIMILARITTY CONSECUTIVE WORDS AND REDUCTION """

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
spectrum  = []
centroids = []
pca_3d    = []
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
    # EIGENVALUES: NEW!
    spect = compute_eigenspectrum(word_vectors, max_components=50)
    alpha = spect['alpha']
    # EMBEDDINGS TO NP.ARRAY POINTS
    points = np.array(word_vectors)
    points_bert = np.array(bert_emb)
    # CALCULATE INTRINSIC DIMENSIONALITY
    id_fast             = intrinsic_dim(word_vectors)[0]
    id_fast_corr_matrix = intrinsic_dim(word_vectors)[1]
    id_fast_sim_matrix  = intrinsic_dim(word_vectors)[2]
    id_bert             = intrinsic_dim(bert_emb)[0]
    id_bert_corr_matrix = intrinsic_dim(bert_emb)[1]
    id_bert_sim_matrix  = intrinsic_dim(bert_emb)[2]
    # PERFORM PCA
    res        =  perform_pca(points)
    res_bert   =  perform_pca(points_bert)
    var_exp      = []
    var_exp_bert = []
    # PRINCIPAL COMPONENTS
    pc1 = res[0][:,0]
    pc2 = res[0][:,1]
    pc3 = res[0][:,2]
    # FASTTEXT VAR
    for i in range(10):
        try:
            var_exp.append(float(res[2][i]))
        except IndexError:
            var_exp.append(np.nan)
    var_exp_1, var_exp_2, var_exp_3, var_exp_4, var_exp_5, var_exp_6, var_exp_7, var_exp_8, var_exp_9, var_exp_10 = var_exp
    # BERT VAR
    for i in range(10):
        try:
            var_exp_bert.append(float(res_bert[2][i]))
        except IndexError:
            var_exp_bert.append(np.nan)
    var_exp_1_bert, var_exp_2_bert, var_exp_3_bert, var_exp_4_bert, var_exp_5_bert, var_exp_6_bert, var_exp_7_bert, var_exp_8_bert, var_exp_9_bert, var_exp_10_bert = var_exp_bert
    # VARIANCE EXPLAINED FASTTEXT
    n_comp_90 = len(res[2])
    res       = perform_pca(points, n_components=0.5)
    n_comp_50 = len(res[2])
    # VARIANCE EXPLAINED BERT
    n_comp_90_bert = len(res_bert[2])
    res_bert       = perform_pca(points_bert, n_components=0.5)
    n_comp_50_bert = len(res_bert[2])
    # sum Euclidean distances
    sum_distances = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    sum_distances = sum_distances.astype(float)
    # dispersion from centroid
    dispersion = centroid_fasttext(c_words_u)[1].astype(float)
    # moving window centroids
    c, d, s = centroid_window_fasttext(c_words_a, 5)
    # average dispersion moving average
    m_dispersion = np.mean(d).astype(float)
    # semsim fasttext
    sem_list_1     = sem_sim_fasttext(c_words_a)
    sem_list_1     = [x for x in sem_list_1 if x < 0.9999]
    sem_list_2     = sem_sim_fasttext(c_words_a, order=2)
    sem_list_2     = [x for x in sem_list_2 if x < 0.9999]
    sem_list_3     = sem_sim_fasttext(c_words_a, order=3)
    sem_list_3     = [x for x in sem_list_3 if x < 0.9999]
    sem_list_4     = sem_sim_fasttext(c_words_a, order=4)
    sem_list_4     = [x for x in sem_list_4 if x < 0.9999]
    n_content      = len(sem_list_1)+1
    # mean semsim & distribution stats
    mean_semsim    = np.mean(sem_list_1).astype(float)
    max_semsim     = np.max(sem_list_1).astype(float)
    min_semsim     = np.min(sem_list_1).astype(float)
    mean_semsim_2  = np.mean(sem_list_2).astype(float)
    mean_semsim_3  = np.mean(sem_list_3).astype(float)
    mean_semsim_4  = np.mean(sem_list_4).astype(float)
    results.append({'name': key, 'n_content_u': n_c_words_u, 'n_content_a': n_c_words_a,
                    'n_tokens': n_tokens,
                    'alpha': alpha,
                    'id_fast': id_fast, 'id_fast_corr_matrix': id_fast_corr_matrix, 'id_fast_sim_matrix': id_fast_sim_matrix,
                    'id_bert': id_bert, 'id_bert_corr_matrix': id_bert_corr_matrix, 'id_bert_sim_matrix': id_bert_sim_matrix,
                    'n_comp_90': n_comp_90, 'n_comp_50': n_comp_50,
                    'var_exp_1': var_exp_1, 'var_exp_2': var_exp_2,
                    'var_exp_3': var_exp_3, 'var_exp_4': var_exp_4,
                    'var_exp_5': var_exp_5, 'var_exp_6': var_exp_6,
                    'var_exp_7': var_exp_7, 'var_exp_8': var_exp_8,
                    'var_exp_9': var_exp_9, 'var_exp_10': var_exp_10,
                    'n_comp_90_bert': n_comp_90_bert, 'n_comp_50_bert': n_comp_50_bert,
                    'var_exp_1_bert': var_exp_1_bert, 'var_exp_2_bert': var_exp_2_bert,
                    'var_exp_3_bert': var_exp_3_bert, 'var_exp_4_bert': var_exp_4_bert,
                    'var_exp_5_bert': var_exp_5_bert, 'var_exp_6_bert': var_exp_6_bert,
                    'var_exp_7_bert': var_exp_7_bert, 'var_exp_8_bert': var_exp_8_bert,
                    'var_exp_9_bert': var_exp_9_bert, 'var_exp_10_bert': var_exp_10_bert,
                    'sum_dist': sum_distances, 'dispersion': dispersion,
                    'm_dispersion': m_dispersion,
                    'mean_semsim': mean_semsim, 'max_semsim': max_semsim,
                    'min_semsim': min_semsim,
                    'mean_semsim_2': mean_semsim_2, 'mean_semsim_3': mean_semsim_3,
                    'mean_semsim_4': mean_semsim_4,
                    })
    spectrum.append({'name': key, 'n_content_u': n_c_words_u,
                    'spectrum': spect})


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
df_eng.to_pickle(path_results+'df_words_dim_reduction_english.pkl')
# df_eng = pd.read_pickle(path_results+'df_words_dim_reduction_english.pkl')


# SPECTRUM DATAFRAME
df_eng_spec = pd.DataFrame(spectrum)
# Extract 'name' and '_bild'
df_eng_spec['bild'] = df_eng_spec['name'].str.extract(r'.+_(\d+)')
df_eng_spec['name'] = df_eng_spec['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_eng_spec = df_eng_spec.merge(df_g, on='name', how='left')
# change positions
column_data = df_eng_spec.pop('group')
df_eng_spec.insert(1, 'group', column_data)
column_data = df_eng_spec.pop('bild')
df_eng_spec.insert(2, 'bild', column_data)
df_eng_spec.to_pickle(path_results+'df_words_dim_spectrum_english.pkl')
