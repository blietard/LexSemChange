import scipy
import numpy as np
import mangoes
from sklearn.utils.extmath import randomized_svd
from tools.utils import standardize


#============COUNT==============


def create_count_matrix(corpus : mangoes.corpus.Corpus, vocabulary : mangoes.vocabulary.Vocabulary, window_size : int):
    return mangoes.counting.count_cooccurrence(corpus, vocabulary, context = mangoes.context.Window(vocabulary=vocabulary,size=window_size))


def creates_count_matrices_pair(corpus1 : mangoes.corpus.Corpus, corpus2: mangoes.corpus.Corpus, window_size=10, verbose=True):
    '''
    Return co-occurences matrices for both corpora given their shared vocabulary.

    Output:
    > matrix1, the co-occurence matrix of the first corpus
    > matrix2, the co-occurence matrix of the second corpus
    > shared_vocabulary, the vocabulary shared by the two text corpora.
    '''
    
    if verbose:
        print('[INFO] Creating shared vocabulary...')
    vocab1 = corpus1.create_vocabulary()
    if verbose:
        print(len(vocab1),"words in corpus 1")
    vocab2 = corpus2.create_vocabulary()
    if verbose:
        print(len(vocab2),"words in corpus 2")
    shared_vocabulary = mangoes.Vocabulary(list(set(vocab1.words) & set(vocab2.words)))
    if verbose:
        print("Shared vocabulary size:", len(list(set(vocab1.words) & set(vocab2.words))) )
        print('[INFO] Computing count matrix for corpus 1...')
    matrix1 = create_count_matrix(corpus1, shared_vocabulary, window_size)
    if verbose:
        print('[INFO] Success!')
        print('[INFO] Computing count matrix for corpus 2...')
    matrix2 = create_count_matrix(corpus2, shared_vocabulary, window_size)
    if verbose:
        print('[INFO] Success!')
    return (matrix1,matrix2,shared_vocabulary)


#============PPMI==============

def create_ppmi_matrix(counts_matrix, alpha, k):
    return mangoes.create_representation(counts_matrix, weighting=mangoes.weighting.ShiftedPPMI(alpha=alpha,shift=k))

def create_ppmi_matrices_pair(counts_matrix1, counts_matrix2, alpha, k, storage_folder : str, verbose=True):
    if verbose:
        print(f'[INFO] Computing PPMI matrices with alpha={alpha} and k={k}.')
        print('[INFO] Computing PPMI matrix for Corpus 1...')

    ppmi1 = create_ppmi_matrix(counts_matrix1, alpha, k)
    if verbose:
        print('[INFO] Success!')
        print('[INFO] Computing PPMI matrix for Corpus 2...')
    ppmi2 = create_ppmi_matrix(counts_matrix2, alpha, k)
    if verbose:
        print('[INFO] Success!')
    ppmi1.save(storage_folder+'/ppmi1')
    ppmi2.save(storage_folder+'/ppmi2')
    if verbose:
        print(f'[INFO] Matrices stored in {storage_folder}/.')

def load_ppmi_matrices_as_csr(storage_folder: str):
    with np.load(f'{storage_folder}/ppmi1/matrix.npz') as loaded:
        ppmi1_matrix = scipy.sparse.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
    with np.load(f'{storage_folder}/ppmi2/matrix.npz') as loaded:
        ppmi2_matrix = scipy.sparse.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
    return (ppmi1_matrix, ppmi2_matrix)
    

#============SVD==============

def compute_SVD_representation(matrix,dim=100,gamma=1.0,random_state=None,n_iter=5):
    '''
    Compute U*Sigma^gamma representations from the truncated SVD of matrix.
    '''
    u, s, v = randomized_svd(matrix, n_components=dim, n_iter=n_iter, transpose=False,random_state=random_state)
    if gamma == 0.0:
        matrix_reduced = u
    elif gamma == 1.0:
        matrix_reduced = s * u
    else:
        matrix_reduced = np.power(s, gamma) * u
    return matrix_reduced

def create_svd_matrices_pair(matrix1,matrix2,standardise=True, dim=100,gamma=1.0,random_state=None,n_iter=5, verbose=True):
    '''
    If `standardise` is True, gamma is ignored as cancelled by the standardisation process.
    '''
    if verbose:
        print(f'[INFO] Computing {"standardised "*standardise}SVD matrices with gamma={gamma} and d={dim}.')
        print('[INFO] Computing SVD matrix for Corpus 1...')
    if standardise:
        svd1 = standardize(compute_SVD_representation(matrix1,dim,0,random_state=random_state,n_iter=n_iter))
        if verbose:
            print('[INFO] Success!')
            print('[INFO] Computing SVD matrix for Corpus 2...')
        svd2 = standardize(compute_SVD_representation(matrix2,dim,0,random_state=random_state,n_iter=n_iter))
        if verbose:
            print('[INFO] Success!')
    else:
        svd1 = compute_SVD_representation(matrix1,dim,gamma,random_state=random_state,n_iter=n_iter)
        if verbose:
            print('[INFO] Success!')
            print('[INFO] Computing SVD matrix for Corpus 2...')
        svd2 = compute_SVD_representation(matrix2,dim,gamma,random_state=random_state,n_iter=n_iter)
        if verbose:
            print('[INFO] Success!')
    return (svd1,svd2)
        