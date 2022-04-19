import mangoes
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cos_dist
from collections import defaultdict

class SemEvalReader():
    '''
    Reader for SemEval2020 Task 1 (subtask 2) data.

    Parameters:
    > 'path': str
        Path to find the data repository.
    '''

    def __init__(self,path : str):
        self.path = path
        self.targets = defaultdict(list)
        self.gold_scores = defaultdict(np.array)

    def read_targets(self,language : str, out=True):
        '''
        Return the list of target words and the numpy array of corresponding gold_scores.
        '''
        with open(f'{self.path}/{language}/truth/graded.txt','r') as f:
            lines = f.read().split('\n')
        lines.pop(-1)
        targets = [ line.split()[0] for line in lines]
        gold_scores = np.array( [ line.split()[1] for line in lines] ,dtype='float64')
        self.targets[language] = targets.copy()
        self.gold_scores[language] = gold_scores.copy()
        if out:
            return (targets,gold_scores)

    def load_corpora(self,language : str, subcorpus=None, verbose=True):
        '''
        Return mangoes.corpus.Corpus object for both corpora of the selected language.
        If subcorpus is provided, will read `self.path/language/corpus{X}/{subcorpus}/`
        '''
        if subcorpus is None:
            path_end = ''
        else:
            path_end = subcorpus+'/'
        if verbose:
            print('[INFO] Building corpus 1...')
        corpus1 = mangoes.Corpus(f'{self.path}/{language}/corpus1/'+path_end)
        if verbose:
            print(f"[INFO] Corpus 1: {corpus1.nb_sentences} sentences \t{len(corpus1._words_count)} words")

            print('[INFO] Building corpus 2...')
        corpus2 = mangoes.Corpus(f'{self.path}/{language}/corpus2/'+path_end)
        if verbose:
            print(f"[INFO] Corpus 2: {corpus2.nb_sentences} sentences \t{len(corpus2._words_count)} words")
        return (corpus1, corpus2)

    def spearman_score(self,matrix1,matrix2,word2index,language,out=True):
        targets = self.targets[language]
        gold_scores = self.gold_scores[language]

        if hasattr(matrix1,'toarray'):
            #need for conversion to array
            dist = lambda x1,x2 : cos_dist(x1.toarray(),x2.toarray()) 
        else:
            #no need
            dist = cos_dist

        distances = np.empty( len(targets) )
        for i, word in enumerate(targets):
            index = word2index[word]
            distances[i] = dist(matrix1[index],matrix2[index])
            
        rho, p = spearmanr( distances, gold_scores )
        if out:
            return (rho.round(5),p.round(4))
        else:
            print(f'Spearman\'s rho: {rho.round(5)} \tp-value: {p.round(4)}')