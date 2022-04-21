import mangoes
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cos_dist
from collections import defaultdict

class Reader():
    '''
    Base class for any dataset reader
    '''
    
    def __init__(self,path : str):
        self.path = path
        self.targets = None
    
    def load_2periods_corpora(self, corpus1_path,corpus2_path, verbose=True):
        '''
        Return mangoes.corpus.Corpus object for a pair of corpora.
        '''
        if verbose:
            print('[INFO] Building corpus 1...')
        corpus1 = mangoes.Corpus(corpus1_path)
        if verbose:
            print(f"[INFO] Corpus 1: {corpus1.nb_sentences} sentences \t{len(corpus1._words_count)} words")

            print('[INFO] Building corpus 2...')
        corpus2 = mangoes.Corpus(corpus2_path)
        if verbose:
            print(f"[INFO] Corpus 2: {corpus2.nb_sentences} sentences \t{len(corpus2._words_count)} words")
        return (corpus1, corpus2)

    def read_targets(self,language : str, out=True):
        pass

    def spearman_score(self,matrix1,matrix2,word2idx_dict1,word2idx_dict2,targets, gold_scores,out=True):
        if hasattr(matrix1,'toarray'):
            #need for conversion to array
            dist = lambda x1,x2 : cos_dist(x1.toarray(),x2.toarray()) 
        else:
            #no need
            dist = cos_dist

        distances = np.empty( len(targets) )
        for i, word in enumerate(targets):
            idx1 = word2idx_dict1[word]
            idx2 = word2idx_dict2[word]
            distances[i] = dist(matrix1[idx1],matrix2[idx2])
            
        rho, p = spearmanr( distances, gold_scores )
        if out:
            return (rho.round(5),p.round(4))
        else:
            print(f'Spearman\'s rho: {rho.round(5)} \tp-value: {p.round(4)}')
    

class SemEvalReader(Reader):
    '''
    Reader for SemEval2020 Task 1 (subtask 2) data.

    Parameters:
    > 'path': str
        Path to find the data repository.
    '''

    def __init__(self,path: str):
        super().__init__(path)
        self.targets = defaultdict(list)
        self.gold_scores = defaultdict(np.array)
    
    
    def load_corpora(self,language : str, subcorpus=None, verbose=True):
        '''
        Return mangoes.corpus.Corpus object for both corpora of the selected language.
        If subcorpus is provided, will read `self.path/language/corpus{X}/{subcorpus}/`
        '''
        if subcorpus is None:
            path_end = ''
        else:
            path_end = subcorpus+'/'
        
        return super().load_2periods_corpora(f'{self.path}/{language}/corpus1/'+path_end,
                                             f'{self.path}/{language}/corpus2/'+path_end,
                                             verbose)

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
    
    def spearman_score(self,matrix1,matrix2,word2idx_dict1,word2idx_dict2,language,out=True):
        targets = self.targets[language]
        gold_scores = self.gold_scores[language]
        score = super().spearman_score(matrix1, matrix2, word2idx_dict1, word2idx_dict2, targets, gold_scores, out)
        if out:
            return score
