import numpy as np
from tools.readers import SemEvalReader
from tools.utils import OrthogProcrustAlign, standardize, unitcenter, centerunit, shared_vocabulary
from tools.count_based import *
from tools.pipelines import *
import os


data_folder= '../semeval2020_ulscd_posteval/starting_kit/test_data_public'
storage_folder= '/home/bastien/lscd/static_embdgs/matrices'
language= 'english'
score_folder = './results/scores/'

ws_list = [1,2,3,5,7,10,12,200]
alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
k_list = [1,1.5,2,3,4,5,6,8,10,15]
svd_dim = 300
svd_niter = 5
rng_seed = None
op_func = centerunit


reader, folder, corpus1, corpus2, vocabulary = prepare_SemEval_data(data_folder, storage_folder, language)
reader.read_targets(language,out=False)

word2index = dict()
for i, word in enumerate(list(vocabulary.words)):
    word2index[word]=i

# #=========COUNT=========
# for window_size in ws_list:
#     print(f'[INFO]..... COUNTING WITH WS={window_size} .....')
#     matrix_name = f'count_ws{window_size}'
#     counts_matrix1,counts_matrix2 = count_SemEval_occurences(corpus1,corpus2,vocabulary,window_size, folder, matrix_name)

#=========PPMI=========
## To start from existing COUNT matrices, uncomment the 2 next lines and comment sections COUNT
#for window_size in ws_list:
max_i = len(alpha_list)*len(k_list)
for window_size in ws_list:
    print(f'[INFO]..... COUNTING WITH WS={window_size} .....')
    matrix_name = f'count_ws{window_size}'
    counts_matrix1, counts_matrix2, vocabulary, word2index = load_count_matrices(folder+'count',matrix_name, vocabulary, word2index)
    i = 0
    for ppmi_alpha in alpha_list:
        for ppmi_k in k_list:
            _,_,ppmi_score = compute_score_PPMI(counts_matrix1,counts_matrix2,ppmi_alpha,ppmi_k,word2index,folder,reader,language)
            ppmi_matrix_name = matrix_name + f'-ppmi_a{ppmi_alpha}_k{ppmi_k}'
            rename_and_clean_PPMIs(folder+'ppmi', ppmi_matrix_name, move_to=matrix_name+'/')
            try:
                with open(score_folder+matrix_name+'/'+ppmi_matrix_name,'r') as f:
                    txt = f.read()+'\n'
            except FileNotFoundError:
                txt = ''
            with open(score_folder+matrix_name+'/'+ppmi_matrix_name,'w') as f:
                txt += str(ppmi_score[0]) + '\t' + str(ppmi_score[1])
                f.write(txt)
            i+=1
            print(f'[INFO]..... {np.round(i*100/max_i,2)} .....')
    del counts_matrix1, counts_matrix2

# #=========SVD=========
# ## To start from existing PPMI matrices, uncomment the 2 next lines and comment sections COUNT and PPMI
# #matrix_name = f'count_ws{window_size}-ppmi_a{ppmi_alpha}_k{ppmi_k}'
# #ppmi1, ppmi2 = load_ppmi_matrices_as_csr(folder+'ppmi',vocabulary, word2index, matrix_name)
# matrix_name += f'-svd_d{svd_dim}'
# svd_score = compute_score_SVD(ppmi1,ppmi2,svd_dim,rng_seed,svd_niter,word2index ,folder, reader, language, op_func, matrix_name)
# del ppmi1, ppmi2

