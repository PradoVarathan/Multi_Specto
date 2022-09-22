#Importing important modules
import pandas as pd
import numpy as np
import Bio
import os
import matplotlib.pyplot as plt
from Bio import Entrez, SeqIO
import itertools
import argparse
import math
from Bio import Entrez
import xmltodict
from pprint import pprint
import torch
from torch import nn
import h5py
from tqdm import tqdm 
import yaml
import pickle
from pyfiglet import Figlet
import sys
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import time
from itertools import compress
#Inputing the resources for Expect.py
inputsize = 2000
batchSize = 32
maxshift = 800
args_cuda = False


#DL model
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)



def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.
    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output
    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize
    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp

def get_predicted_diff(snp_comb_seq,inputsize = 2000, batchSize = 32, maxshift = 800, args_cuda = False):
    """
    Function to obtain all the predicted chromatin values for reference and alterante 
    and find the difference among them for further analysis.
    Args:
        snp_comb_seq: A dictionary of sequences as string object with A,T,G,C characters
                        and keys corresponding to snps and combinations of snps with atleast
                        one snp having 'Ref' in the key name to denote reference variant
    Return:
            A dictionary of matrix size 4000x2002 for the chromatin difference values for each 
            variant and combination except the reference
    """
    refseqs = [seq for key, seq in snp_comb_seq.items() if 'ref' in key.lower()]
    ref_encoded = encodeSeqs(refseqs, inputsize=inputsize).astype(np.float32)

    ref_preds = []
    for i in range(int(1 + (ref_encoded.shape[0]-1) / batchSize)):
        input = torch.from_numpy(ref_encoded[int(i*batchSize):int((i+1)*batchSize),:,:]).unsqueeze(2)
        if args_cuda:
            input = input.cuda()
        ref_preds.append(model.forward(input).cpu().detach().numpy().copy())
    ref_preds = np.vstack(ref_preds)
    
    comb_diff_pred = {}
    for comb_seq in snp_comb_seq.keys():

        if('Ref' not in comb_seq):

            altseqs = [snp_comb_seq[comb_seq]]
            alt_encoded = encodeSeqs(altseqs, inputsize=inputsize).astype(np.float32)

            alt_preds = []
            for i in range(int(1 + (alt_encoded.shape[0]-1) / batchSize)):
                input = torch.from_numpy(alt_encoded[int(i*batchSize):int((i+1)*batchSize),:,:]).unsqueeze(2)
                if args_cuda:
                    input = input.cuda()
                alt_preds.append(model.forward(input).cpu().detach().numpy().copy())
            alt_preds = np.vstack(alt_preds)

            diff = np.log2(ref_preds/(1-ref_preds)) - np.log2(alt_preds/(1-alt_preds)) 
            comb_diff_pred[comb_seq] = diff
    
    
    return comb_diff_pred


file_path_template = sys.argv[1]
file = open(file_path_template,'r')
cfc = yaml.load(file,Loader=yaml.FullLoader)['default']


#Defining a SNP class to perform simple LD filtering duties
class SNP:
    
    def __init__(self,rsid,position,chromosome):
        self.rsid = rsid
        self.position = position
        self.chr = chromosome
    

        
    def check_ld_snps(self,dataset,window = 1000):
        start_position = self.position - window + 1
        end_position = self.position + window
        dataset = dataset[dataset['Chromosome'] == self.chr]
        def extract_neighbour_snps(start_position, end_position, dataset):
            neighbour_snps = []
            for index,row in dataset.iterrows():
                if start_position <= dataset.loc[index,'Position'] <= end_position:
                    neighbour_snps.append(dataset.loc[index,'MarkerName'])
                else:
                    continue
            return neighbour_snps
    
        self.snps_in_window = extract_neighbour_snps(start_position,end_position,dataset)
        return self.snps_in_window
    

    def obtain_snp_sequence(self,dataset,window = 1000):
        start_position = self.position - window +1
        end_position = self.position + window 
        if int(self.chr) < 10:
            id_chr = "".join(["NC_00000",str(self.chr)])
        else:
            id_chr = "".join(["NC_0000",str(self.chr)])

        handle = Entrez.efetch(db="nucleotide",
                        id = id_chr,
                        rettype = "fasta",
                        strand = 1,
                        seq_start = start_position,
                        seq_stop  = end_position)
        record = SeqIO.read(handle,"fasta")
        Seq_temp = str(record.seq)
        idx = dataset['MarkerName'] == self.rsid
        allele = str(dataset.loc[idx,'Non_Effect_allele'].values[0])                      #Effect allele is the minor allele; Non-effect allele is the major allele
        self.snp_sequence = Seq_temp[:1000-1] + allele + Seq_temp[1000:]
        #Changing other alleles in neighbouring area to major allele
        start_position = self.position - window + 1
        for snp_neigh in self.snps_in_window:
            idx = dataset['MarkerName'] == snp_neigh
            allele = str(dataset.loc[idx,'Non_Effect_allele'].values[0])
            pos = dataset.loc[idx,'Position']
            net_pos = int(pos) - int(start_position)
            self.snp_sequence = self.snp_sequence[:net_pos-1] + allele + self.snp_sequence[net_pos:]

        return self.snp_sequence
    
    def obtain_all_comb_seq(self,dataset,sign_num = 'null', window = 1000):
        
        def all_snp_combinations(a):
            combinations = []
            for k in range(0,len(a)):
                t = list(itertools.combinations(a,k+1))
                combinations.extend(t)
            return combinations
        
        self.combinations = all_snp_combinations(self.snps_in_window)
        comb_names = ['_'.join(x) for x in self.combinations if len(x)> 0]
        comb_names.append('_'.join(['Ref',self.rsid]))
        combination_dataset = dataset[dataset['MarkerName'].isin(self.snps_in_window)]
        if sign_num != 'null':
            combination_dataset = combination_dataset.sort_values('Pvalue')
            combination_dataset = combination_dataset.iloc[0:int(sign_num),:]
        sequences = []
        

        for comb in self.combinations:
            seq_to_change = self.snp_sequence
            start_position = self.position - window + 1
            end_position = self.position + window
            for k in range(0,len(comb)):
                idx = combination_dataset['MarkerName'] == comb[k]
                pos = combination_dataset.loc[idx,'Position']
                allele = str(combination_dataset.loc[idx,'Effect_allele'].values[0])
                net_pos = int(pos) - int(start_position)
                seq_to_change = seq_to_change[:net_pos-1] + allele + seq_to_change[net_pos:]
            sequences.append(seq_to_change)
        sequences.append(self.snp_sequence)
        sequences_named = dict(zip(comb_names,sequences))
        return sequences_named
    
    def plot_combinations_over_length(self,dataset,window = 1000):
        self.check_ld_snps(dataset,window)
        t_snps = self.snps_in_window
        temp = dataset[dataset['MarkerName'].isin(t_snps)]
        temp_pos = temp['Position'] - self.position
        ax = plt.hist(temp_pos,bins=10, edgecolor="black")
        ax.title(self.rsid + " Combinations Frequency with Total "+str(len(t_snps)))
        ax.axvline(0, color="black", ls="--", label=self.rsid)
        out_name = str(cfc['output_files']['combination_distribution_folder']) + self.rsid + "_Combinations_Frequency_over_length.jpg"
        ax.savefig(out_name)
        plt.clf()
                
    def seq_combination(self,dataset,sign_num = 'null',window = 1000):
        self.check_ld_snps(dataset,window)
        self.obtain_snp_sequence(dataset)
        #self.plot_combinations_over_length(dataset)
        self.combination_seq = self.obtain_all_comb_seq(dataset,sign_num,window)
        return self.combination_seq
        
    
    def __str__(self):
        return "The SNP in object is "+self.rsid
        
        
def group_check(snp_feature_list,snp_comb_name_list):
    snp_groups_bool = []
    for k in snp_comb_name_list:
        snp_groups_bool.append('_' not in k)
    snp_groups_bool = sum(([ss]*2 for ss in snp_groups_bool),[])
    group1 = snp_feature_list[snp_groups_bool]

    # changed with absolute values but keep the sign
    a = np.argmax(np.abs(group1),axis=0)
    b = np.array(range(0,2002))
    group1_overall = [group1[x][y] for x, y in zip(a,b)]

 
    
    group2_idxs = [not idx for idx in snp_groups_bool]
    group2 = snp_feature_list[group2_idxs]
    group2_sub = np.subtract(group1_overall,group2)
    group2_ratio = np.divide(group2,group1_overall)

    return group1,group2,group1_overall,group2_sub,group2_ratio,group2_idxs,snp_groups_bool


def correlation_check(snp_feature_list, snp_comb_name_list):
    
    filtered_combs = []
    for snp1 in range(0,len(snp_feature_list)):
        for snp2 in range(0,len(snp_feature_list)):
            corr,_ = pearsonr(snp_feature_list[snp1],snp_feature_list[snp2])
            if -0.5 <= corr <= 0.5:
                filtered_combs.append(snp1)
                filtered_combs.append(snp2)
    filtered_combs = list(set(filtered_combs))            
    return filtered_combs        
              
        
        




f = Figlet(font='slant')
print(f.renderText('MultiSpecto Basic Run'))
print(" --------  Expecto based model for top N snps from provided GWAS along with their neighoburs  --------")

print("Loading summary statistic file..")
Entrez.email  = str(cfc['entrez_cred']['entrez_email']) 
Entrez.api_key = str(cfc['entrez_cred']['entrez_api']) 
model = Beluga()
model.load_state_dict(torch.load(str(cfc['input_files']['model_dict'])))
model.eval()
skip_snps = ['rs59007384','rs111789331']
igap_19 = pd.read_csv(str(cfc['input_files']['secondary_dataset']))
igap = pd.read_csv(str(cfc['input_files']['primary_dataset']), sep='\t')
#Filtering the igap snps 
igap = igap.sort_values(by = ['Pvalue'], ascending=True)
N = int(cfc['input_files']['n'])
top_100_igap_snps = igap.iloc[0:N,:]

for index,row in top_100_igap_snps.iterrows():
    response = Entrez.efetch(db='SNP', id=str(top_100_igap_snps.loc[index,'MarkerName'])).read()
    response = response[:-1]
    response_o = xmltodict.parse(response)
    pos = response_o['DocumentSummary']['CHRPOS']
    pos = pos.split(':')[1]
    top_100_igap_snps.loc[index,'Position'] = int(pos)

## Obtaining features
print("Loading feature info...")

features = pd.read_csv(str(cfc['input_files']['features_info']), sep = '\t')
features['feature_names'] = features['Cell type'] +'__'+ features['Assay']+'__'+ features['Assay type']
features_ids_dnase = [features['Assay type']=='DNase']
features_ids_tf = [features['Assay type']=='TF']
features_ids_histone = [features['Assay type']=='Histone']
feature_names = features['feature_names']


#Running over top n snps
freq_df = {}
data_matrix_grp1 = {}
data_matrix_grp2 = {}
data_matrix_ind_snp = {}
data_matrix_grp1_overall = {}
for k in tqdm(range(0,len(top_100_igap_snps))):
    snp_test = top_100_igap_snps.iloc[k,2]
    if snp_test not in skip_snps:
        #print("Running %s ...."%(snp_test))
        snp_obj = SNP(top_100_igap_snps.iloc[k,2],top_100_igap_snps.iloc[k,1],top_100_igap_snps.iloc[k,0])
        #print("Obtaining combinations for %s ...."%(snp_test))
        freq_df[snp_test] = (len(snp_obj.check_ld_snps(igap_19)),'_'.join(snp_obj.check_ld_snps(igap_19)))
        snp_comb_seq = snp_obj.seq_combination(igap_19)
        #print("Predicting the sequence profiles for %s ...."%(snp_test))
        comb_diff_pred = get_predicted_diff(snp_comb_seq)
        f = h5py.File(str(cfc['output_files']['h5_folder'])+'/' + snp_test +'.diff.h5', 'w')
        key_names = list(comb_diff_pred.keys())
        #print("Saving the result for %s"%(snp_test))
        for i in key_names:
            f.create_dataset(i, data=comb_diff_pred[i])
        f.close()
        f_name = str(cfc['output_files']['h5_folder'])+'/' + snp_test +'.diff.h5'
        f = h5py.File(f_name,'r+')  
        combs_keys = list(f.keys())
        snp_list = np.zeros((len(combs_keys)*2,2002))
        for i in range(0,2*len(combs_keys),2):
            snp_list[i,:] = f[combs_keys[int(i/2)]][()][0,:]
            snp_list[i+1,:] = f[combs_keys[int(i/2)]][()][0,:]
        t_name =snp_test
        # Am I running with the threshold combination or not?
        snp_vals,comb_vals,group1,met2_sub,met2_ratio,idd,snp_groups_bool = group_check(snp_list,combs_keys)      
        data_matrix_grp1[t_name] = {'data': group1,'ylabels':list(feature_names)}
        xlabs = key_names
        xlabs = [[x]*2 for x in xlabs]
        xlabs =  [item for sublist in xlabs for item in sublist]
        iddx = [idd[i] for i in range(0,len(idd),2)]
        xlabels = [key_names[i] for i in range(0,len(iddx)) if iddx[i]]
        xlabels = sum(([ss]*2 for ss in xlabels),[])
        snp_groups_bool = [snp_groups_bool[x] for x in range(0,len(snp_groups_bool),2)]
        xlabels = [combs_keys[x] for x in range(0,len(snp_groups_bool)) if not snp_groups_bool[x]]
        xlabels = sum(([ss]*2 for ss in xlabels),[])
        data_matrix_grp2[t_name] = {'data': comb_vals,'ylabels':xlabels, 'xlabels':list(feature_names)}
        xlabels = [combs_keys[x] for x in range(0,len(snp_groups_bool)) if snp_groups_bool[x]]
        xlabels = sum(([ss]*2 for ss in xlabels),[])
        data_matrix_ind_snp[t_name] = {'data':snp_vals,'xlabels':list(xlabels)}


with open(str(cfc['output_files']['h5_folder'])+'/'+'DataMatrixGrp2.pickle', 'wb') as file:
    pickle.dump(data_matrix_grp2, file, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(cfc['output_files']['h5_folder'])+'/'+'DataMatrixGrp1.pickle', 'wb') as file:
    pickle.dump(data_matrix_grp1, file, protocol=pickle.HIGHEST_PROTOCOL)
with open(str(cfc['output_files']['h5_folder'])+'/'+'DataMatrixOg.pickle','wb') as file:
    pickle.dump(data_matrix_ind_snp,file, protocol=pickle.HIGHEST_PROTOCOL)
    
pd.DataFrame.from_dict(freq_df,orient='index').to_csv(str(cfc['output_files']['combination_distribution_folder'])+'/Freq_neighbouring.csv')

f = Figlet(font='crawford')
print(f.renderText('MultiSpecto Completed'))


