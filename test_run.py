## Required Modules

import pandas as pd
import numpy as np
import Bio
import os
from Bio import Entrez, SeqIO
import itertools
import argparse
import math
import xmltodict
from pprint import pprint
import torch
from torch import nn
import h5py
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import time
from itertools import compress
import pickle
from tqdm import tqdm 

      
if __name__ == '__main__':
  trial_run = 10000
  summarystat = '/Input/Random_Million_SNPs_Part3.txt'
  args_model = '/Input/deepsea.beluga.pth'
  features = '/Input/deepsea_beluga_2002_features.tsv.txt'
  numsnps = 1000
  randomset = True
  ## Filtering the top N snps
  print("Loading random SNPs chromosome and position..")
  ss = pd.read_csv(summarystat, sep='\t')
  top_n_snps = ss
  
  ## Obtaining features
  print("Loading feature info...")
  
  features = pd.read_csv(features, sep = '\t')
  features['feature_names'] = features['Cell type'] +'__'+ features['Assay']+'__'+ features['Assay type']
  features_ids_dnase = [features['Assay type']=='DNase']
  features_ids_tf = [features['Assay type']=='TF']
  features_ids_histone = [features['Assay type']=='Histone']
  feature_names = features['feature_names']
  
  
  ## Inputing the resources for Expect.py
  inputsize = 2000
  batchSize = 32
  maxshift = 800
  args_cuda = True
  
  ## Importing the DL Model
  model = Beluga()
  model.load_state_dict(torch.load(args_model))
  model.eval()
  
  
  
  
  print("Loading Fasta sequence ...")
  fasta_available = True
  fasta_whole_genome = SeqIO.to_dict(SeqIO.parse("Hg38/hg38.fa","fasta"))
  import multiprocessing
  import time
  data_matrix_ind_snp_p = {}
  args_cuda = True

  print("Overall Check done. Ready for deployment!")
