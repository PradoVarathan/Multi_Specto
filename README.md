# Deciphering the tissue-specific functional effect of Alzheimer risk SNPs with deep genome annotation

Alzheimer’s disease (AD) is a highly heritable brain dementia characterized by substantial failure of cognitive function. Large-scale genome-wide association studies (GWAS) have led to a significant set of SNPs associated with AD and related traits. GWAS hits usually emerge as clusters where a lead SNP with the highest significance is surrounded by other less significant neighboring SNPs. Although functionality is not guaranteed even with the strongest associations in GWASs, lead SNPs have historically been the focus of the field, with the remaining associations inferred to be redundant. Recent deep genome annotation tools enable the prediction of function from a segment of a DNA sequence with significantly improved precision, which allows in silico mutagenesis to interrogate the functional effect of SNP alleles. In this project, we explored the impact of top AD GWAS hits on chromatin functions and whether chromatin functions are altered by the genomic context (i.e., alleles of neighboring SNPs) using a deep learning tool, Expecto. Our results showed that highly correlated SNPs in the same LD block could have distinct impacts on downstream functions. Although some GWAS lead SNPs showed dominant functional effects regardless of the neighborhood SNP alleles, several other SNPs did exhibit enhanced loss or gain of function under certain genomic contexts, suggesting potential additional information hidden in the LD blocks.

## Toolkit Usage

This repository contains all the code used in the process for the above mentioned paper. We have expanded on Expecto model to obtain multiple variants and have modified the code such that anyone with adequate summary statistcs of GWAS data can perform the same analysis as a preliminary step.

### To run the analysis

* Download the repo to your local computer or hpc using git
  ```
  git clone PradoVarathan/Multi_Specto
  ```
* Install the required files for the environment
  ```
  cd Multi_Specto
  virtualenv multispecto
  source env_name/bin/activate
  pip install -r requirements.txt
  ```
  * Docker functionality is available as well but direct installtion is recommended since there has been issues with the contained wrapping in R functionalities for downstream analysis.
* Visit the NCBI website (https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/) to obtain the fasta sequence for Hg38 genome assembly and store it in the working directory under Hg38/hg38.fa
* Edit the MultiSpecto_Run.yml file for preparing the run for all combinations.
  ```
  default:
    input_files:
      model_dict: ./deepsea.beluga.pth 
      primary_dataset: --insert your GWAS dataset --
      secondary_dataset: --insert your updated GWAS dataseet if originial GWAS used a old reference--
      n: 100 # Number of top variants for the analysis
      features_info: ./deepsea_beluga_2002_features.tsv.txt # Features name and information 
    entrez_cred:
      entrez_email: --can be obtained from entrez website for free--
      entrez_api: --can be obtained from entrez website for free--
    output_files:
      h5_folder: --output folder name for the h5 storage files --
      combination_distribution_folder: --folder to store distributions of all the combination dataset--
  ```
* MultiSpecto_Run.py 
  This file is a python executable file requring a config file for documenting the path and output file names for the analysis. Please feel free to download the config file and make the necessary changes.
    ```
  python MultiSpecto_Run.py MultiSpecto_Run.yml
  ```
    
* For post analysis, we have developed few R functionalities to use the pickle file from the above run to plot heatmaps and distributions to discuss results as seen in paper.
  * Similar to the config file above, use the config.yml as template and run the Heatmap Analysis R file as follows
    ```
    R Heatmap_Analysis.R config.yml
    ```


  
