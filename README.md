# 

![Project Image](https://github.com/ACDBio/GCDPipe/blob/main/app_default_assets/gcdbanner_small.png)
> An easy-to-use radnom forest-based tool for risk gene, cell type and drug ranking for complex traits using GWAS-derived genetic evidence.
---
### Warning! Drug ranking produced by the pipeline is for scientific purposes only and should not be used as a guidance for clinical practice.
---

### Table of Contents

- [DESCRIPTION](#description)
- [INSTALLATION](#installation)
- [INPUT](#input)
- [OUTPUT](#output)
- [SETTINGS](#settings)
- [USAGE DEMONSTRATION](#demo)
- [CITATION](#citation)

---

## Description

 - The pipeline is designed to use the data on known risk genes (which can be obtained from GWAS fine-mapping) and expression profiles characterizing cell types/tissues to construct a random forest classifier to distinguish risk genes.   
- A modification of feature importance analysis with SHAP values is used to rank the cell types/tissues based on their importance for risk class assignment.   
- The information on drug gene-targets can then be used to rank the drugs by maximal risk class assignment probability of any of their targets.   

### Pipeline performance checking
- The pipleline was tested on IBD, schizophrenia, Alzheimer's disease. 
- For schizophrenia, it displays ROC-AUC above 0.9 on test gene set in a risk gene discrimination task. In the other case studies, it shows ROC-AUC above 0.8. 
- The risk genes identified with GCDPipe for schizophrenia link dopaminergic synapse, retrograde endocannabinoid signaling, nicotine addiction ad other dysregulations at molecular level; for Alzheimer’s disease it shows diuretics as a leading enriched drug target category, and for IBD it prioritizes TH1/17 and other CD4+ T cells.  
- In the case studies on IBD and schizophrenia, the obtained gene ranking is significantly enriched with drug targets for the corresponding diseases and drug ranking - in the corresponding disease drugs.
  
For further details on the pipeline, see the publication ...  
### General pipeline scheme
![Pipeline Scheme](https://github.com/ACDBio/GCDPipe/blob/main/app_default_assets/gcdpipe_scheme.png)  

[Back To The Top](# )

---
## Installation
  
```shell
# A virtual environment can be created and activated with:
python3 -m venv GCDpipe_env
source GCDpipe_env/bin/activate
# Downloading the code:
git clone https://github.com/ACDBio/GCDPipe.git
# It might be required to upgrade pip with: 
python -m pip install --upgrade pip
# Downloading the required packages: 
pip install -r ./GCDPipe/requirements.txt
# Launching the GCDpipe Dash App:
python ./GCDPipe/GCDPipe.py
# To use GCDPipe interface, open up the link depicted after the phrase 'Dash is running on' in the console. 
```  

[Back To The Top](# )

---
## Input
For gene classification, only first two fields need to be filled. The files to the other fields are uploaded in cases when drug prioritization and its initial quality assessment are required.  

 #### Field 1: Gene Data (a data on risk and non-risk genes used for classifier training and testing)  
 Two types of .csv files can be uploaded in this field:  
 - A file with gene identifiers in the first column and their attribution to risk (1) or non-risk (0) class.  
   
| pipe_genesymbol | is_True  |
| :-----: | :-: |
| ACP5 | 0 |
| GRIN2A | 1 |  
 - A file, specifying locations of the significant loci in GRCh38 coordinates and the corresponding risk genes for each locus. In this case, all genes lying within 500 kbase window around the locus center (which is expected to be a genome-wide significant variant (a 'leading variant') from GWAS), except for the risk ones are considered as 'false', and a gene set for classifier training and testing is constructed in an automated manner. If not variant ID is unknown, an rsid field can be left blank.  
  
| pipe_genesymbol | rsid  |  location  |  chromosome  |
| :-----: | :-: | :-: | :-: |
| NOD2 | rs2066844 | 50712015 | 16 |  
  
#### Field 2: Feature data (expression profiles)
- Here, a .csv file with expression profiles characterizing cell types/tissues of interest can be uploaded. The data can be obtained from a range of publicly available expression atlases and other sources (such as Allen Brain Mep, DropViz, DICE  Immune Cell Atlas, GTEx etc.). These profiles are used as features to build a risk gene classifier. It requires pipe_genesymbol column with gene identifiers and other columns with custom names (for example, names of cell types).
  
| pipe_genesymbol | Exc.L5.6.FEZF2.ANKRD20A1  |  Exc.L5.6.THEMIS.TMEM233  |  Inh.L1.LAMP5.NDNF  |
| :-----: | :-: | :-: | :-: |
| A2M | 4.15 | 3.83 | 0 |  

#### Field 3: Drug-target interaction data
- For drug ranking, the pipeline needs a .csv file with drug identifiers and their gene targets. We assembled an example of such dataset from DGIdb and DrugCentral, which can be found [here](https://github.com/ACDBio/GCDPipe/blob/main/app_input_examples/drug_targets_data.csv). The field with drug IDs is expected to be named 'DRUGBANK_ID' and the field with genes - 'pipe_genesymbol'.  
  
| DRUGBANK_ID | pipe_genesymbol  |
| :-----: | :-: |
| DB05969 | CDK7 |  
| DB01054 | ADORA3 |
| DB01054 | TTR |

#### Field 4: Target drug category Drugbank IDs
- In cases when there exists a list of the drugs with desired functions (drugs for treatment of a disease of interest), it can be uploaded here for  initial quality assessment of gene and drug ranking. The .csv file is expected with one column - DRUGBANK_ID.  
  
| DRUGBANK_ID |
| :-----: |
| DB00321 |
| DB01238 |
| DB14185 |

[Back To The Top](# )

---
## Output  
The pipeline gives the ROC curve, ROC-AUC and a range of classifier performance metrics for the obtained classifier (the metrics are calculated from testing on a test set of genes automatically generated within the pipeline).  
In addition, it provides a range of output files:  
#### Output file 1: Gene Classification Results
- A .csv file can be downloaded with information on gene attribution to a risk class by the obtained classifier with the probability threshold corresponding to maximal difference between tpr and fpr on the ROC ('is_risk_class' field), probabilities of the genes to be assigned to this class ('score' field) and information on whether the gene was considered as the risk one in the original training-testing set ('is_input_risk_gene' field).  
  
| pipe_genesymbol | score | is_risk_class | is_input_risk_gene |
| :-----: | :-: | :-: | :-: |
| PPP2R2B | 0.939833072 | TRUE | 0 |
| CALM2 | 0.939833072 | TRUE | 0 |
| CACNA1C | 0.936927967 | TRUE | 1 |  
  
#### Output file 2: Expression profile (cell type/tissue) ranking
- The pipeline returns ranking of features (expression profiles) used for classifier training (characterizing cell types/tissues) based on correlation between SHAP values for the risk class and values of these features (expression intensity in these profiles).  
  
| expression_profile | importance_based_score |
| :-----: | :-: |
| Inh.L4.5.PVALB.TRIM67 | 0.971200485 |
  
#### Output file 3: Drug ranking
- If drug ranking is performed, the pipeline gives the file, in which drugs are scored by maximal probability of any of their gene-target to be assigned to the risk class ('score' field). The drugs, which have at least one gene-target attributed to the risk class at default probability threshold (corresponding to maximal difference between tpr and fpr) are marked in the field 'has_risk_class_targets'.  
  
| DRUGBANK_ID | score | has_risk_class_targets |
| :-----: | :-: | :-: |
| DB06288 | 0.939833072 | TRUE |
 
In addition, if a drug set of interest is provided, GCDPipe can run two Mann-Whitney tests: comparing the gene scores of all targets of the given drugs with those of other genes and comparing their drug scores with scores of all other drugs. Then it shows the corresponding p-values, U statistics and displays boxplots and density plots for the compared distributions.

[Back To The Top](# )

---
## Settings  
The pipeline interface allows to define a range of settings for classifier generation:  
- Number of estimators: decision tree count in a random forest can be changed.
- Testing gene set/training gene set ratio: a fraction of the genes from the training-testing set to be used for testing.
- Max tree depth - a specific maximal depth can be set.
- Number of samples per leaf: up to 10 different values can be set for hyperparameter search.
- Min number of samples required to split an internal node: up to 10 different values can be set for hyperparameter search.

[Back To The Top](# )

---  
## Demo  
[USAGE DEMONSTRATION](https://github.com/ACDBio/GCDPipe/blob/main/app_default_assets/demo.gif)


[Back To The Top](# )

---  
## Citation  
  
 

[Back To The Top](# )

---
