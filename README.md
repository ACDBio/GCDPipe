# 

![Project Image](https://github.com/ACDBio/GCDPipe/blob/main/app_default_assets/gcdbanner_small.png)
> An easy-to-use radnom forest-based tool for risk gene, disease-relevant cell type and drug ranking for complex traits using GWAS-derived genetic evidence.
---

### Table of Contents

- [DESCRIPTION](#description)
- [INSTALLATION](#installation)
- [INPUT](#input)
- [SETTINGS](#settings)
- [OUTPUT](#output)
- [USAGE EXAMPLE](#example)
- [CITATION](#cite)

---

## Description

 - The pipeline is designed to use the data on known risk genes (which can be obtained from GWAS fine-mapping) and expression profiles characterizing cell types/tissues to construct a random forest classifier to distinguish risk genes.   
- A modification of feature importance analysis with SHAP values is used to rank the cell types/tissues by their importance to risk class assignment.   
- The information on drug gene-targets can then be used to rank the drugs by maximal risk class assignment probability of any of their targets.   

### Pipeline performance checking
- The pipleline was tested on IBD, schizophrenia, Alzheimer's disease. 
- For schizophrenia, it displays ROC-AUC above 0.9 on test gene set in a risk gene discrimination task.
- The risk genes identified with GCDPipe for schizophrenia link dopaminergic synapse, retrograde endocannabinoid signaling, nicotine addiction ad other dysregulations at molecular level; for Alzheimer’s disease it shows diuretics as a leading enriched drug target category, and for IBD it prioritizes TH1/17 and other CD4+ T cells.  
- In the case studies on IBD and schizophrenia, the obtained gene ranking is significantly enriched with drug targets for the corresponding diseases and drug ranking - in the corresponding disease drugs.
  
For further details on the pipeline, see the publication ...  
### General pipeline scheme
![Pipeline Scheme](https://github.com/ACDBio/GCDPipe/blob/main/app_default_assets/gcdpipe_scheme.png)  
[Back To The Top](#read-me-template)
