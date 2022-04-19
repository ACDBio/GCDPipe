library(dplyr)
library(fgsea)
library(tibble)
library(tidyverse)
library(org.Hs.eg.db)
library(fgsea)
library(ggplot2)
library(clusterProfiler)

setwd('/home/biorp/Gitrepos/GCDPipe_Deployment/results_analysis_IBD')

#----Analysis settings----
target_disease_drug_category_name<-'crohns_uc' #name of the drug category of interest for the analysis (is used in atc_drugs and atc_targets file)
topk_disesae_dataset_genes=200 #Number of top ranking genes to subset from disease-specific datasets for overrepresentation analysis
topk_model_genes=500 #Number of top ranking genes to subset from gene scoring obtained with the classifier
drugcats_for_comparison=c('schizophrenia','cad_and_chd', 'asthma','crohns_uc')

#----Loading and preprocessing the files----
load('./data/atc_drugs.rdata') #Drug lists by ATC categories from DrugBank
load('./data/atc_targets.rdata') #Gene targets for drugs with Drugbank IDs from DGIDB and DrugCentral
load('./data/full_drugdb.rdata') #Dataset assembled from Drug Central and DGIDB with drugs and their gene targets
full_drugdb<-full_drugdb %>% 
  dplyr::select(DRUGBANK_ID,pipe_genesymbol) %>% 
  filter(DRUGBANK_ID!='') %>% 
  distinct()

disease_datasets_path<-'./data/disease_datasets/fully_processed' #Path to datasets with differential analysis results (with two columns - with gene symbols and with the metric calculated as -log10(adj. pval)*logFC and for SCHEMA data a metric is calculted as -log10(Q meta); gene symbols are mapped to HUGO
tfset<-read.csv('./pipeline_data/IBD_mapped_tf_geneset.csv') #path to the training-testing set used for the classifier training
generank<-read.csv('./pipeline_data/gene_scores.csv') #gene probabilities to be assigned to the risk class
cellrank<-read.csv('./pipeline_data/feature_scores.csv') #expression profile ranking

#Subsetting the genes used for training the classifier
gwas_truegenes<-tfset %>% 
  filter(is_True==1) %>% 
  dplyr::select(pipe_genesymbol) %>% 
  pull()

#Marking the genes used for classifier training in the dataframe with gene scores
generank<-generank %>% 
  mutate(gwas_truegene=pipe_genesymbol %in% gwas_truegenes)

#Filtering the genes used for classifier training
generank_filtered<-generank %>% 
  filter(gwas_truegene==FALSE)

generank_filtered_topgenes <-generank_filtered%>% 
  slice_max(n=topk_model_genes, order_by=score,with_ties=FALSE) %>%
  dplyr::select(pipe_genesymbol) %>% 
  distinct() %>% 
  pull()




#----Plot A. Enrichment (over-representation) analysis with disease-specific datasets----

disease_genesets<-c()
generank_geneset<-list(generank_filtered_topgenes)
names(generank_geneset)<-'generank_filtered_genes'
disease_genesets<-c(disease_genesets, generank_geneset)
generank_geneset<-list(generank_filtered_topgenes[1:5])
names(generank_geneset)<-'generank_filtered_genes_not_complete'
disease_genesets<-c(disease_genesets, generank_geneset)


difexp_gene_clusters<-list()
gene_universes<-list()

for (file in list.files(path = disease_datasets_path)){
  gsea<-read.csv(paste(disease_datasets_path,file, sep='/'), header=FALSE)
  
  data_name<-word(file, 1, sep='.csv')
  ranked_list<-gsea$V2
  names(ranked_list)<-gsea$V1
  
  listed_ranked_list<-list(ranked_list)
  names(listed_ranked_list)<-data_name
  gene_universes=c(gene_universes,listed_ranked_list)
  
  srted<-sort(abs(ranked_list), decreasing = TRUE)
  sel<-names(head(srted, topk_disesae_dataset_genes))
  sel<-list(sel)
  names(sel)<-data_name
  difexp_gene_clusters<-c(difexp_gene_clusters, sel)
}


fora_reslist=list()

for (i in c(1:length(difexp_gene_clusters))){
  setname<-names(difexp_gene_clusters[i])
  gene_cluster<-difexp_gene_clusters[[i]]
  universe<-names(gene_universes[[i]])
  fora_res<-list(tibble(fora(pathways=disease_genesets, genes=gene_cluster, universe=universe)) %>% 
                  mutate(gene_set_top200=setname))
  fora_reslist<-c(fora_reslist,fora_res)
  
}

fora_difexp_results<-dplyr::bind_rows(fora_reslist)
fora_difexp_results<-fora_difexp_results %>% 
  filter(pathway %in% c('generank_filtered_genes'))


res.plotA.fora_difexp_results<-fora_difexp_results %>% 
  arrange(padj) %>% 
  dplyr::select(gene_set_top200, padj) %>% 
  mutate(neg_log10_padj=-log10(padj)) %>% 
  mutate(signif=padj<0.05) %>%
  ggplot(aes(x=reorder(gene_set_top200, neg_log10_padj), y=neg_log10_padj, fill=signif)) +
  geom_bar(stat='identity')+
  coord_flip()+
  theme_classic()+
  scale_fill_manual(values=c('#8F3985',"#07BEB8"))+
  xlab('Dataset')+
  ylab('-log10(Padj.)')+
  labs(fill="Statistically significant enrichment")
res.plotA.fora_difexp_results
  
#----Plot B. Expression profile ranking----

res.plotB.cellranks_plot<-cellrank %>% 
  rename(cell_type=expression_profile,corr_based_score=importance_based_score) %>% 
  arrange(desc(corr_based_score)) %>% 
  top_n(15) %>% 
  ggplot(aes(x=reorder(cell_type, corr_based_score), y=corr_based_score))+
  geom_bar(stat='identity', fill='#ADB1BA')+
  coord_flip()+
  theme_classic()+
  theme(text=element_text(size=17),
        axis.text=element_text(size=17, angle = 0))+
  xlab('Expression profile')+
  ylab('Correlation-based score')
res.plotB.cellranks_plot


#----Plot C. Enrichment of gene profile with drug targets----  
gene_vector<-generank_filtered$score
names(gene_vector)<-generank_filtered$pipe_genesymbol

#adding random noise
additions=runif(length(names(gene_vector)), 0, 100)/10000000000
gene_vector=gene_vector+additions

res.targets_fgsea=fgsea(atc_targets[drugcats_for_comparison], gene_vector)
res.targets_enrichment_plot=plotEnrichment(atc_targets[[target_disease_drug_category_name]], gene_vector)

res.plotC.drug_target_enrichmentplot<-res.targets_enrichment_plot+
  theme_classic()+
  geom_line(size=1, color='#07BEB8')+
  xlab('Gene rank by probability to be assigned to a risk class')+
  ylab('Enrichment score')+
  theme(text=element_text(size=15))
res.plotC.drug_target_enrichmentplot
  
  

#----Plot D. Comparison of enrichment for the selected categories----

res.plotD.drug_target_enrichment_comparison<-res.targets_fgsea %>% 
  mutate(log10padj=-log10(padj)) %>% 
  mutate(signif=log10padj>-log10(0.05)) %>% 
  ggplot(aes(x=reorder(pathway,log10padj), y=log10padj, fill=signif))+
  geom_bar(stat='identity')+
  coord_flip()+
  theme_classic()+
  scale_fill_manual(values=c('#8F3985',"#07BEB8"))+
  xlab('Drug set')+
  ylab('-log10(Padj.)')+
  labs(fill="Statistically significant enrichment")+
  theme(text=element_text(size=20))+
  scale_x_discrete(labels=c('Schizophrenia','CAD and CHD','Asthma','IBD'))

#----Plot E. Drug enrichment----
drugdata_full<-merge(full_drugdb,generank) %>% 
  select(-gwas_truegene) %>% 
  drop_na()

drugdata<-drugdata_full%>% 
  group_by(DRUGBANK_ID) %>%
  summarise_at('score', .funs='max')

drug_vector<-drugdata$score
names(drug_vector)<-drugdata$DRUGBANK_ID

additions<-runif(length(names(drug_vector)), 0, 100)/10000000000
drug_vector<-drug_vector+additions

res.drugs_fgsea<-fgsea(atc_drugs, drug_vector)
res.drugs_enrichment_plot<-plotEnrichment(atc_drugs[[target_disease_drug_category_name]], drug_vector)
res.plotE.drug_enrichmentplot<-res.drugs_enrichment_plot+
  theme_classic()+
  geom_line(size=1, color='#07BEB8')+
  xlab('Drug rank by max target gene probability to be in a risk class')+
  ylab('Enrichment score')+
  theme(text=element_text(size=15))
res.plotE.drug_enrichmentplot

res.drugs_fgsea %>% 
  filter(pathway=='crohns_uc')

#----Plot F. Comparison of gene scores for targets of disease drugs and random drug sample of the same size----
random_drug_sample_selection=sample(unique(drugdata_full$DRUGBANK_ID), length(atc_drugs[[target_disease_drug_category_name]]))

drug_group_scores<-drugdata_full %>% 
  group_by(DRUGBANK_ID) %>% 
  summarise_at('score', .funs='max') %>% 
  mutate(arandom_drug_sample=DRUGBANK_ID %in% random_drug_sample_selection) %>% 
  mutate(ctarget_disease=DRUGBANK_ID %in% atc_drugs[[target_disease_drug_category_name]]) %>% 
  mutate(bmuscle_relaxants=DRUGBANK_ID %in% atc_drugs$MUSCLE_RELAXANTS_M03) %>% 
  dplyr::select(score, arandom_drug_sample, ctarget_disease, bmuscle_relaxants) %>% 
  gather(-score, key='drug_group', value='is_present') %>% 
  filter(is_present==TRUE)

my_comparisons<-list(c('arandom_drug_sample','ctarget_disease'), c('bmuscle_relaxants','ctarget_disease'), c('bmuscle_relaxants', 'arandom_drug_sample'))
comparison_res=compare_means(score ~ drug_group,  data = drug_group_scores,
                             method = "wilcox.test")

res.plotF.drugcat_target_probs_comparison<-drug_group_scores %>% 
  ggplot(aes(x=drug_group, y=score, fill=drug_group))+
  geom_boxplot()+
  stat_compare_means(comparisons = my_comparisons, size=7, method = "wilcox.test", label='p.signif')+
  theme_classic()+
  scale_fill_manual(values = c('#808180FF','#8F3985','#07BEB8'))+
  xlab('Drug group')+
  ylab('Max target gene probability to be assigned to the risk class')+
  stat_compare_means(label.y = 0.58, size=6.5)+
  theme(text=element_text(size=16.5),
        axis.text=element_text(size=15, angle = 0))+
  labs(fill='Drug group')+
  scale_x_discrete(labels=c('Random drug sample','Muscle relaxants','IBD drugs'))

res.plotF.drugcat_target_probs_comparison

