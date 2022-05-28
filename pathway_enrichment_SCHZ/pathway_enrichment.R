library(dplyr)
library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)

n_topgenes<-500
gene_ranking<-read.csv('gene_scores.csv')
gene_ranking
traintest<-read.csv('tfset.csv')

gene_ranking<-left_join(gene_ranking,traintest)
gene_ranking[is.na(gene_ranking)]<-0
  
gene_ranking_notraintest_top<-gene_ranking %>% 
  filter(is_True==0) %>% 
  top_n(n=n_topgenes, wt=score) %>% 
  dplyr::select(pipe_genesymbol) %>% 
  pull()

gene_ranking_traintest<-gene_ranking %>% 
  filter(is_True==1) %>% 
  dplyr::select(pipe_genesymbol) %>% 
  pull()

gene_ranking_all_top<-gene_ranking %>% 
  top_n(n=n_topgenes, wt=score)%>% 
  dplyr::select(pipe_genesymbol) %>% 
  pull()



gene_sets<-c()

gene_sets$risk_genes_excl_tt<-bitr(gene_ranking_notraintest_top, fromType="SYMBOL", 
                             toType = "ENTREZID",
                             OrgDb = org.Hs.eg.db) %>% 
  dplyr::select(ENTREZID) %>% 
  pull()

gene_sets$risk_genes_incl_tt<-bitr(gene_ranking_all_top, fromType="SYMBOL", 
                                    toType = "ENTREZID",
                                    OrgDb = org.Hs.eg.db) %>% 
  dplyr::select(ENTREZID) %>% 
  pull()

gene_sets$tt_risk_genes<-bitr(gene_ranking_traintest, fromType="SYMBOL", 
                              toType = "ENTREZID",
                              OrgDb = org.Hs.eg.db) %>% 
  dplyr::select(ENTREZID) %>% 
  pull()





cluster_comparison<-compareCluster(gene_sets[c('risk_genes_excl_tt','tt_risk_genes')], fun="enrichKEGG",
                                   organism="hsa", pvalueCutoff=0.05)
kegg_enrichment<-enrichKEGG(gene_sets$risk_genes_incl_tt)


kegg_compareclusterres <- setReadable(cluster_comparison, 'org.Hs.eg.db', 'ENTREZID')
kegg_compareclusterres_df<-kegg_compareclusterres@compareClusterResult
kegg_compareclusterres_df %>% 
  write.csv('pathway_enrichment_results.csv', quote=FALSE, row.names = FALSE)
kegg_compareclusterres_pt <- pairwise_termsim(kegg_compareclusterres)     
#----Visualization----
emapplot(kegg_compareclusterres_pt, layout="kk")
dotplot(kegg_compareclusterres, showCategory = 20)
cnetplot(kegg_compareclusterres)
upsetplot(kegg_enrichment)
