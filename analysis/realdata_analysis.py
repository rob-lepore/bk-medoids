from gridsearch import GridSearch
from bkmedoids import BKmedoids
import numpy as np
import pandas as pd
from utils import *
from graphics import *
import json
import joblib

spec = "yeast"

with open(f"results/real_data/{spec}.json") as f:
    bics = json.load(f)
    
rows = bics["rows"]
cols = bics["cols"]

go_dict = pd.read_csv(f"real_datasets/data/{spec}_dream5/gsets/go_filtered.tsv", sep="\t", header=0, index_col=0)
kegg_dict = pd.read_csv(f"real_datasets/data/{spec}_dream5/gsets/kegg_filtered.tsv", sep="\t", header=0, index_col=0)


# Assume go_dict is your DataFrame (genes x GO terms, bool)
N = go_dict.shape[0]  # total number of genes

import pandas as pd
from scipy.stats import fisher_exact
import statsmodels.stats.multitest as smm

def enrichment_results(gene_list, go_df, kegg_df, fdr_threshold=0.05):
    """
    gene_list: list of gene IDs (must match the index of go_df / kegg_df)
    go_df: DataFrame of bool (genes x GO terms)
    kegg_df: optional, DataFrame of bool (genes x KEGG pathways)
    fdr_threshold: cutoff to report significant terms
    """
    genes_in_list = set(gene_list)
    all_genes = set(go_df.index)
    
    # --- GO enrichment ---
    results = []
    for go_term in go_df.columns:
        term_genes = set(go_df.index[go_df[go_term]])
        k = len(term_genes & genes_in_list)  # hits in bicluster
        K = len(term_genes)                  # total genes with term
        n = len(genes_in_list)
        N = len(all_genes)
        
        # Fisher's exact test
        table = [[k, n - k],
                 [K - k, N - K - (n - k)]]
        _, pval = fisher_exact(table, alternative='greater')
        
        fold_enr = (k/len(genes_in_list)) / (K/len(all_genes))
        results.append({'GO_term': go_term, 'k': k, 'K': K, 'pval': pval, "fold": fold_enr})
    
    go_res = pd.DataFrame(results)
    go_res['FDR'] = smm.multipletests(go_res['pval'], method='fdr_bh')[1]
    go_res = go_res[go_res['FDR'] <= fdr_threshold].sort_values('fold', ascending=False)
    
    print("=== GO Enrichment ===")
    if go_res.empty:
        print("No significant GO terms")
    else:
        print(go_res)
    
    # --- KEGG enrichment ---
    if kegg_df is not None:
        kegg_results = []
        for pathway in kegg_df.columns:
            term_genes = set(kegg_df.index[kegg_df[pathway]])
            k = len(term_genes & genes_in_list)
            K = len(term_genes)
            n = len(genes_in_list)
            N = len(all_genes)
            
            table = [[k, n - k],
                     [K - k, N - K - (n - k)]]
            _, pval = fisher_exact(table, alternative='greater')
            fold_enr = (k/len(genes_in_list)) / (K/len(all_genes))
            kegg_results.append({'KEGG_pathway': pathway, 'k': k, 'K': K, 'pval': pval, 'fold': fold_enr})
        
        kegg_res = pd.DataFrame(kegg_results)
        kegg_res['FDR'] = smm.multipletests(kegg_res['pval'], method='fdr_bh')[1]
        kegg_res = kegg_res[kegg_res['FDR'] <= fdr_threshold].sort_values('fold', ascending=False)
        
        print("\n=== KEGG Enrichment ===")
        if kegg_res.empty:
            print("No significant KEGG pathways")
        else:
            print(kegg_res)

# Example usage:
# enrichment_results(list_of_genes, go_dict, kegg_dict)


for i in range(10):
    genes = [f"G{r}" for r in rows[i]]
    print("\n\nBicluster ", i, " - ", len(rows[i]), "x", len(cols[i]))
    enrichment_results(genes, go_dict, kegg_dict)

