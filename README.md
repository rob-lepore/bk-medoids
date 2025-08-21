# BK-medoids

BK-medoids is a generalization of the traditional k-medoids clustering method to the biclustering setting.

This is the codebase produced for my Master Thesis in Bioinformatics (University of Bologna), called "Advanced biclustering in genomic data analysis: review of techniques and new approaches", developed at the University Pablo de Olavide
(Sevilla, Spain).

Currently, BK-medoids can be executed with two distance measures: a **shift-detecting** measure (to identify constant rows/columns and shifting biclusters), and a **scale-detecting** one (to identify constant rows/columns and scaling biclusters).

## Contents of this repository

This repository contains the scripts used to implement the algorithm and to run the analysis presented in the Thesis.

BK-medoids implementation:
- `bkmedoids.py`: definition of the class BKmedoids, which implements the core steps of the algorithm.
- `medoid.py`: definition of the class Medoid. Objects of this class are used as bicluster representatives and contain the rows and columns of the corresponding bicluster.
- `utils.py`: definition of distance metrics and scoring functions. 

Analysis: 
- `analysis/arbic_analysis.py`: computes statistics about the ARBic's performance on the ARBic datasets.
- `analysis/bicgen.py`: definition of the class BicGen, which can build biclusters with desired patterns or whole datasets from a JSON file outputted by GBic.
- `analysis/bkmedoids_analysis.py`: runs and extracts statistics about the performance of BK-medoids on the ARBic datasets.
- `analysis/fabia_analysis.py`: computes statistics about the FABIA's performance on the ARBic datasets.
- `analysis/generate_dataset.py`: generates 100 datasets (25 per pattern) from GBic JSON files.  
- `analysis/graphics.py`: utility function to visulize biclusters and biclustering solutions.
- `analysis/gridsearch.py`: definition of the class GridSearch, used to run a grid search of the optimal parameters of BK-medoids on a given dataset. 
- `analysis/realdata_analysis.py`: GO and KEGG enrichment analysis of the results of BK-medoids on the *E. coli* and yeast datasets
- `analysis/realdata_test.py`: runs BK-medoids on the *E. coli* and yeast datasets.
- `analysis/test.py`: runs model selection and test evaluation of BK-medoids on the synthetic datasets.
- `scripts/`: contains scripts used to run FABIA and ARBic on the ARBic datasets.

The release contains all datasets used and results obtained in the analysis.
