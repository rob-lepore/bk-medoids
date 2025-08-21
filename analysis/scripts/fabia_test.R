library(fabia)
library(jsonlite)

datasets <- list("trend/0","trend/1","trend/2", "trend/3")

all_found <- list()
ds_id <- 0
for (ds in datasets){

  X <- as.matrix(read.table(paste0("../ARBic_data/3.six_type/",ds), header = TRUE, sep = "\t", row.names = 1))
  
  fab <- fabia(X)
  ext <- extractBic(fab)
  
  bics <- lapply(seq_len(6), function(i) {
    list(
      dataset_id = ds_id,
      bicluster_id = i - 1,
      rows = ext$bic[i,]$bixn,
      cols = ext$bic[i,]$biypn
    )
  })
  all_found <- c(all_found, bics)
  ds_id <- ds_id + 1
}


write_json(all_found, "../results/comparison/fabia/trend.json", pretty = TRUE)