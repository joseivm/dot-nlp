library(data.table)
library(stringdist)

census_file <- '~/Projects/dot-nlp/data/clean/census_occs.csv'
dot_file <- '~/Projects/dot-nlp/data/clean/structured_1977.csv'

load_dot <- function(){
  dot <- fread(dot_file)
  dot[, Title := tolower(Name)]
  dot <- dot[,.(Name,Title,Code)]
  return(dot)
}

load_cens_occs <- function(){
  cens <- fread(census_file)
  cens_occs <- cens[,.N,by=Title]$Title
  return(cens_occs)
}

get_similar_titles <- function(occ,threshold=0.8,p=0.1){
  sim_vector <- 1-stringdist(occ,cens_occs,method='jw',p=p)
  sim_titles <- sim_vector >= threshold
  sim_titles <- cens_occs[sim_titles]
  return(sim_titles)
}

cens_occs <- load_cens_occs()
dot <- load_dot()
