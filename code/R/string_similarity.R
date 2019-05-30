library(data.table)
library(stringdist)

census_file <- '~/dot-nlp/data/clean/census_occ_counts.csv'
dot_file <- '~/dot-nlp/data/clean/structured_1977.csv'

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
  sim_vector <- 1-stringdist(occ,dot_occs,method='jw',p=p)
  sim_titles <- sim_vector >= threshold
  sim_titles <- dot_occs[sim_titles]
  return(sim_titles)
}

match <- function(unmatched_occs,threshold=0.9){
  occ_matches <- data.table('Census Occupation'=character(),'DOT Match'=character())
  i <- 1
  for (occ in unmatched_occs){
    if (i %% 1000 == 0){
      print(i)
    }
    sim_vector <- 1-stringdist(occ,dot_occs,method='jw',p=0.1)
    sim_titles <- sim_vector >= threshold
    if (sum(sim_titles) == 0){
      sim_title <- NA
    }else{
      sim_title <- dot_occs[sim_titles][1]
    }
    occ_matches <- rbind(occ_matches,list('Census Occupation'=occ,'DOT Match'=sim_title))
    i <- i+1
  }
  fwrite(occ_matches,'~/dot-nlp/data/clean/occ_matches.csv')
}

dot <- load_dot()
cens <- fread(census_file)
unmatched_occs <- cens[!(Title %in% dot$Title),.N,by=Title]$Title
match(unmatched_occs,0.95)
