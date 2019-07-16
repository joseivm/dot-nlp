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

get_similar_titles <- function(occ,occ_list,threshold=0.8,p=0.1){
  sim_vector <- 1-stringdist(occ,occ_list,method='jw',p=p)
  sim_titles <- sim_vector >= threshold
  sim_titles <- occ_list[sim_titles]
  return(sim_titles)
}

match <- function(unmatched_occs,occ_list,threshold=0.9){
  occ_matches <- data.table('Census Occupation'=character(),'DOT Match'=character())
  i <- 1
  for (occ in unmatched_occs){
    if (i %% 1000 == 0){
      print(i)
    }
    sim_vector <- 1-stringdist(occ,occ_list,method='jw',p=0.1)
    sim_titles <- sim_vector >= threshold
    if (is.na(sim_titles) | sum(sim_titles) == 0){
      sim_title <- NA
    }else{
      sim_title <- occ_list[sim_titles][1]
    }
    occ_matches <- rbind(occ_matches,list('Census Occupation'=occ,'DOT Match'=sim_title))
    i <- i+1
  }
  fwrite(occ_matches,'~/dot-nlp/data/clean/occ_matches.csv')
}

census_file <- '~/dot-nlp/data/clean/census_occ_counts.csv'
census_stats <- fread(census_file)
census_stats[, Title := tolower(occstr)]
census_occ_counts <- census_stats[,.(N=sum(count)),by=Title]
census_occ_counts <- census_occ_counts[order(-N)]
dot <- fread('dot-nlp/data/raw/1939_structured.csv')
dot[, Title := gsub("\\(.+\\)",'',job_title)]
dot[, Title := gsub("\\.",'',Title)]
dot[, Title := trimws(Title)]
dot[, Title := tolower(Title)]
all_titles <- c()
for (title in dot$Title){
  sub_titles <- unlist(strsplit(title,";"))
  sub_titles <- trimws(sub_titles)
  all_titles <- c(all_titles,sub_titles)
}

dot <- load_dot()
cens <- fread(census_file)
unmatched_occs <- cens[!(Title %in% dot$Title),.N,by=Title]$Title
dot_occs <- dot$Title
match(unmatched_occs,dot_occs,0.95)
