library(ggplot2)
library(data.table)
library(ggthemes)

figures_dir <- '~/dot-nlp/figures/'
predictions_dir <- '~/dot-nlp/output/joint'

make_filename <- function(figure_name){
  return(paste0(figures_dir,figure_name,'.png'))
}

model_predictions <- function(model){
  pred_df <- fread(file.path(predictions_dir,model))
  true_df <- copy(pred_df)

  pred_df[, DPT := as.character(DPT)]
  pred_df[as.numeric(DPT) < 100, DPT := paste0('0',DPT)]
  pred_df[, Data := substr(DPT,1,1)]
  pred_df[Data %in% c('7','8','9'), Data := '6']
  pred_df[, People := substr(DPT,2,2)]
  pred_df[, Things := substr(DPT,3,3)]
  pred_df[Things == '8', Things := '7']
  pred_df[, Source := 'Predicted']

  true_df[, Data := substr(Code,5,5)]
  true_df[Data %in% c('7','8','9'), Data := '6']
  true_df[, People := substr(Code,6,6)]
  true_df[, Things := substr(Code,7,7)]
  true_df[Things %in% c('8','9'), Things := '7']
  true_df[, Source := 'True']

  all <- rbind(pred_df,true_df)
  return(all)
}

get_accuracy_df <- function(model_df){
  df <- melt(model_df,id.vars=c('Code','Source'))
  df[, value := as.numeric(value)]
  df <- dcast(df[variable != 'Title'],Code ~ variable + Source)
  df[, DataAccuracy := Data_Predicted == Data_True]
  df[, ThingsAccuracy := Things_Predicted == Things_True]
  return(df)
}

model_histograms <- function(df,model_name){
  attributes <- c('Data','People','Things')
  for (attribute in attributes){
    filename <- make_filename(paste0(model_name,'_',attribute,'_Hist'))
    ggplot(df,aes_string(attribute,fill='Source'))+geom_bar(position='dodge')+theme_gdocs()+
    scale_fill_manual(values=c('darkred','skyblue4'))
    ggsave(filename,width=6,height=5)
  }
}
