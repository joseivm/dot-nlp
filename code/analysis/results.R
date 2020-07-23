library(data.table)
library(ggplot2)
library(ggpubr)
source('.env.R')

# Input files/dirs
preds_dir <- paste0(PROJECT_DIR,'/output/Full/')
data_dir <- paste0(PROJECT_DIR,'/data/')

# Output files/dirs
results_dir <- paste0(PROJECT_DIR,'/results/Full/')

make_histograms <- function(year){
  filename <- paste0(preds_dir,'domain_1965md_',year,'_full_data_preds.csv')
  df <- fread(filename)
  df <- df[!is.na(pred_DPT) & !is.na(GED)]
  df[, Data := as.numeric(substr(DPT,2,2))]
  df[, People := as.numeric(substr(DPT,3,3))]
  df[, Things := as.numeric(substr(DPT,4,4))]

  df[, pred_Data := as.numeric(substr(pred_DPT,2,2))]
  df[, pred_People := as.numeric(substr(pred_DPT,3,3))]
  df[, pred_Things := as.numeric(substr(pred_DPT,4,4))]

  df[,c('pred_DPT','DPT','Title','Industry','Definition','Code','Attr') := NULL]
  df[, ID := seq.int(nrow(df))]

  df <- melt(df,id.vars='ID')
  df[variable %like% 'pred_', Source := 'Predicted']
  df[is.na(Source), Source := 'Actual']
  df[variable %like% 'pred_', variable := substring(variable,6)]

  vars <- c('Data','People','Things','DCP','STS','GED','SVP','FingerDexterity','EHFCoord')
  plot_list <- list()
  i <- 1
  for (var in vars){
    p <- ggplot(df[variable == var],aes(value,group=Source,fill=Source))+
          geom_bar(aes(y=(..count..)/sum(..count..)),position = 'dodge',width=0.4)+
          theme_bw()+scale_fill_manual(values=c('darkred','skyblue4'))+
          labs(y='Density',x=paste(var,'Score'))
    plot_list[[i]] <- p
    i <- i+1
  }
  ggarrange(plotlist=plot_list,ncol=3,nrow=3,common.legend = TRUE,legend='bottom')
  plot_filename <- paste0(results_dir,year,'_histograms.png')
  ggsave(plot_filename,width=12,height=10)
}

make_1965_histograms <- function(){
  filename <- paste0(preds_dir,'domain_1965md_1965md_full_data_preds.csv')
  tst_df <- fread(paste0(data_dir,'Attr/1965/test.csv'))
  df <- fread(filename)
  df <- merge(df,tst_df[,.(Title,Industry)],by=c('Title','Industry'))
  df <- df[!is.na(pred_DPT)]
  df[, Data := as.numeric(substr(DPT,2,2))]
  df[, People := as.numeric(substr(DPT,3,3))]
  df[, Things := as.numeric(substr(DPT,4,4))]

  df[, pred_Data := as.numeric(substr(pred_DPT,2,2))]
  df[, pred_People := as.numeric(substr(pred_DPT,3,3))]
  df[, pred_Things := as.numeric(substr(pred_DPT,4,4))]

  df[,c('pred_DPT','DPT','Title','Industry','Definition','Code','Attr') := NULL]
  df[, ID := seq.int(nrow(df))]

  df <- melt(df,id.vars='ID')
  df[variable %like% 'pred_', Source := 'Predicted']
  df[is.na(Source), Source := 'Actual']
  df[variable %like% 'pred_', variable := substring(variable,6)]

  vars <- c('Data','People','Things','DCP','STS','GED','SVP','FingerDexterity','EHFCoord')
  plot_list <- list()
  i <- 1
  for (var in vars){
    p <- ggplot(df[variable == var],aes(value,group=Source,fill=Source))+
          geom_bar(aes(y=(..count..)/sum(..count..)),position = 'dodge',width=0.4)+
          theme_bw()+scale_fill_manual(values=c('darkred','skyblue4'))+
          labs(y='Density',x=paste(var,'Score'))
    plot_list[[i]] <- p
    i <- i+1
  }
  ggarrange(plotlist=plot_list,ncol=3,nrow=3,common.legend = TRUE,legend='bottom')
  plot_filename <- paste0(results_dir,'1965_histograms.png')
  ggsave(plot_filename,width=12,height=10)
}

make_1939_histograms <- function(){
  filename <- paste0(preds_dir,'domain_1965md_1939md_full_data_preds.csv')
  df <- fread(filename)
  df <- df[!is.na(pred_DPT)]

  df[, pred_Data := as.numeric(substr(pred_DPT,2,2))]
  df[, pred_People := as.numeric(substr(pred_DPT,3,3))]
  df[, pred_Things := as.numeric(substr(pred_DPT,4,4))]

  df[,c('pred_DPT','Title','Industry','Definition','Code','Original_Definition','pred_exists_bool','merged_bool') := NULL]
  df[, ID := seq.int(nrow(df))]

  df <- melt(df,id.vars='ID')
  df[variable %like% 'pred_', Source := 'Predicted']
  df[is.na(Source), Source := 'Actual']
  df[variable %like% 'pred_', variable := substring(variable,6)]

  vars <- c('Data','People','Things','DCP','STS','GED','SVP','FingerDexterity','EHFCoord')
  plot_list <- list()
  i <- 1
  for (var in vars){
    p <- ggplot(df[variable == var],aes(value,group=Source,fill=Source))+
          geom_bar(aes(y=(..count..)/sum(..count..)),position = 'dodge',width=0.4)+
          theme_bw()+scale_fill_manual(values=c('skyblue4'))+
          labs(y='Density',x=paste(var,'Score'))
    plot_list[[i]] <- p
    i <- i+1
  }
  ggarrange(plotlist=plot_list,ncol=3,nrow=3,common.legend = TRUE,legend='bottom')
  plot_filename <- paste0(results_dir,'1939_histograms.png')
  ggsave(plot_filename,width=12,height=10)
}

make_histograms('1977')
make_histograms('1991')
make_1965_histograms()
make_1939_histograms()
