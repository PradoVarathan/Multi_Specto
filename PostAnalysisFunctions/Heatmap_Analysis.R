
# Uploading library and files ---------------------------------------------

library(igraph)
library(reticulate)
library(biclust)
library(dplyr)
library(pheatmap)
library(gridExtra)
library(RColorBrewer)
library(VennDiagram)
library(ggVennDiagram)
library(ggplot2)

#Importing the cofig file with all the input files
config_path = commandArgs(trailingOnly=TRUE)
config <- config::get(file = config_path[1],"heatmap_analysis", use_parent = FALSE)


#Importing all pickle files for top 10
pd <- import("pandas")
plot_dist_funcs = config$plot_dist_funcs
use_hap_combs = config$use_hap_combs
#grp1_grp2_sub <- pd$read_pickle("~/Downloads/Top100/DataMatrixGrp2_Grp1.pickle")
result_df = read.csv(config$result_df,header = T)
grp1_sub <- pd$read_pickle(config$grp1_sub)
grp2_whole <- pd$read_pickle(config$grp2_whole)
originalsnp <- pd$read_pickle(config$originalsnp)
output_dir = config$output_dir
top_snp_ids = names(grp2_whole)



# Obtaining important SNPs for work ---------------------------------------

main_snp_vals_pass_thres = c("rs12972156","rs184017","rs584007","rs2972559","rs157585",
                             "rs12462573","rs112849259","rs114536010","rs74579864")
comb_snp_vals_pass_thres = c("rs483082","rs41377151","rs1038025","rs2238681","rs1305062","rs2075649",
                             "rs56317818","rs584007","rs2972559","rs12462573","rs157585","rs79398853",
                             "rs114536010","rs112849259","rs73052307","rs112019714","rs484195","rs58132661",
                             "rs74579864","rs4803762","rs12459575")
main_snps_top_100 = union(main_snp_vals_pass_thres,comb_snp_vals_pass_thres)

main_snps_top_100_filt =  c("rs1038025","rs1305062",
                            "rs56317818","rs584007","rs2972559","rs12462573",
                            "rs112849259","rs73052307","rs112019714","rs484195","rs58132661",
                            "rs74579864","rs4803762","rs12459575")




# Distribution plots ------------------------------------------------------
# Obtaining the Brain Features from all features info
all_features = unlist(grp2_whole$rs429358$xlabels)
features_imp = c('Gliobla','Astrocytes','Monocytes','Brain','Neuro')
brain_related_features = c()
brain_related_features_boolean = matrix(0,1,2002)
for (f in features_imp){
  idx_feats = grepl(f,all_features)
  brain_related_features_boolean = brain_related_features_boolean + as.numeric(idx_feats)
  brain_related_features = c(brain_related_features, unlist(all_features[idx_feats]))
  print(paste0("Total number of features for ",f," is ",as.character(sum(idx_feats))))
}
# removing featal indexes
featal_idx = grepl('Fetal',all_features)
brain_related_features_boolean[featal_idx] = 0
brain_related_features = brain_related_features[!grepl('Fetal',brain_related_features)]


# Main SNP distribution
plot_distribution_mainsnp = function(snp_name,brain_related = FALSE){
  title = paste0("Distribution of MainSNP only: ",snp_name)
  if (brain_related){
    t = grp1_sub[[snp_name]]$data
    t = t[as.logical(brain_related_features_boolean)]
    title = paste0('Brain Related Regions Distribution - ',snp_name)
    if(is.null(t)){
      absolute_values = 0
    }else{
    absolute_values = abs(as.numeric(as.matrix(t)))
    }
  }else{  
    title = paste0('Whole Distribution - ',snp_name)
    absolute_values = abs(as.numeric(as.matrix(grp1_sub[[snp_name]]$data)))
  }
  
  return(hist(absolute_values,breaks = 100,  main =snp_name))
}





# Combination Max SNP distribution
plot_distribution_combsnps = function(snp_name,brain_related = FALSE){
  
  title = paste0("Distribution of MainSNP only: ",snp_name)
  group2_data_matrix = grp2_whole[snp_name][1][[1]]$data
  if (brain_related){
    t = group2_data_matrix
    t = t[,as.logical(brain_related_features_boolean)]
    title = paste0('Brain Related Regions Distribution - ',snp_name)
    absolute_max_values = apply(abs(t),2,max)
  }else{  
    title = paste0('Whole Distribution - ',snp_name)
    absolute_max_values = apply(abs(group2_data_matrix),2,max)
  }
  return(hist(absolute_max_values,breaks = 100,  main =snp_name))
}

# Combination Max SNP - Absolute Ind distribution
plot_distribution_diff = function(snp_name,brain_related= FALSE){
  idx = which(originalsnp[[snp_name]]$xlabels==snp_name)[1]
  absolute_values = abs(as.matrix(originalsnp[snp_name][1][[1]]$data)[idx,as.logical(brain_related_features_boolean)])
  title = paste0("Distribution of Difference: ",snp_name)
  group2_data_matrix = grp2_whole[snp_name][1][[1]]$data[,as.logical(brain_related_features_boolean)]
  if (brain_related){
    absolute_values = abs(as.matrix(originalsnp[snp_name][1][[1]]$data)[idx,as.logical(brain_related_features_boolean)])
    title = paste0("Distribution of Difference: ",snp_name)
    group2_data_matrix = grp2_whole[snp_name][1][[1]]$data[,as.logical(brain_related_features_boolean)]
  }else{  
    absolute_values = abs(as.matrix(originalsnp[snp_name][1][[1]]$data)[1,])
    title = paste0("Distribution of Difference: ",snp_name)
    group2_data_matrix = grp2_whole[snp_name][1][[1]]$data
  }
  absolute_max_values = apply(abs(group2_data_matrix),2,max)
  res = absolute_values - absolute_max_values
  return(hist(res,breaks = 100,  main =snp_name))
}





if(plot_dist_funcs){
  ## Distribution Plots of Main SNP Only
  
  par(mfrow = c(2,5))
  for (name in top_snp_ids){
    plot_distribution_mainsnp(name,brain_related = TRUE)
  }
  
  
  ## Distribution Plots of Comb SNP Only
  
  par(mfrow = c(2,5))
  for (name in top_snp_ids){
    try(plot_distribution_combsnps(name,brain_related = TRUE),silent = TRUE)
  }
  
  
  ## Distribution Plots of Difference Only
  
  par(mfrow = c(2,5))
  for (name in main_snps_top_100){
    try(plot_distribution_diff(name,brain_related = TRUE),silent = TRUE)
  }
  
}



file_path_list = list()

# Printing and Making all heatmaps ----------------------------------------

start = TRUE
for (snp_name in main_snps_top_100_filt){
  print(paste0('Loading file for SNP...',snp_name))
  print('---------------------------------------')
  group2_data_matrix = grp2_whole[[snp_name]]$data
  group2_snps_combs = grp2_whole[snp_name][1][[1]]$ylabels
  snps_boolean = matrix(0,1,length(group2_snps_combs))
  
  # SNP combs with main SNP
  idx_feats = grepl(snp_name,group2_snps_combs)
  snps_boolean_combs_with_main_snp = snps_boolean + as.numeric(idx_feats)
  all_features = unlist(grp1_sub$rs184017$ylabels)
  
  idx_ms = which(originalsnp[[snp_name]]$xlabels==snp_name)[1]
  
  final_whole_combs = group2_data_matrix[,as.logical(brain_related_features_boolean)]
  final_grp1_all = originalsnp[snp_name][1][[1]]$data
  final_grp1_all = final_grp1_all[,as.logical(brain_related_features_boolean)]
  rownames(final_grp1_all) = originalsnp[snp_name][1][[1]]$xlabels
  colnames(final_grp1_all) = all_features[as.logical(brain_related_features_boolean)]
  colnames(final_whole_combs) = all_features[as.logical(brain_related_features_boolean)]
  rownames(final_whole_combs) = group2_snps_combs
  col_Names = all_features[as.logical(brain_related_features_boolean)]
  col_Name_Atr = c()
  col_Name_Feat = c()
  
  for(nn in col_Names){
    temp_col = strsplit(nn,'__')[[1]]
    col_Name_Atr = c(col_Name_Atr,temp_col[3])
    col_Name_Feat = c(col_Name_Feat,paste(unlist(temp_col[1:2]),collapse = '__'))
  }
  #col_Name_Atr = data.frame(col_Name_Atr,row.names = colnames(final_imp_snp_feat))
  rg <- max(abs(final_whole_combs));
  
  
  grp1_max_sub_snp = unlist(grp1_sub[[snp_name]]$data)
  grp1_max_sub_snp = t(as.matrix(grp1_max_sub_snp))
  names(grp1_max_sub_snp) = snp_name
  #grp1_max_sub_snp = unlist(grp1_max_sub_snp[as.logical(brain_related_features_boolean)])
  dim_snps = nrow(final_whole_combs)
  if (dim_snps < 10){
    k = pheatmap::pheatmap(final_whole_combs,show_rownames=FALSE, labels_col='',legend = TRUE, 
                           breaks = seq(-rg, rg, length.out = 100),cutree_rows = dim_snps,cutree_cols = 4,,silent = TRUE) 
  }else{
    k = pheatmap::pheatmap(final_whole_combs,show_rownames=FALSE, labels_col='',legend = TRUE, 
                           breaks = seq(-rg, rg, length.out = 100),cutree_rows = 10,cutree_cols = 4,,silent = TRUE) 
    dim_snps = 10
  }
  
  #reoreder_index = k$tree_col$order
  
  print('Done with calculation of clusters ...')
  print('---------------------------------------')
  
  # redoing the snp names for repeats
  snps_rows = rownames(final_whole_combs) 
  emp = c()
  new_snps_rows = c()
  for (s in snps_rows){
    if(s %in% emp){
      new_s = paste0(s,'_2')
      new_snps_rows = c(new_snps_rows,new_s)
    }
    else{
      new_s = paste0(s,'_1')
      new_snps_rows = c(new_snps_rows,new_s)
      emp = c(emp,s)
    }
  }
  rownames(final_whole_combs) = new_snps_rows
  
  ind_snps_rows = rownames(final_grp1_all)
  emp = c()
  new_snps_rows_2 = c()
  for (s in ind_snps_rows){
    if(s %in% emp){
      new_s = paste0(s,'_2')
      new_snps_rows_2 = c(new_snps_rows_2,new_s)
    }
    else{
      new_s = paste0(s,'_1')
      new_snps_rows_2 = c(new_snps_rows_2,new_s)
      emp = c(emp,s)
    }
  }
  rownames(final_grp1_all) = new_snps_rows_2
  
  #redoing the column names for annotations
  feat_cols = colnames(final_whole_combs)
  new_feat_cols_names = c()
  new_feat_cols = c()
  specs = c()
  cats3 = c()
  emp = data.frame('Name' = character(),'Freq' = numeric())
  for (s in feat_cols){
    splts = strsplit(s,'__')[[1]]
    specs = c(specs,splts[2])
    cats3 = c(cats3,splts[3])
    new_feat_cols_names = c(new_feat_cols_names,paste0(head(strsplit(s,'__')[[1]],-1),collapse = '_'))
    if(splts[1] %in% emp$Name){
      idxx = which(emp$Name %in% splts[1])
      n = emp$Freq[idxx] + 1
      new_feat_cols = c(new_feat_cols,paste0(splts[1],'__',as.character(n)))
      emp$Freq[idxx] = n
    }
    else{
      new_feat_cols = c(new_feat_cols,paste0(splts[1],'__1'))
      emp = rbind(emp,data.frame('Name'=splts[1],'Freq' = 1))
    }
  }
  colnames(final_whole_combs) = new_feat_cols
  colnames(final_grp1_all) = new_feat_cols
  #names(grp1_max_sub_snp) = col_Names
  print('Making annotations')
  print('---------------------------------------')
  
  annotation_feats = data.frame('ColName' =new_feat_cols,'Original_feat_name' = feat_cols,'Feature Type'= cats3,'Feature SubType' = specs,'Labels' = new_feat_cols_names)
  rownames(annotation_feats) = annotation_feats$ColName
  annotation_feats$ColName = NULL
  
  
  snp_comb_temp = list()
  snp_freq_comb = list()
  k.clust = data.frame(cluster = cutree(k$tree_row, k = dim_snps))
  k.clust = cbind(as.data.frame(new_snps_rows),k.clust)
  for(i in unique(k.clust$cluster)){
    temp_cl = k.clust %>% filter(cluster == i)
    snp_comb_temp[[i]] = temp_cl$new_snps_rows
    temp = list()
    for(combs in temp_cl$new_snps_rows){
      temp = c(temp,head(strsplit(combs,'_')[[1]],-1))
    }
    temp = unlist(temp)
    temp_df = as.data.frame(table(temp))
    snp_freq_comb[[i]] = temp_df %>% arrange(desc(Freq))
    
  }
  print(snp_freq_comb)
  print('---------------------------------------')
  
  
  print('Compare line  results with snp_freq_comb dataframe to see which SNPs are most important')
  unlist(lapply(snp_comb_temp,length))
  k.clust = data.frame(cluster = cutree(k$tree_row, k = dim_snps))
  k.clust = cbind(as.data.frame(new_snps_rows),k.clust)
  row.names(k.clust) = k.clust$new_snps_rows
  k.clust$new_snps_rows = NULL
  k.clust$cluster = as.character(k.clust$cluster)
  
  #feat_cols = colnames(final_imp_snp_feat)
  all_feature_idx = all_features %in% feat_cols
  all_feat_cols = all_features[all_feature_idx]
  grp1_max_sub_final = grp1_max_sub_snp[all_feature_idx]
  grp1_max_sub_final = as.data.frame(grp1_max_sub_final)
  rownames(grp1_max_sub_final) = new_feat_cols
  colnames(grp1_max_sub_final) = c('Max_MainSNP')
  rg2 = max(abs(grp1_max_sub_final$Max_MainSNP))
  
  #annotation_feats$cols = rownames(annotation_feats)
  grp1_max_sub_final$ColName = new_feat_cols
  annotation_feats = merge(annotation_feats,grp1_max_sub_final,by = 'row.names')
  
  #rg <- max(abs(group2_data_matrix));
  
  
  print('Working on the color palette')
  print('---------------------------------------')
  
  color=colorRampPalette(c("navy", "white", "red"))(100)
  breaks = seq(-rg, rg, length.out = 100)
  color_pal_df = data.frame('Color' = color,'Break' = breaks)
  
  annot_main_snp_cols = c()
  annote_matrix = annotation_feats$Max_MainSNP
  #annotation_feats$Max_MainSNP_Col = NULL
  #annote_matrix = annote_matrix[reoreder_index]
  for(i in annotation_feats$Max_MainSNP){
    for(j in 2:nrow(color_pal_df)){
      if ((i < color_pal_df$Break[j]) && (i > color_pal_df$Break[j-1])){
        annot_main_snp_cols = c(annot_main_snp_cols,as.character(color_pal_df$Color[j]))
        break
      }
    }
  }
  
  
  annot_main_snp_cols_list = list()
  col_list = list()
  for (i in 1:128){
    tit = paste0('feat_main_snp_',as.character(i))
    col_list = c(col_list,tit)
    annot_main_snp_cols_list = c(annot_main_snp_cols_list, tit = annot_main_snp_cols[i])
  }
  names(annot_main_snp_cols_list) = col_list
  #Adding to annotations
  annotation_feats$Max_MainSNP = names(annot_main_snp_cols_list)
  ann_cols = list(
    'Max_MainSNP' = unlist(annot_main_snp_cols_list),
    'Feature.Type' = c(DNase = 'red',Histone = 'green', TF = 'yellow')
  )
  
  
  print('Working on the original SNP values')
  print('---------------------------------------')
  
  ### Original SNP and Orignal SNP combinations
  
  # eff_snp = switch(snp_name,"rs2972559" = "rs4802241",
  #                  "rs12462573" = "rs75765623",
  #                  "rs484195" = "rs12721056",
  #                  "rs1305062" = "rs141864196",
  #                  "rs112849259" = "rs157585",
  #                  snp_name)
  eff_snp = snp_name
  snps_boolean = matrix(0,1,length(group2_snps_combs))
  
  # SNP combs with main SNP
  idx_feats = grepl(eff_snp,group2_snps_combs)
  snps_boolean_combs_with_eff_snp = snps_boolean + as.numeric(idx_feats)
  
  final_combs_with_main_snp = final_whole_combs[as.logical(snps_boolean_combs_with_eff_snp),]
  final_combs_without_main_snp = final_whole_combs[!as.logical(snps_boolean_combs_with_eff_snp),]
  
  
  
  og_snp_data_1 = originalsnp[[snp_name]]$data[1,]
  og_snp_data = originalsnp[snp_name][1][[1]]$data
  og_snp_data = og_snp_data[which(snp_name == originalsnp[snp_name][1][[1]]$xlabels),]
  if (sum(og_snp_data[1,] == og_snp_data[2,]) == 2002){
    og_snp_data = og_snp_data[1,]
  }
  og_snp_data = og_snp_data_1[all_feature_idx]
  og_main_snp_cols = c()
  print('Obtaining color palette for SNP')
  print('---------------------------------------')
  
  for(i in og_snp_data){
    for(j in 2:nrow(color_pal_df)){
      if ((i < color_pal_df$Break[j]) && (i > color_pal_df$Break[j-1])){
        og_main_snp_cols = c(og_main_snp_cols,as.character(color_pal_df$Color[j]))
        #print(j)
        break
      }
    }
  }
  
  
  og_snp_data_snp_cols_list = list()
  col_list = list()
  for (i in 1:128){
    tit = paste0('og_main_snp_',as.character(i))
    col_list = c(col_list,tit)
    og_snp_data_snp_cols_list = c(og_snp_data_snp_cols_list, og = og_main_snp_cols[i])
  }
  og_df = data.frame('OrgSNP' = new_feat_cols,'Color' = og_main_snp_cols)
  rownames(og_df) = og_df$OrgSNP
  rownames(annotation_feats) =annotation_feats$Row.names
  annotation_feats = merge(annotation_feats,og_df,by='row.names')
  rownames(annotation_feats) =annotation_feats$ColName
  annotation_feats$Row.names = NULL
  og_snp_data_snp_cols_list = annotation_feats$Color
  names(og_snp_data_snp_cols_list) = annotation_feats$OrgSNP
  
  ann_cols = list(
    'Max_G1' = unlist(annot_main_snp_cols_list),
    'OrgSNP' = unlist(og_snp_data_snp_cols_list),
    'F.Type' = c(DNase = 'red',Histone = 'green', TF = 'yellow')
  )
  
  annotation_feats$ColName = NULL
  annotation_feats$Original_feat_name = NULL
  annotation_feats$Row.names = NULL
  annotation_feats$Color = NULL
  annotation_feats$Labels = NULL
  
  colnames(annotation_feats) = c('F.Type','F.S.Type','Max_G1','OrgSNP')
  
  label_col_list = c()
  for(i in annotation_feats$F.Type){
    if(i == 'Histone'){
      label_col_list = c(label_col_list,'black')
    }
    else if(i == 'TF'){
      label_col_list = c(label_col_list,'blue')
    }
    else{
      label_col_list = c(label_col_list,'red')
    }
  }
  
  
  # Save pheatmap function --------------------------------------------------
  
  save_pheatmap_pdf <- function(x, filename, width=19, height=17) {
    stopifnot(!missing(x))
    stopifnot(!missing(filename))
    pdf(filename, width=width, height=height)
    grid::grid.newpage()
    grid::grid.draw(x$gtable)
    dev.off()
  }
  
  
  # Plotting all heatmaps ---------------------------------------------------
  

  title_temp = paste0('Heatmap of all combination clusters of ',snp_name)
  print(paste0('Making ',title_temp))
  print('---------------------------------------')
  
  plot_file_path <- paste0(output_dir,'Heatmap_of_all_combination_clusters_of_',snp_name, ".png")
  file_path_list <- append(file_path_list, list(plot_file_path))
  
  annotation_feats$F.Type = NULL
  annotation_feats$F.S.Type = NULL
  dim_snps = nrow(final_whole_combs)
  if(dim_snps >= 10){
    k = pheatmap::pheatmap(final_whole_combs
                         ,show_rownames = FALSE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                         ,annotation_col = annotation_feats, cutree_rows = 10,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                         annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                         main = title_temp,silent = TRUE) 
  }else{
    k = pheatmap::pheatmap(final_whole_combs
                           ,show_rownames = FALSE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                           ,annotation_col = annotation_feats, cutree_rows = dim_snps,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                           annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                           main = title_temp,silent = TRUE) 
  }
  reoreder_index = k$tree_col$order
  label_col_list = rep('blue',length(new_feat_cols_names))
  label_col_list[grep('_H',new_feat_cols_names)] = 'black'
  label_col_list[grep('_DNase',new_feat_cols_names)] = 'red'
  k$gtable$grobs[[5]]$gp$col <- label_col_list[reoreder_index]
  
  leg <- legendGrob(c("Histone", "TF","DNase"), nrow = 3, pch = 15, gp = gpar(fontsize = 8.5, col = c("black", "blue","red")))
  k_1 <- arrangeGrob(k$gtable, leg, ncol = 2, widths = c(10,1))
  grid.draw(k_1)
  ggsave(filename = plot_file_path, plot = k_1, width=25, height=12, units = "in", scale = 1, dpi = 1200)
  if(!is.null(dev.list())) dev.off()
  
  
  
  
  title_temp = paste0('Heatmap of combinations with ',snp_name)
  print(paste0('Making ',title_temp))
  print('---------------------------------------')
  
  plot_file_path <- paste0(output_dir,'Heatmap_of_combination_with_',snp_name, ".png")
  file_path_list <- append(file_path_list, list(plot_file_path))
  dim_snps = nrow(final_combs_with_main_snp)
  if(dim_snps >= 10){
    k2 = pheatmap::pheatmap(final_combs_with_main_snp
                          ,show_rownames = FALSE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                          ,annotation_col = annotation_feats, cutree_rows = 10,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                          annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                          main = title_temp,silent = TRUE) 
  }else{
    k2 = pheatmap::pheatmap(final_combs_with_main_snp
                            ,show_rownames = FALSE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                            ,annotation_col = annotation_feats, cutree_rows = dim_snps,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                            annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                            main = title_temp,silent = TRUE) 
  }
  reoreder_index = k2$tree_col$order
  label_col_list = rep('blue',length(new_feat_cols_names))
  label_col_list[grep('_H',new_feat_cols_names)] = 'black'
  label_col_list[grep('_DNase',new_feat_cols_names)] = 'red'
  k2$gtable$grobs[[5]]$gp$col <- label_col_list[reoreder_index]
  leg <- legendGrob(c("Histone", "TF","DNase"), nrow = 3, pch = 15, gp = gpar(fontsize = 8, col = c("black", "blue","red")))
  k_2 <- arrangeGrob(k2$gtable, leg, ncol = 2, widths = c(11,1))
  grid.draw(k_2)
  ggsave(filename = plot_file_path, plot = k_2, width=25, height=12, units = "in", scale = 1, dpi = 1200)
  if(!is.null(dev.list())) dev.off()
  
  title_temp = paste0('Heatmap of combinations without ',snp_name)
  print(paste0('Making ',title_temp))
  print('---------------------------------------')
  
  plot_file_path <- paste0(output_dir,'Heatmap_of_combination_without_',snp_name, ".png")
  file_path_list <- append(file_path_list, list(plot_file_path))
  dim_snps = nrow(final_combs_without_main_snp)
  if(dim_snps >= 10){
    k2_1 = pheatmap::pheatmap(final_combs_without_main_snp
                              ,show_rownames = FALSE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                              ,annotation_col = annotation_feats, cutree_rows = 8,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                              annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                              main = title_temp,silent = TRUE)
  }else{
    k2_1 = pheatmap::pheatmap(final_combs_without_main_snp
                              ,show_rownames = FALSE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                              ,annotation_col = annotation_feats, cutree_rows = dim_snps,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                              annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                              main = title_temp,silent = TRUE)
  }
   
  reoreder_index = k2_1$tree_col$order
  k2_1$gtable$grobs[[5]]$gp$col <- label_col_list[reoreder_index]
  leg <- legendGrob(c("Histone", "TF","DNase"), nrow = 3, pch = 15, gp = gpar(fontsize = 8, col = c("black", "blue","red")))
  k2_1 <- arrangeGrob(k2_1$gtable, leg, ncol = 2, widths = c(11,1))
  grid.draw(k2_1)
  ggsave(filename = plot_file_path, plot = k2_1, width=25, height=12, units = "in", scale = 1, dpi = 1200)
  if(!is.null(dev.list())) dev.off()
  
  
  title_temp = paste0('Heatmap of Individual SNPs with ',eff_snp)
  print(paste0('Making ',title_temp))
  print('---------------------------------------')
  
  plot_file_path <- paste0(output_dir,'Heatmap_of_Individual_SNPs_',snp_name, ".png")
  file_path_list <- append(file_path_list, list(plot_file_path))
  dim_snps = nrow(final_grp1_all)
  if(start){
    grp1_all_ind_mtx = final_grp1_all
    start = FALSE
  }else{
    grp1_all_ind_mtx = rbind(grp1_all_ind_mtx,final_grp1_all)
  }
  if(dim_snps >= 10){
    k3 = pheatmap::pheatmap(final_grp1_all
                          ,show_rownames = TRUE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                          ,annotation_col = annotation_feats, cutree_rows = 10,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                          annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                          main = title_temp,silent = TRUE,annotation_names_col = TRUE) 
  }else{
    k3 = pheatmap::pheatmap(final_grp1_all
                            ,show_rownames = TRUE, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                            ,annotation_col = annotation_feats, cutree_rows = dim_snps,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                            annotation_colors = ann_cols,color = color,annotation_legend = FALSE,
                            main = title_temp,silent = TRUE,annotation_names_col = TRUE) 
  }
  reoreder_index = k3$tree_col$order
  k3$gtable$grobs[[5]]$gp$col <- label_col_list[reoreder_index]
  leg <- legendGrob(c("Histone", "TF","DNase"), nrow = 3, pch = 15, gp = gpar(fontsize = 8, col = c("black", "blue","red")))
  k_3 <- arrangeGrob(k3$gtable, leg, ncol = 2, widths = c(10,1))
  grid.draw(k_3)
  ggsave(filename = plot_file_path, plot = k_3, width=25, height=12, units = "in", scale = 1, dpi = 1200)
  if(!is.null(dev.list())) dev.off()
  gc()
  
  
  
  # Post-general heatmaps-- haplotype based  --------------------------------
  if(use_hap_combs){
    snps_boolean_combs = matrix(0,1,length(group2_snps_combs))
    snps_from_iso = c()
    filtered_combs_df = result_df %>% filter(Main_SNP == snp_name)
    filtered_combs_snp = filtered_combs_df$Combination
    
    for(comb_seq in filtered_combs_snp){
      if(comb_seq != ""){
        if(length(strsplit(comb_seq,'_')[[1]])>1){
          snps_boolean_combs[which(group2_snps_combs == comb_seq)[1]] = 1
        }
        else{
          snps_from_iso = c(snps_from_iso,paste0(comb_seq,'_1'))
        }
      }
    }
    
    
    
    # SNP combs with main SNP
    
    final_combs_with_hap_combs = final_whole_combs[as.logical(snps_boolean_combs),]
    final_combs_with_hap_combs = rbind(final_combs_with_hap_combs,final_grp1_all[rownames(final_grp1_all) %in% snps_from_iso,])
    rows_new = c()
    for(k in rownames(final_combs_with_hap_combs)){
      rows_new = c(rows_new, strsplit(k,'_1')[[1]][1])
    }
    rownames(final_combs_with_hap_combs) = rows_new
    title_temp = paste0('Heatmap of combinations with Hap Combinations ',snp_name)
    
    print(paste0('Making ',title_temp))
    print('---------------------------------------')
    
    
    
    
    new_feat_cols_names_cell = c()
    new_feat_cols_names_feat = c()
    for(n in new_feat_cols_names){
      temp = strsplit(n,'[_]')[[1]]
      new_feat_cols_names_feat = c(new_feat_cols_names_feat,temp[length(temp)])
      new_feat_cols_names_cell = c(new_feat_cols_names_cell,paste0(temp[1:length(temp)-1],collapse = '_'))
    }
    
    annotation_col = data.frame("labels" = colnames(grp1_all_ind_mtx),
                                "Features" = new_feat_cols_names_feat)
    rownames(annotation_col) = annotation_col$labels
    annotation_col$labels = NULL
    annotation_col$Features = NULL
    annotation_col$Max_G1 = annotation_feats$Max_G1
    annotation_col$OrgSNP = annotation_feats$OrgSNP
    #plot_file_path <- paste0('~/Documents/Heatmaps_Top_100/Grp1_Elements/Heatmap_of_combination_with_',snp_name, ".png")
    #file_path_list <- append(file_path_list, list(plot_file_path))
    dim_snps = nrow(final_combs_with_hap_combs)
    plot_file_path <- paste0(output_dir,'Heatmap_of_HaplotypeCombinations_',snp_name, ".png")
    file_path_list <- append(file_path_list, list(plot_file_path))
    if(dim_snps >= 10){
      k2 = pheatmap::pheatmap(final_combs_with_hap_combs
                              ,show_rownames = T, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                              ,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,fontsize_col=10,
                              annotation_col = annotation_col,annotation_legend = F,color = color,annotation_colors=ann_cols,
                              main = title_temp,silent = TRUE) 
    }else{
      k2 = pheatmap::pheatmap(final_combs_with_hap_combs
                              ,show_rownames = T, labels_col=new_feat_cols_names,legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                              ,fontsize_col  = 7.5,fontsize_row = 10,fontsize = 11,
                              annotation_col = annotation_col,annotation_legend = F,color = color,annotation_colors=ann_cols,
                              main = title_temp,silent = TRUE) 
    }
    
    
    reoreder_index = k2$tree_col$order
    label_col_list = rep('blue',length(new_feat_cols_names))
    label_col_list[grep('_H',new_feat_cols_names)] = 'black'
    label_col_list[grep('_DNase',new_feat_cols_names)] = 'red'
    k2$gtable$grobs[[5]]$gp$col <- label_col_list[reoreder_index]
    k_2 <- arrangeGrob(k2$gtable, ncol = 2, widths = c(11,1))
    grid.draw(k_2)
    ggsave(filename = plot_file_path, plot = k_2, width=25, height=12, units = "in", scale = 1, dpi = 1200)
    if(!is.null(dev.list())) dev.off()
    gc()
  }
  
}

if(make_all_ind){
        # Creating heatmap of all individual SNPs ---------------------------------
  rg <- max(abs(grp1_all_ind_mtx))
  #temp change in max
  rg = 1.5
  row_names_heat = c()
  for(i in 1:length(rownames(grp1_all_ind_mtx))){
    if(i%%2 == 0){
      row_names_heat = c(row_names_heat,'')
    }else{
      row_names_heat = c(row_names_heat,as.character(strsplit(rownames(grp1_all_ind_mtx)[i],"[_]")[[1]][1]))
    }
  }
  hist(colSums(abs(grp1_all_ind_mtx)),bins = 500)
  threshold_set = 1
  colmaxese = c()
  for(r in 1:ncol(grp1_all_ind_mtx)){
    temp = max(abs(grp1_all_ind_mtx[,r]))
    colmaxese = c(colmaxese,temp)
  }
  idx_threshold = colmaxese > 1
  new_feat_cols_names_cell = c()
  new_feat_cols_names_feat = c()
  for(n in new_feat_cols_names){
    temp = strsplit(n,'[_]')[[1]]
    new_feat_cols_names_feat = c(new_feat_cols_names_feat,temp[length(temp)])
    new_feat_cols_names_cell = c(new_feat_cols_names_cell,paste0(temp[1:length(temp)-1],collapse = '_'))
  }
  annotation_col = data.frame("labels" = colnames(grp1_all_ind_mtx)[idx_threshold],
                              "Features" = new_feat_cols_names_feat[idx_threshold],
                              "Cell Types" = new_feat_cols_names_cell[idx_threshold])
  rownames(annotation_col) = annotation_col$labels
  annotation_col$labels = NULL
  annotation_col$Features = NULL
  plot_file_path=paste0(output_dir,'Heatmap_of_all_Individual_SNP.png')
  k_all = pheatmap::pheatmap(t(grp1_all_ind_mtx[,idx_threshold])
                             ,labels_col  = row_names_heat, labels_row  =new_feat_cols_names_feat[idx_threshold],legend = TRUE, breaks = seq(-rg, rg, length.out = 100)
                             ,cutree_cols = 4,fontsize_col  = 7.5,fontsize_row = 7.5,fontsize = 11,annotation_row = annotation_col,annotation_legend = TRUE,cluster_cols = F,
                             main = "Heatmap of all Individual SNPs pass threshold",color = color,silent = T) 
  ggsave(filename = plot_file_path, plot = k_3, width=25, height=12, units = "in", scale = 1, dpi = 1200)
  if(!is.null(dev.list())) dev.off()
  gc()
}




