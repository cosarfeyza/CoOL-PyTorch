# r/CoOL_6_dendrogram_runner.R
# Exact CoOL_6_dendrogram backend (ClustGeo + ggtree)

suppressPackageStartupMessages({
  library(ClustGeo)
  library(plyr)
  library(ggtree)
  library(ggplot2)
  library(wesanderson)
})

args <- commandArgs(trailingOnly = TRUE)

# Args:
# 1: risk_contributions CSV (header=TRUE)
# 2: ipw CSV (one column, no header) OR literal "1"
# 3: number_of_subgroups (K)
# 4: title (string)
# 5: colours (comma-separated) OR "NA"
# 6: out_clusters_csv
# 7: out_plot_png

if (length(args) < 7) {
  stop("Usage: Rscript CoOL_6_dendrogram_runner.R <risk_csv> <ipw_csv_or_1> <K> <title> <colours_or_NA> <out_clus_csv> <out_plot_png>")
}

risk_path <- args[1]
ipw_arg   <- args[2]
K         <- as.integer(args[3])
plot_title<- args[4]
col_arg   <- args[5]
out_clus  <- args[6]
out_plot  <- args[7]

# --- Load risk contributions ---
risk_contributions <- read.csv(risk_path, check.names = FALSE)
n <- nrow(risk_contributions)

# --- IPW handling (exact CoOL behavior) ---
ipw <- rep(1, n)
if (ipw_arg != "1") {
  ipw_tmp <- as.numeric(read.csv(ipw_arg, header = FALSE)[, 1])
  if (length(ipw_tmp) == n) {
    ipw <- ipw_tmp
  } else {
    ipw <- rep(1, n)
    print("Equal weights are applied (assuming no selection bias)")
  }
}

# --- Collapse identical profiles (CoOL logic) ---
p <- cbind(risk_contributions)
p$ipw <- ipw
p <- plyr::count(p, wt_var = "ipw")
pfreq <- p$freq
p_mat <- p[, 1:(ncol(p) - 3)]

# --- Hierarchical clustering (CoOL) ---
p_h_c <- hclustgeo(dist(p_mat, method = "manhattan"), wt = pfreq)
pclus <- cutree(p_h_c, K)

# --- Map clusters back to individuals (CoOL merge logic) ---
id <- 1:n
temp <- merge(cbind(id, risk_contributions), cbind(p_mat, pclus))
temp <- temp[duplicated(temp) == FALSE, ]
clus <- temp$pclus[order(temp$id)]

# --- Print subgroup sizes (like original CoOL) ---
print(table(clus))

# --- Colours ---
if (col_arg == "NA") {
  pal <- c("grey", wesanderson::wes_palette("Darjeeling1"))
} else {
  pal <- strsplit(col_arg, ",")[[1]]
}

# --- Save dendrogram to PNG ---
png(out_plot, width = 1200, height = 1200, res = 200)
print(
  ggtree::ggtree(p_h_c, layout = "equal_angle") +
    ggtree::geom_tippoint(
      size = sqrt(pfreq) / 2,
      alpha = .2,
      color = pal[pclus]
    ) +
    ggtitle(plot_title) +
    ggtree::theme(plot.title = element_text(size = 15, face = "bold"))
)
dev.off()

# --- Save individual cluster labels ---
write.csv(data.frame(cluster = clus), out_clus, row.names = FALSE)
