# CoOL_dendrogram_runner.r
#
# The R side of the R/ClustGeo clustering backend. Not called directly — invoked
# as a subprocess (Rscript) from CoolTorch/metrics/dendo_clustgeo.py, which writes
# the risk contributions to CSV, calls this script, and reads the resulting cluster
# labels + dendrogram PNG back into Python. Used by metrics/sub_groups_clustgeo.py
# and metrics/number_of_sub_groups_clustgeo.py for the R-faithful clustering path
# (the scipy-Ward alternative in metrics/sub_groups.py needs neither this script
# nor R at all).
#
# R equivalent: CoOL_6_dendrogram (CoOL_functions.R) — this is that function's
# clustering + dendrogram logic, factored out into a standalone script so it can
# be called from Python instead of from an R session.

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

# --- Read the risk contributions written out by dendo_clustgeo.py ---
risk_contributions <- read.csv(risk_path, check.names = FALSE)
n <- nrow(risk_contributions)

# --- IPW: per-row weights, or 1 (equal weights) if the caller passed the literal "1" ---
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

# --- Collapse rows with identical risk-contribution profiles before clustering,
# carrying their combined (ipw-weighted) count as that unique row's weight ---
p <- cbind(risk_contributions)
p$ipw <- ipw
p <- plyr::count(p, wt_var = "ipw")
pfreq <- p$freq
p_mat <- p[, 1:(ncol(p) - 3)]

# --- The actual clustering: ClustGeo's hclustgeo on the unique, weighted rows,
# Manhattan distance, cut into K sub-groups ---
p_h_c <- hclustgeo(dist(p_mat, method = "manhattan"), wt = pfreq)
pclus <- cutree(p_h_c, K)

# --- Expand the per-unique-row cluster labels back out to one label per
# individual (reversing the collapse step above) ---
id <- 1:n
temp <- merge(cbind(id, risk_contributions), cbind(p_mat, pclus))
temp <- temp[duplicated(temp) == FALSE, ]
clus <- temp$pclus[order(temp$id)]

# --- Print sub-group sizes, same as the CoOL_6_dendrogram console output ---
print(table(clus))

# --- Colours: one per cluster, either the default palette or the caller's own ---
if (col_arg == "NA") {
  pal <- c("grey", wesanderson::wes_palette("Darjeeling1"))
} else {
  pal <- strsplit(col_arg, ",")[[1]]
}

# --- Draw the equal-angle dendrogram (ggtree), tips sized by row weight and
# coloured by cluster, and save it to out_plot for Python to load back in ---
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

# --- Write the per-individual cluster labels to out_clus for Python to read back in ---
write.csv(data.frame(cluster = clus), out_clus, row.names = FALSE)
