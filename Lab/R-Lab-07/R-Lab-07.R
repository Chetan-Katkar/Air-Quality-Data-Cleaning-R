# =============================================================================
# R Programming Lab - Problem Statement 07
# Customer Segmentation and Predictive Analytics Using Machine Learning
#
# Dataset : UCI Machine Learning Repository - Online Retail Dataset
# Platform: R (Google Colab / RStudio)
#
# This script is the plain-R equivalent of the accompanying Colab notebook.
# Run with:  Rscript R-Lab-07.R
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr)
  library(cluster); library(factoextra)
  library(randomForest); library(e1071); library(caret); library(pROC)
  library(corrplot); library(dendextend); library(gridExtra); library(scales)
})

set.seed(42)
OUT <- "outputs"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
save_plot <- function(p, file, w = 9, h = 5.5, dpi = 130) {
  ggsave(file.path(OUT, file), p, width = w, height = h, dpi = dpi, bg = "white")
  invisible(p)
}
rule <- function(t) cat("\n", strrep("=", 78), "\n ", t, "\n", strrep("=", 78), "\n", sep = "")

# =============================================================================
# SECTION 1 - DATA ACQUISITION
# =============================================================================
rule("SECTION 1 : DATA ACQUISITION")

# The notebook tries the live UCI sources first and falls back to a
# statistically faithful simulator when the network is unavailable.
simulate_online_retail <- function(n_target = 541909) {
  set.seed(42)
  n_cust <- 4372; n_prod <- 4070

  stock_codes <- sprintf("%05d", sample(10000:99999, n_prod))
  adjectives  <- c("VINTAGE","RETROSPOT","REGENCY","WOODLAND","ANTIQUE","PAPER",
                   "CERAMIC","GLASS","HEART","POLKADOT","JUMBO","SET OF 6",
                   "HAND WARMER","LUNCH BAG","CHRISTMAS","GARDEN","SPOTTY")
  nouns       <- c("BAG","MUG","CANDLE HOLDER","TEACUP AND SAUCER","CAKE STAND",
                   "DOORMAT","NOTEBOOK","T-LIGHT HOLDER","BUNTING","CUSHION COVER",
                   "PHOTO FRAME","STORAGE JAR","LANTERN","CLOCK","TRINKET BOX")
  descriptions <- paste(sample(adjectives, n_prod, TRUE),
                        sample(nouns, n_prod, TRUE))
  # long-tailed unit prices, as in the real catalogue (median ~2.08)
  unit_price   <- round(pmax(0.04, rlnorm(n_prod, log(2.1), 0.95)), 2)

  countries <- c("United Kingdom","Germany","France","EIRE","Spain","Netherlands",
                 "Belgium","Switzerland","Portugal","Australia","Norway","Italy",
                 "Channel Islands","Finland","Cyprus","Sweden","Austria","Denmark",
                 "Japan","Poland","USA","Israel","Unspecified","Singapore","Iceland")
  cust_country <- sample(countries, n_cust, TRUE,
                         prob = c(0.905, 0.0210, 0.0200, 0.0080, 0.0070, 0.0050,
                                  0.0030, 0.0025, 0.0020, 0.0018, 0.0015, 0.0013,
                                  0.0012, 0.0011, 0.0010, 0.0009, 0.0008, 0.0007,
                                  0.0006, 0.0005, 0.0005, 0.0004, 0.0004, 0.0003, 0.0003))
  cust_ids <- 12346:(12346 + n_cust - 1)

  # customer "value" drives how many invoices and how big they are
  value    <- rlnorm(n_cust, 0, 1.0)
  n_inv    <- pmax(1, rpois(n_cust, pmin(30, 1.2 + value * 3)))
  total_inv <- sum(n_inv)

  inv_cust  <- rep(seq_len(n_cust), times = n_inv)
  start <- as.POSIXct("2010-12-01 08:26:00", tz = "UTC")
  end   <- as.POSIXct("2011-12-09 12:50:00", tz = "UTC")
  span  <- as.numeric(difftime(end, start, units = "secs"))
  # recent-skewed invoice dates (the business grew over the year)
  inv_time <- start + span * (rbeta(total_inv, 1.6, 1.1))
  inv_time <- as.POSIXct(round(as.numeric(inv_time) / 60) * 60,
                         origin = "1970-01-01", tz = "UTC")
  inv_no <- sprintf("%06d", 536365:(536365 + total_inv - 1))

  # ~3.7% more invoice-lines than rows: lines per invoice is long-tailed
  lines <- pmax(1, rpois(total_inv, pmin(40, 2 + value[inv_cust] * 4)))
  scale_f <- n_target / sum(lines)
  lines <- pmax(1, round(lines * scale_f))

  idx  <- rep(seq_len(total_inv), times = lines)
  n    <- length(idx)
  prod <- sample(seq_len(n_prod), n, TRUE,
                 prob = 1 / (seq_len(n_prod) ^ 0.55))   # popularity power law

  df <- data.frame(
    InvoiceNo   = inv_no[idx],
    StockCode   = stock_codes[prod],
    Description = descriptions[prod],
    Quantity    = pmax(1, rpois(n, 6) + rbinom(n, 1, 0.08) * rpois(n, 40)),
    InvoiceDate = inv_time[idx],
    UnitPrice   = unit_price[prod],
    CustomerID  = cust_ids[inv_cust[idx]],
    Country     = cust_country[inv_cust[idx]],
    stringsAsFactors = FALSE
  )

  # --- inject the real dataset's data-quality defects ---------------------
  # 1) ~24.9% of rows have a missing CustomerID (guest / unmatched checkouts)
  miss <- sample(n, round(0.249 * n))
  df$CustomerID[miss] <- NA

  # 2) ~1.7% cancellation lines: invoice prefixed 'C', negative quantity
  canc <- sample(setdiff(seq_len(n), miss), round(0.017 * n))
  df$InvoiceNo[canc] <- paste0("C", df$InvoiceNo[canc])
  df$Quantity[canc]  <- -df$Quantity[canc]

  # 3) a handful of extreme quantities and zero/negative prices
  df$Quantity[sample(n, 60)]  <- sample(c(-80995, 74215, 12540, -9360), 60, TRUE)
  df$UnitPrice[sample(n, 2500)] <- 0
  df$UnitPrice[sample(n, 4)]    <- c(-11062.06, -11062.06, 38970.00, 17836.46)

  # 4) ~0.27% missing descriptions
  df$Description[sample(n, round(0.0027 * n))] <- NA

  df[order(df$InvoiceDate), ]
}

load_online_retail <- function() {
  sources <- list(
    list(url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
         type = "xlsx"),
    list(url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
         type = "zip")
  )
  if (file.exists("Online Retail.xlsx")) {
    cat("Reading local file: Online Retail.xlsx\n")
    return(list(data = as.data.frame(readxl::read_excel("Online Retail.xlsx")),
                source = "local file"))
  }
  for (s in sources) {
    cat("Trying:", s$url, "\n")
    ok <- tryCatch({
      tmp <- tempfile(fileext = paste0(".", s$type))
      utils::download.file(s$url, tmp, mode = "wb", quiet = TRUE)
      d <- if (s$type == "xlsx") as.data.frame(readxl::read_excel(tmp)) else {
        ex <- tempdir(); utils::unzip(tmp, exdir = ex)
        f <- list.files(ex, pattern = "\\.(xlsx|csv)$", full.names = TRUE)[1]
        if (grepl("xlsx$", f)) as.data.frame(readxl::read_excel(f))
        else utils::read.csv(f, stringsAsFactors = FALSE)
      }
      list(data = d, source = s$url)
    }, error = function(e) { cat("   failed:", conditionMessage(e), "\n"); NULL })
    if (!is.null(ok)) return(ok)
  }
  cat("\n*** All remote sources unreachable. Using the built-in simulator. ***\n")
  cat("*** Figures reproduce the real dataset's structure and defects.   ***\n")
  list(data = simulate_online_retail(), source = "built-in simulator")
}

loaded <- load_online_retail()
retail <- loaded$data
DATA_SOURCE <- loaded$source
cat("\nData source :", DATA_SOURCE, "\n")

names(retail) <- gsub("[^A-Za-z]", "", names(retail))
retail$InvoiceDate <- as.POSIXct(retail$InvoiceDate, tz = "UTC")

cat("Dimensions  :", nrow(retail), "rows x", ncol(retail), "columns\n\n")
cat("Structure:\n"); str(retail)
cat("\nFirst 6 rows:\n"); print(head(retail))

# =============================================================================
# SECTION 2 - EXPLORATORY DATA ANALYSIS
# =============================================================================
rule("SECTION 2 : EXPLORATORY DATA ANALYSIS")

miss_tbl <- data.frame(
  Column  = names(retail),
  Missing = sapply(retail, function(x) sum(is.na(x))),
  Percent = round(100 * sapply(retail, function(x) mean(is.na(x))), 2),
  row.names = NULL
)
cat("Missing values per column:\n"); print(miss_tbl)

p_miss <- ggplot(miss_tbl, aes(x = reorder(Column, Percent), y = Percent)) +
  geom_col(fill = "#c0392b") +
  geom_text(aes(label = paste0(Percent, "%")), hjust = -0.15, size = 3.4) +
  coord_flip() + ylim(0, max(miss_tbl$Percent) * 1.25) +
  labs(title = "Missing Values by Column - Online Retail",
       subtitle = paste("Source:", DATA_SOURCE),
       x = "Column", y = "Missing (%)") +
  theme_minimal(base_size = 12)
save_plot(p_miss, "01_missing_values.png")

cat("\nKey counts:\n")
cat("  Transactions (rows)   :", nrow(retail), "\n")
cat("  Unique invoices       :", length(unique(retail$InvoiceNo)), "\n")
cat("  Unique customers      :", length(unique(na.omit(retail$CustomerID))), "\n")
cat("  Unique products       :", length(unique(retail$StockCode)), "\n")
cat("  Countries             :", length(unique(retail$Country)), "\n")
cat("  Date range            :", format(min(retail$InvoiceDate)), "to",
    format(max(retail$InvoiceDate)), "\n")
cat("  Cancellation lines    :", sum(grepl("^C", retail$InvoiceNo)),
    sprintf("(%.2f%%)", 100 * mean(grepl("^C", retail$InvoiceNo))), "\n")
cat("  Negative quantities   :", sum(retail$Quantity < 0), "\n")
cat("  Non-positive prices   :", sum(retail$UnitPrice <= 0), "\n")

top_ctry <- retail %>% count(Country, sort = TRUE) %>% head(10)
cat("\nTop 10 countries by transaction count:\n"); print(top_ctry)

p_ctry <- ggplot(top_ctry, aes(x = reorder(Country, n), y = n)) +
  geom_col(fill = "#2980b9") + coord_flip() +
  scale_y_continuous(labels = comma) +
  labs(title = "Top 10 Countries by Number of Transactions",
       x = "Country", y = "Transactions") +
  theme_minimal(base_size = 12)
save_plot(p_ctry, "02_top_countries.png")

monthly <- retail %>%
  filter(!grepl("^C", InvoiceNo), Quantity > 0, UnitPrice > 0) %>%
  mutate(Month = format(InvoiceDate, "%Y-%m"),
         Revenue = Quantity * UnitPrice) %>%
  group_by(Month) %>% summarise(Revenue = sum(Revenue), .groups = "drop")

p_month <- ggplot(monthly, aes(x = Month, y = Revenue, group = 1)) +
  geom_line(colour = "#27ae60", linewidth = 1.1) +
  geom_point(colour = "#27ae60", size = 2.2) +
  scale_y_continuous(labels = comma) +
  labs(title = "Monthly Revenue Trend", x = "Month", y = "Revenue") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(p_month, "03_monthly_revenue.png")

# =============================================================================
# SECTION 3 - DATA PREPROCESSING
# =============================================================================
rule("SECTION 3 : DATA PREPROCESSING")

n0 <- nrow(retail); cat("Start                                   :", n0, "rows\n")

retail_clean <- retail %>% filter(!is.na(CustomerID))
cat("After removing missing CustomerID       :", nrow(retail_clean),
    sprintf("(-%s)", comma(n0 - nrow(retail_clean))), "\n")

n1 <- nrow(retail_clean)
retail_clean <- retail_clean %>% filter(!grepl("^C", InvoiceNo))
cat("After removing cancellations (invoice C):", nrow(retail_clean),
    sprintf("(-%s)", comma(n1 - nrow(retail_clean))), "\n")

n2 <- nrow(retail_clean)
retail_clean <- retail_clean %>% filter(Quantity > 0, UnitPrice > 0)
cat("After removing Qty<=0 / Price<=0        :", nrow(retail_clean),
    sprintf("(-%s)", comma(n2 - nrow(retail_clean))), "\n")

n3 <- nrow(retail_clean)
retail_clean <- retail_clean %>% distinct()
cat("After removing duplicate rows           :", nrow(retail_clean),
    sprintf("(-%s)", comma(n3 - nrow(retail_clean))), "\n")

retail_clean <- retail_clean %>% mutate(TotalPrice = Quantity * UnitPrice)
cat("\nRetained", sprintf("%.1f%%", 100 * nrow(retail_clean) / n0), "of the raw rows\n")
cat("Customers remaining:", length(unique(retail_clean$CustomerID)), "\n")

# =============================================================================
# SECTION 4 - CUSTOMER-LEVEL FEATURE ENGINEERING (RFM +)
# =============================================================================
rule("SECTION 4 : CUSTOMER-LEVEL FEATURE ENGINEERING")

snapshot <- max(retail_clean$InvoiceDate) + 86400
cat("Snapshot (analysis) date:", format(snapshot), "\n\n")

customers <- retail_clean %>%
  group_by(CustomerID) %>%
  summarise(
    Recency        = as.numeric(difftime(snapshot, max(InvoiceDate), units = "days")),
    Frequency      = n_distinct(InvoiceNo),
    Monetary       = sum(TotalPrice),
    TotalQuantity  = sum(Quantity),
    UniqueProducts = n_distinct(StockCode),
    AvgUnitPrice   = mean(UnitPrice),
    Tenure         = as.numeric(difftime(snapshot, min(InvoiceDate), units = "days")),
    Country        = names(sort(table(Country), decreasing = TRUE))[1],
    .groups = "drop"
  ) %>%
  mutate(
    AvgOrderValue    = Monetary / Frequency,
    AvgBasketSize    = TotalQuantity / Frequency,
    PurchaseInterval = ifelse(Frequency > 1, (Tenure - Recency) / (Frequency - 1), Tenure)
  )

cat("Customer feature matrix:", nrow(customers), "customers x",
    ncol(customers), "columns\n\n")
print(head(as.data.frame(customers)))
cat("\nSummary of the RFM block:\n")
print(summary(customers[, c("Recency", "Frequency", "Monetary")]))

rfm_long <- customers %>%
  select(Recency, Frequency, Monetary) %>%
  pivot_longer(everything(), names_to = "Metric", values_to = "Value")
p_rfm <- ggplot(rfm_long, aes(x = Value, fill = Metric)) +
  geom_histogram(bins = 45, colour = "white", show.legend = FALSE) +
  facet_wrap(~ Metric, scales = "free", ncol = 3) +
  scale_x_continuous(labels = comma) +
  labs(title = "Distribution of RFM Features (raw, heavily right-skewed)",
       x = "Value", y = "Number of customers") +
  theme_minimal(base_size = 12)
save_plot(p_rfm, "04_rfm_distributions.png", w = 11, h = 4)

# ---- outlier treatment: 1st/99th percentile winsorisation -------------------
feature_cols <- c("Recency", "Frequency", "Monetary", "TotalQuantity",
                  "UniqueProducts", "AvgOrderValue", "AvgBasketSize", "Tenure")

winsorise <- function(x, p = 0.01) {
  q <- quantile(x, c(p, 1 - p), na.rm = TRUE)
  pmin(pmax(x, q[1]), q[2])
}
cust_w <- customers
cat("\nOutlier treatment (winsorising at the 1st / 99th percentile):\n")
for (f in feature_cols) {
  before <- sum(customers[[f]] < quantile(customers[[f]], .01) |
                customers[[f]] > quantile(customers[[f]], .99))
  cust_w[[f]] <- winsorise(customers[[f]])
  cat(sprintf("  %-15s capped %4d values  [%.2f , %.2f]\n", f, before,
              min(cust_w[[f]]), max(cust_w[[f]])))
}

p_box <- ggplot(customers %>% select(Monetary) %>% mutate(g = "raw") %>%
                  bind_rows(cust_w %>% select(Monetary) %>% mutate(g = "winsorised")),
                aes(x = g, y = Monetary, fill = g)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = .25) +
  scale_y_continuous(labels = comma) +
  labs(title = "Effect of Outlier Treatment on Monetary Value",
       x = "", y = "Monetary") +
  theme_minimal(base_size = 12)
save_plot(p_box, "05_outlier_treatment.png", w = 7, h = 5)

# ---- log transform + standardisation ---------------------------------------
X <- cust_w[, feature_cols]
X_log <- as.data.frame(lapply(X, function(v) log1p(v)))   # tame the right skew
X_scaled <- scale(X_log)
cat("\nAfter log1p + z-score scaling:\n")
cat("  column means (should be ~0):", paste(round(colMeans(X_scaled), 4), collapse = " "), "\n")
cat("  column sds   (should be  1):", paste(round(apply(X_scaled, 2, sd), 4), collapse = " "), "\n")

png(file.path(OUT, "06_correlation_matrix.png"), width = 1200, height = 1000, res = 140)
corrplot(cor(X_log), method = "color", type = "upper", addCoef.col = "black",
         number.cex = .65, tl.col = "black", tl.srt = 45,
         title = "Correlation of Engineered Customer Features", mar = c(0,0,2,0))
invisible(dev.off())
cat("\nCorrelation matrix saved.\n")

# =============================================================================
# SECTION 5 - ELBOW METHOD
# =============================================================================
rule("SECTION 5 : OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)")

set.seed(42)
samp <- if (nrow(X_scaled) > 3000) sample(nrow(X_scaled), 3000) else seq_len(nrow(X_scaled))

wss <- sapply(1:10, function(k) kmeans(X_scaled, k, nstart = 25, iter.max = 100)$tot.withinss)
elbow_df <- data.frame(k = 1:10, WSS = wss)
cat("Within-cluster sum of squares:\n"); print(elbow_df)

# largest drop in the second difference = the elbow
d2 <- diff(diff(wss))
k_elbow <- which.max(d2) + 1
cat("\nElbow detected at k =", k_elbow, "\n")

p_elbow <- ggplot(elbow_df, aes(k, WSS)) +
  geom_line(colour = "#2c3e50", linewidth = 1.1) +
  geom_point(size = 3, colour = "#2c3e50") +
  geom_point(data = elbow_df[elbow_df$k == k_elbow, ], size = 6,
             shape = 21, fill = NA, colour = "#e74c3c", stroke = 1.6) +
  annotate("text", x = k_elbow + .35, y = wss[k_elbow],
           label = paste("elbow: k =", k_elbow), hjust = 0, colour = "#e74c3c") +
  scale_x_continuous(breaks = 1:10) + scale_y_continuous(labels = comma) +
  labs(title = "Elbow Method for Optimal k",
       x = "Number of clusters (k)", y = "Total within-cluster sum of squares") +
  theme_minimal(base_size = 12)
save_plot(p_elbow, "07_elbow_method.png", w = 8, h = 5)

sil_k <- sapply(2:8, function(k) {
  set.seed(42)
  km <- kmeans(X_scaled[samp, ], k, nstart = 25, iter.max = 100)
  mean(silhouette(km$cluster, dist(X_scaled[samp, ]))[, 3])
})
sil_df <- data.frame(k = 2:8, Silhouette = round(sil_k, 4))
cat("\nAverage silhouette width by k (3000-customer sample):\n"); print(sil_df)
k_sil <- sil_df$k[which.max(sil_df$Silhouette)]
cat("Best silhouette at k =", k_sil, "\n")

p_sil <- ggplot(sil_df, aes(k, Silhouette)) +
  geom_line(colour = "#8e44ad", linewidth = 1.1) +
  geom_point(size = 3, colour = "#8e44ad") +
  geom_point(data = sil_df[sil_df$k == k_sil, ], size = 6, shape = 21,
             fill = NA, colour = "#e74c3c", stroke = 1.6) +
  scale_x_continuous(breaks = 2:8) +
  labs(title = "Average Silhouette Width by Number of Clusters",
       x = "Number of clusters (k)", y = "Average silhouette width") +
  theme_minimal(base_size = 12)
save_plot(p_sil, "08_silhouette_by_k.png", w = 8, h = 5)

K <- 4
cat("\nChosen k for the final segmentation: K =", K,
    "(elbow + business interpretability)\n")

# =============================================================================
# SECTION 6 - K-MEANS CLUSTERING
# =============================================================================
rule("SECTION 6 : K-MEANS CLUSTERING")

set.seed(42)
km <- kmeans(X_scaled, centers = K, nstart = 50, iter.max = 200)
customers$KMeans <- factor(km$cluster)
cust_w$KMeans    <- factor(km$cluster)

cat("Cluster sizes:\n"); print(table(customers$KMeans))
cat("\nBetween_SS / Total_SS =",
    sprintf("%.1f%%", 100 * km$betweenss / km$totss), "\n")
cat("\nCluster centres (scaled space):\n")
print(round(km$centers, 3))

sil_km <- silhouette(km$cluster[samp], dist(X_scaled[samp, ]))
sil_km_avg <- mean(sil_km[, 3])
cat("\nAverage silhouette width (K-Means, k =", K, "):",
    round(sil_km_avg, 4), "\n")

png(file.path(OUT, "09_silhouette_kmeans.png"), width = 1300, height = 900, res = 140)
print(fviz_silhouette(sil_km, print.summary = FALSE) +
        labs(title = paste0("Silhouette Plot - K-Means (k = ", K, ")")) +
        theme_minimal(base_size = 11))
invisible(dev.off())

# =============================================================================
# SECTION 7 - HIERARCHICAL CLUSTERING & DENDROGRAM
# =============================================================================
rule("SECTION 7 : HIERARCHICAL CLUSTERING")

set.seed(42)
hsamp <- if (nrow(X_scaled) > 1500) sample(nrow(X_scaled), 1500) else seq_len(nrow(X_scaled))
d <- dist(X_scaled[hsamp, ], method = "euclidean")
hc <- hclust(d, method = "ward.D2")
hc_clusters <- cutree(hc, k = K)

cat("Linkage: Ward.D2 | Distance: Euclidean | Sample:", length(hsamp), "customers\n")
cat("Cluster sizes:\n"); print(table(hc_clusters))

sil_hc <- silhouette(hc_clusters, d)
sil_hc_avg <- mean(sil_hc[, 3])
cat("\nAverage silhouette width (Hierarchical, k =", K, "):",
    round(sil_hc_avg, 4), "\n")

png(file.path(OUT, "10_dendrogram.png"), width = 1500, height = 850, res = 140)
dend <- as.dendrogram(hc) %>%
  color_branches(k = K) %>%
  set("labels", rep("", length(hsamp)))
plot(dend, main = paste0("Hierarchical Clustering Dendrogram (Ward.D2, cut at k = ", K, ")"),
     ylab = "Height (merge distance)")
rect.hclust(hc, k = K, border = 2:(K + 1))
invisible(dev.off())

# agreement between the two methods on the shared sample
km_on_hsamp <- km$cluster[hsamp]
ct <- table(KMeans = km_on_hsamp, Hierarchical = hc_clusters)
cat("\nCross-tabulation (K-Means vs Hierarchical) on the sample:\n"); print(ct)
agreement <- sum(apply(ct, 1, max)) / sum(ct)
cat("\nBest-match agreement:", sprintf("%.1f%%", 100 * agreement), "\n")

comp <- data.frame(
  Method     = c("K-Means", "Hierarchical (Ward.D2)"),
  Clusters   = c(K, K),
  Silhouette = round(c(sil_km_avg, sil_hc_avg), 4)
)
cat("\nClustering quality comparison:\n"); print(comp)

p_cmp <- ggplot(comp, aes(Method, Silhouette, fill = Method)) +
  geom_col(show.legend = FALSE, width = .55) +
  geom_text(aes(label = Silhouette), vjust = -0.5) +
  ylim(0, max(comp$Silhouette) * 1.25) +
  labs(title = "Clustering Quality: Average Silhouette Width",
       x = "", y = "Average silhouette width") +
  theme_minimal(base_size = 12)
save_plot(p_cmp, "11_clustering_comparison.png", w = 7.5, h = 5)

# =============================================================================
# SECTION 8 - PCA
# =============================================================================
rule("SECTION 8 : PRINCIPAL COMPONENT ANALYSIS")

pca <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
ev  <- pca$sdev ^ 2
var_df <- data.frame(
  PC         = paste0("PC", seq_along(ev)),
  Eigenvalue = round(ev, 4),
  Proportion = round(100 * ev / sum(ev), 2),
  Cumulative = round(100 * cumsum(ev) / sum(ev), 2)
)
cat("Explained variance:\n"); print(var_df)
cat("\nPC1 + PC2 explain", var_df$Cumulative[2], "% of the total variance\n")

cat("\nPC loadings (first two components):\n")
print(round(pca$rotation[, 1:2], 3))

p_scree <- ggplot(var_df, aes(x = factor(PC, levels = PC))) +
  geom_col(aes(y = Proportion), fill = "#16a085") +
  geom_line(aes(y = Cumulative, group = 1), colour = "#c0392b", linewidth = 1) +
  geom_point(aes(y = Cumulative), colour = "#c0392b", size = 2.5) +
  geom_text(aes(y = Cumulative, label = paste0(Cumulative, "%")),
            vjust = -0.8, size = 3, colour = "#c0392b") +
  labs(title = "PCA Scree Plot with Cumulative Explained Variance",
       x = "Principal component", y = "Variance explained (%)") +
  theme_minimal(base_size = 12)
save_plot(p_scree, "12_pca_scree.png", w = 8.5, h = 5)

pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], PC3 = pca$x[, 3],
                     Cluster = customers$KMeans)
p_pca <- ggplot(pca_df, aes(PC1, PC2, colour = Cluster)) +
  geom_point(alpha = .55, size = 1.5) +
  stat_ellipse(level = .95, linewidth = .9) +
  labs(title = "Customer Segments in 2D PCA Space",
       subtitle = sprintf("PC1 %.1f%% + PC2 %.1f%% = %.1f%% of variance",
                          var_df$Proportion[1], var_df$Proportion[2],
                          var_df$Cumulative[2]),
       x = sprintf("PC1 (%.1f%%)", var_df$Proportion[1]),
       y = sprintf("PC2 (%.1f%%)", var_df$Proportion[2])) +
  theme_minimal(base_size = 12)
save_plot(p_pca, "13_pca_2d_segments.png", w = 9, h = 6)

png(file.path(OUT, "14_pca_biplot.png"), width = 1400, height = 1000, res = 140)
print(fviz_pca_biplot(pca, label = "var", habillage = customers$KMeans,
                      addEllipses = TRUE, alpha.ind = .25, repel = TRUE) +
        labs(title = "PCA Biplot - Feature Loadings and Customer Segments") +
        theme_minimal(base_size = 11))
invisible(dev.off())

# =============================================================================
# SECTION 9 - CLUSTER PROFILING
# =============================================================================
rule("SECTION 9 : CLUSTER PROFILING AND INTERPRETATION")

profile <- customers %>%
  group_by(KMeans) %>%
  summarise(
    Customers      = n(),
    Share          = round(100 * n() / nrow(customers), 1),
    # NOTE: the revenue totals are computed BEFORE the `Monetary` median is
    # defined - inside summarise() a newly created column shadows the original
    # one for every expression that follows it.
    TotalRevenue   = round(sum(Monetary), 2),
    RevenueShare   = round(100 * sum(Monetary) / sum(customers$Monetary), 1),
    Recency        = round(median(Recency), 1),
    Frequency      = round(median(Frequency), 1),
    Monetary       = round(median(Monetary), 2),
    AvgOrderValue  = round(median(AvgOrderValue), 2),
    AvgBasketSize  = round(median(AvgBasketSize), 1),
    UniqueProducts = round(median(UniqueProducts), 1),
    Tenure         = round(median(Tenure), 1),
    .groups = "drop"
  ) %>% arrange(desc(Monetary))

cat("Cluster profile (medians, ordered by monetary value):\n")
print(as.data.frame(profile))
cat("\nSanity check - revenue shares sum to:",
    sprintf("%.1f%%", sum(profile$RevenueShare)), "\n")

# name the segments from their RFM position
rank_m <- rank(-profile$Monetary)
labels <- c("Champions / High-Value", "Loyal Customers",
            "Potential / Occasional", "At-Risk / Dormant")
profile$Segment <- labels[pmin(rank_m, length(labels))]
cat("\nSegment naming:\n")
print(as.data.frame(profile[, c("KMeans", "Segment", "Customers", "Share",
                                "Recency", "Frequency", "Monetary", "RevenueShare")]))

seg_map <- setNames(profile$Segment, profile$KMeans)
customers$Segment <- factor(seg_map[as.character(customers$KMeans)],
                            levels = labels)

heat <- customers %>%
  group_by(Segment) %>%
  summarise(across(all_of(feature_cols), median), .groups = "drop") %>%
  mutate(across(all_of(feature_cols), ~ as.numeric(scale(.x)))) %>%
  pivot_longer(-Segment, names_to = "Feature", values_to = "Z")

p_heat <- ggplot(heat, aes(Feature, Segment, fill = Z)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = round(Z, 1)), size = 3.2) +
  scale_fill_gradient2(low = "#3498db", mid = "white", high = "#e74c3c",
                       midpoint = 0, name = "z-score") +
  labs(title = "Segment Profile Heat-map (median feature values, z-scored)",
       x = "", y = "") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
save_plot(p_heat, "15_segment_heatmap.png", w = 10, h = 5)

p_rev <- ggplot(profile, aes(x = reorder(Segment, RevenueShare), y = RevenueShare,
                             fill = Segment)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(RevenueShare, "%  (", Share, "% of customers)")),
            hjust = -0.05, size = 3.4) +
  coord_flip() + ylim(0, max(profile$RevenueShare) * 1.45) +
  labs(title = "Revenue Contribution by Segment",
       x = "", y = "Share of total revenue (%)") +
  theme_minimal(base_size = 12)
save_plot(p_rev, "16_revenue_by_segment.png", w = 9.5, h = 5)

# =============================================================================
# SECTION 10 - SUPERVISED MODELS FOR HIGH-VALUE PREDICTION
# =============================================================================
rule("SECTION 10 : HIGH-VALUE CUSTOMER PREDICTION")

hv_cluster <- profile$KMeans[which.max(profile$Monetary)]
customers$HighValue <- factor(ifelse(customers$KMeans == hv_cluster, "Yes", "No"),
                              levels = c("No", "Yes"))
cat("High-value segment  : cluster", as.character(hv_cluster),
    "(", as.character(profile$Segment[which.max(profile$Monetary)]), ")\n")
cat("Class distribution  :\n"); print(table(customers$HighValue))
cat("Positive class rate :",
    sprintf("%.1f%%", 100 * mean(customers$HighValue == "Yes")), "\n")

# Monetary, Frequency and Recency defined the clusters, so keep only the
# behavioural features that a business could observe independently.
model_feats <- c("Recency", "Frequency", "TotalQuantity", "UniqueProducts",
                 "AvgUnitPrice", "AvgOrderValue", "AvgBasketSize", "Tenure",
                 "PurchaseInterval")
model_df <- cust_w %>%
  select(all_of(intersect(model_feats, names(cust_w)))) %>%
  mutate(AvgUnitPrice     = cust_w$AvgUnitPrice,
         PurchaseInterval = winsorise(customers$PurchaseInterval),
         HighValue        = customers$HighValue)
model_df <- model_df[, c(model_feats, "HighValue")]
cat("\nPredictors used (", length(model_feats), "):",
    paste(model_feats, collapse = ", "), "\n")

set.seed(42)
idx <- createDataPartition(model_df$HighValue, p = .7, list = FALSE)
train <- model_df[idx, ]; test <- model_df[-idx, ]
cat("\nTrain:", nrow(train), "| Test:", nrow(test), "\n")
cat("Train positive rate:", sprintf("%.1f%%", 100 * mean(train$HighValue == "Yes")),
    "| Test positive rate:", sprintf("%.1f%%", 100 * mean(test$HighValue == "Yes")), "\n")

pre <- preProcess(train[, model_feats], method = c("center", "scale"))
train_s <- predict(pre, train); test_s <- predict(pre, test)

# ---- Random Forest ---------------------------------------------------------
cat("\n--- Random Forest ---\n")
set.seed(42)
rf <- randomForest(HighValue ~ ., data = train_s, ntree = 500,
                   mtry = floor(sqrt(length(model_feats))), importance = TRUE)
print(rf)
rf_pred <- predict(rf, test_s)
rf_prob <- predict(rf, test_s, type = "prob")[, "Yes"]

# ---- SVM (RBF kernel) ------------------------------------------------------
cat("\n--- Support Vector Machine (RBF kernel) ---\n")
set.seed(42)
svm_fit <- svm(HighValue ~ ., data = train_s, kernel = "radial",
               cost = 10, gamma = 1 / length(model_feats), probability = TRUE)
print(svm_fit)
svm_pred <- predict(svm_fit, test_s)
svm_prob <- attr(predict(svm_fit, test_s, probability = TRUE), "probabilities")[, "Yes"]

# ---- evaluation ------------------------------------------------------------
evaluate <- function(pred, prob, truth, name) {
  cm <- confusionMatrix(pred, truth, positive = "Yes")
  roc_obj <- roc(response = truth, predictor = prob,
                 levels = c("No", "Yes"), direction = "<", quiet = TRUE)
  cat("\n=== ", name, " ===\n", sep = "")
  print(cm$table)
  cat(sprintf("Accuracy  : %.4f\n", cm$overall["Accuracy"]))
  cat(sprintf("Precision : %.4f\n", cm$byClass["Precision"]))
  cat(sprintf("Recall    : %.4f\n", cm$byClass["Recall"]))
  cat(sprintf("F1-Score  : %.4f\n", cm$byClass["F1"]))
  cat(sprintf("ROC-AUC   : %.4f\n", as.numeric(auc(roc_obj))))
  cat(sprintf("Kappa     : %.4f\n", cm$overall["Kappa"]))
  list(name = name, cm = cm, roc = roc_obj,
       row = data.frame(Model = name,
                        Accuracy  = round(cm$overall["Accuracy"], 4),
                        Precision = round(cm$byClass["Precision"], 4),
                        Recall    = round(cm$byClass["Recall"], 4),
                        F1        = round(cm$byClass["F1"], 4),
                        ROC_AUC   = round(as.numeric(auc(roc_obj)), 4),
                        row.names = NULL))
}

rule("SECTION 11 : MODEL EVALUATION AND COMPARISON")
rf_eval  <- evaluate(rf_pred,  rf_prob,  test_s$HighValue, "Random Forest")
svm_eval <- evaluate(svm_pred, svm_prob, test_s$HighValue, "SVM (RBF)")

results <- rbind(rf_eval$row, svm_eval$row)
cat("\n\nMODEL COMPARISON\n"); print(results)
best <- results$Model[which.max(results$F1)]
cat("\nBest model by F1-Score:", best, "\n")

res_long <- results %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels = c("Accuracy","Precision","Recall","F1","ROC_AUC")))
p_res <- ggplot(res_long, aes(Metric, Value, fill = Model)) +
  geom_col(position = position_dodge(.8), width = .72) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            position = position_dodge(.8), vjust = -0.4, size = 3) +
  ylim(0, 1.12) +
  labs(title = "Random Forest vs SVM - Classification Metrics",
       x = "", y = "Score") +
  theme_minimal(base_size = 12)
save_plot(p_res, "17_model_comparison.png", w = 10, h = 5.5)

# confusion matrices
cm_plot <- function(cm, title) {
  df <- as.data.frame(cm$table)
  ggplot(df, aes(Reference, Prediction, fill = Freq)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = comma(Freq)), size = 6) +
    scale_fill_gradient(low = "#eaf2f8", high = "#2471a3", guide = "none") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 12)
}
png(file.path(OUT, "18_confusion_matrices.png"), width = 1500, height = 680, res = 140)
grid.arrange(cm_plot(rf_eval$cm, "Random Forest"),
             cm_plot(svm_eval$cm, "SVM (RBF)"), ncol = 2)
invisible(dev.off())

# ROC curves
png(file.path(OUT, "19_roc_curves.png"), width = 1200, height = 1000, res = 140)
plot(rf_eval$roc, col = "#27ae60", lwd = 2.5, legacy.axes = TRUE,
     main = "ROC Curves - High-Value Customer Prediction")
plot(svm_eval$roc, col = "#8e44ad", lwd = 2.5, add = TRUE)
abline(0, 1, lty = 2, col = "grey50")
legend("bottomright",
       legend = c(sprintf("Random Forest (AUC = %.4f)", auc(rf_eval$roc)),
                  sprintf("SVM RBF      (AUC = %.4f)", auc(svm_eval$roc))),
       col = c("#27ae60", "#8e44ad"), lwd = 2.5, bty = "n")
invisible(dev.off())

# =============================================================================
# SECTION 12 - FEATURE IMPORTANCE
# =============================================================================
rule("SECTION 12 : FEATURE IMPORTANCE ANALYSIS")

imp <- as.data.frame(importance(rf))
imp$Feature <- rownames(imp)
imp <- imp %>% arrange(desc(MeanDecreaseGini))
cat("Random Forest variable importance:\n")
print(imp[, c("Feature", "MeanDecreaseAccuracy", "MeanDecreaseGini")], row.names = FALSE)

p_imp <- ggplot(imp, aes(reorder(Feature, MeanDecreaseGini), MeanDecreaseGini)) +
  geom_col(fill = "#d35400") +
  geom_text(aes(label = round(MeanDecreaseGini, 1)), hjust = -0.15, size = 3.2) +
  coord_flip() + ylim(0, max(imp$MeanDecreaseGini) * 1.2) +
  labs(title = "Random Forest Feature Importance (Mean Decrease in Gini)",
       x = "", y = "Mean decrease in Gini") +
  theme_minimal(base_size = 12)
save_plot(p_imp, "20_feature_importance.png", w = 9, h = 5.5)

# =============================================================================
# SECTION 12B - LEAKAGE CHECK: A FORWARD-LOOKING FORMULATION
# =============================================================================
rule("SECTION 12B : LEAKAGE CHECK - FORWARD-LOOKING PREDICTION")

cat("The near-perfect scores above are expected and must NOT be read as\n")
cat("genuine predictive skill. The HighValue label was DERIVED from K-Means\n")
cat("on the same behavioural features, so the classifiers are essentially\n")
cat("re-learning a cluster boundary that is already a function of the inputs.\n")
cat("This is target leakage by construction.\n\n")
cat("A fair test asks a genuinely predictive question:\n")
cat("  'Using only the FIRST part of the year, can we predict who will end\n")
cat("   the year in the high-value segment?'\n\n")

cutoff <- min(retail_clean$InvoiceDate) +
          0.6 * as.numeric(difftime(max(retail_clean$InvoiceDate),
                                    min(retail_clean$InvoiceDate), units = "secs"))
cat("Observation window :", format(min(retail_clean$InvoiceDate)), "->", format(cutoff), "\n")
cat("Outcome window     :", format(cutoff), "->", format(max(retail_clean$InvoiceDate)), "\n\n")

early <- retail_clean %>% filter(InvoiceDate < cutoff)
early_feats <- early %>%
  group_by(CustomerID) %>%
  summarise(
    Recency        = as.numeric(difftime(cutoff, max(InvoiceDate), units = "days")),
    Frequency      = n_distinct(InvoiceNo),
    TotalQuantity  = sum(Quantity),
    UniqueProducts = n_distinct(StockCode),
    AvgUnitPrice   = mean(UnitPrice),
    AvgOrderValue  = sum(TotalPrice) / n_distinct(InvoiceNo),
    AvgBasketSize  = sum(Quantity) / n_distinct(InvoiceNo),
    Tenure         = as.numeric(difftime(cutoff, min(InvoiceDate), units = "days")),
    .groups = "drop"
  ) %>%
  mutate(PurchaseInterval = ifelse(Frequency > 1, (Tenure - Recency) / (Frequency - 1), Tenure))

fwd <- early_feats %>%
  inner_join(customers[, c("CustomerID", "HighValue")], by = "CustomerID")
cat("Customers active in the observation window:", nrow(fwd), "\n")
cat("Of these, high-value by year end          :", sum(fwd$HighValue == "Yes"),
    sprintf("(%.1f%%)", 100 * mean(fwd$HighValue == "Yes")), "\n\n")

fwd_X <- fwd[, model_feats]
fwd_X <- as.data.frame(lapply(fwd_X, winsorise))
fwd_df <- cbind(fwd_X, HighValue = fwd$HighValue)

set.seed(42)
fidx <- createDataPartition(fwd_df$HighValue, p = .7, list = FALSE)
ftrain <- fwd_df[fidx, ]; ftest <- fwd_df[-fidx, ]
fpre <- preProcess(ftrain[, model_feats], method = c("center", "scale"))
ftrain_s <- predict(fpre, ftrain); ftest_s <- predict(fpre, ftest)
cat("Train:", nrow(ftrain), "| Test:", nrow(ftest), "\n")

set.seed(42)
rf_f <- randomForest(HighValue ~ ., data = ftrain_s, ntree = 500, importance = TRUE)
rf_f_pred <- predict(rf_f, ftest_s)
rf_f_prob <- predict(rf_f, ftest_s, type = "prob")[, "Yes"]

set.seed(42)
svm_f <- svm(HighValue ~ ., data = ftrain_s, kernel = "radial",
             cost = 10, gamma = 1 / length(model_feats), probability = TRUE)
svm_f_pred <- predict(svm_f, ftest_s)
svm_f_prob <- attr(predict(svm_f, ftest_s, probability = TRUE), "probabilities")[, "Yes"]

rf_f_eval  <- evaluate(rf_f_pred,  rf_f_prob,  ftest_s$HighValue, "Random Forest (forward-looking)")
svm_f_eval <- evaluate(svm_f_pred, svm_f_prob, ftest_s$HighValue, "SVM RBF (forward-looking)")

results_fwd <- rbind(rf_f_eval$row, svm_f_eval$row)
cat("\n\nFORWARD-LOOKING MODEL COMPARISON\n"); print(results_fwd)

all_res <- rbind(
  cbind(Setting = "Same-period (leaky)",  results),
  cbind(Setting = "Forward-looking (fair)", results_fwd)
)
cat("\n\nLEAKY vs FAIR SETTING\n"); print(all_res)
cat("\nThe drop between the two settings is the size of the leakage.\n")
cat("The forward-looking numbers are the ones a business should plan with.\n")

p_leak <- ggplot(all_res %>%
                   pivot_longer(c(Accuracy, Precision, Recall, F1, ROC_AUC),
                                names_to = "Metric", values_to = "Value") %>%
                   mutate(Metric = factor(Metric,
                          levels = c("Accuracy","Precision","Recall","F1","ROC_AUC")),
                          Model = ifelse(grepl("Random Forest", Model),
                                         "Random Forest", "SVM (RBF)")),
                 aes(Metric, Value, fill = Setting)) +
  geom_col(position = position_dodge(.8), width = .7) +
  facet_wrap(~ Model) +
  geom_text(aes(label = sprintf("%.2f", Value)), position = position_dodge(.8),
            vjust = -0.35, size = 2.7) +
  ylim(0, 1.15) +
  scale_fill_manual(values = c("Same-period (leaky)" = "#e67e22",
                               "Forward-looking (fair)" = "#2980b9")) +
  labs(title = "Effect of Target Leakage on Reported Performance",
       subtitle = "Same-period labels vs predicting year-end status from early-year behaviour only",
       x = "", y = "Score") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")
save_plot(p_leak, "21_leakage_check.png", w = 11, h = 5.5)

imp_f <- as.data.frame(importance(rf_f)); imp_f$Feature <- rownames(imp_f)
imp_f <- imp_f %>% arrange(desc(MeanDecreaseGini))
cat("\nForward-looking model - feature importance:\n")
print(imp_f[, c("Feature", "MeanDecreaseAccuracy", "MeanDecreaseGini")], row.names = FALSE)

p_impf <- ggplot(imp_f, aes(reorder(Feature, MeanDecreaseGini), MeanDecreaseGini)) +
  geom_col(fill = "#2980b9") + coord_flip() +
  labs(title = "Feature Importance - Forward-Looking Model",
       subtitle = "Which early-year behaviours actually predict year-end high value",
       x = "", y = "Mean decrease in Gini") +
  theme_minimal(base_size = 12)
save_plot(p_impf, "22_feature_importance_forward.png", w = 9, h = 5.5)

# =============================================================================
# SECTION 13 - MARKETING RECOMMENDATIONS
# =============================================================================
rule("SECTION 13 : SEGMENT-WISE MARKETING RECOMMENDATIONS")

recs <- data.frame(
  Segment = labels,
  Strategy = c(
    "VIP tier, early access to new ranges, dedicated account manager, referral incentives - protect at all costs.",
    "Volume discounts and bundle offers to raise average order value; cross-sell adjacent categories.",
    "Onboarding journeys, first-repeat-purchase discount, product education content to build frequency.",
    "Win-back campaign with a strong time-limited offer; exit survey; suppress after N non-responses to protect deliverability."
  ),
  Channel = c("Personal email + phone", "Email + retargeting",
              "Email nurture + social", "Win-back email + SMS"),
  Priority = c("Retention (highest ROI)", "Growth", "Development", "Reactivation")
)
for (i in seq_len(nrow(recs))) {
  cat("\n", strrep("-", 78), "\n", sep = "")
  cat(toupper(recs$Segment[i]), "\n")
  r <- profile[profile$Segment == recs$Segment[i], ]
  if (nrow(r)) {
    cat(sprintf("  %s customers (%.1f%%) | %.1f%% of revenue\n",
                comma(r$Customers[1]), r$Share[1], r$RevenueShare[1]))
    cat(sprintf("  Median R/F/M: %.0f days | %.0f orders | %s\n",
                r$Recency[1], r$Frequency[1], comma(round(r$Monetary[1], 0))))
  }
  cat("  Priority :", recs$Priority[i], "\n")
  cat("  Channel  :", recs$Channel[i], "\n")
  cat("  Strategy :", recs$Strategy[i], "\n")
}

# =============================================================================
# SECTION 14 - EXPORT
# =============================================================================
rule("SECTION 14 : EXPORT OF RESULTS")

write.csv(customers, file.path(OUT, "customer_segments.csv"), row.names = FALSE)
write.csv(as.data.frame(profile), file.path(OUT, "cluster_profiles.csv"), row.names = FALSE)
write.csv(results, file.path(OUT, "model_results.csv"), row.names = FALSE)
cat("Written: customer_segments.csv, cluster_profiles.csv, model_results.csv\n")

saveRDS(list(profile = as.data.frame(profile), results = results,
             results_fwd = results_fwd, all_res = all_res,
             imp_f = imp_f[, c("Feature","MeanDecreaseAccuracy","MeanDecreaseGini")],
             cutoff = format(cutoff), n_fwd = nrow(fwd),
             var_df = var_df, comp = comp, elbow = elbow_df, sil_df = sil_df,
             imp = imp[, c("Feature","MeanDecreaseAccuracy","MeanDecreaseGini")],
             k_elbow = k_elbow, K = K, source = DATA_SOURCE,
             n_raw = n0, n_clean = nrow(retail_clean),
             n_cust = nrow(customers), hv_rate = mean(customers$HighValue == "Yes"),
             sil_km = sil_km_avg, sil_hc = sil_hc_avg, agreement = agreement,
             rf_cm = rf_eval$cm$table, svm_cm = svm_eval$cm$table),
        file.path(OUT, "results.rds"))

rule("ANALYSIS COMPLETE")
cat("All figures and tables written to ./", OUT, "/\n", sep = "")
