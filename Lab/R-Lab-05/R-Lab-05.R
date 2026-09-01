# ==============================================================================
# Title: Social Network Analysis (SNA) with R - Text & Term Association Network
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 1: Install & Load Required Packages
# ------------------------------------------------------------------------------
required_packages <- c("tm", "igraph", "RColorBrewer")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(tm)
library(igraph)
library(RColorBrewer)

# ------------------------------------------------------------------------------
# Step 2: Read and Prepare Data
# ------------------------------------------------------------------------------
cat("Select your CSV dataset file (e.g., apple.csv or any text dataset with a 'text' column)\n")
apple <- read.csv(file.choose(), header = TRUE, stringsAsFactors = FALSE)

# Build Corpus using universal UTF-8 encoding (Works across Windows, Mac, and Linux)
corpus <- iconv(apple$text, to = "UTF-8", sub = "byte")
corpus <- Corpus(VectorSource(corpus))

# ------------------------------------------------------------------------------
# Step 3: Text Cleaning (Compatible with modern 'tm' package)
# ------------------------------------------------------------------------------
# Custom URL removal function
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus <- tm_map(corpus, content_transformer(removeNumbers))
cleanset <- tm_map(corpus, removeWords, stopwords("english"))
cleanset <- tm_map(cleanset, content_transformer(removeURL))
cleanset <- tm_map(cleanset, removeWords, c("aapl", "apple"))

# Custom word replacement
replace_word <- content_transformer(function(x, pattern, replacement) {
  return(gsub(pattern, replacement, x))
})
cleanset <- tm_map(cleanset, replace_word, pattern = "stocks", replacement = "stock")
cleanset <- tm_map(cleanset, stripWhitespace)

# ------------------------------------------------------------------------------
# Step 4: Build Term-Document Matrix (TDM)
# ------------------------------------------------------------------------------
tdm <- TermDocumentMatrix(cleanset)
tdm_m <- as.matrix(tdm)

# Filter out infrequent terms (keep terms appearing more than 30 times)
# Note: Lower the threshold (e.g., > 10) if your dataset is smaller
frequent_rows <- rowSums(tdm_m) > 30
if (sum(frequent_rows) == 0) {
  warning("No terms exceeded threshold > 30. Lowering threshold to > 5.")
  tdm_m <- tdm_m[rowSums(tdm_m) > 5, ]
} else {
  tdm_m <- tdm_m[frequent_rows, ]
}

cat("Top 10 terms and first 10 documents matrix:\n")
print(tdm_m[1:min(10, nrow(tdm_m)), 1:min(10, ncol(tdm_m))])

# ------------------------------------------------------------------------------
# Step 5: Network of Terms (Co-occurrence Network)
# ------------------------------------------------------------------------------
# Convert matrix to boolean adjacency (term co-occurrence)
tdm_binary <- tdm_m
tdm_binary[tdm_binary > 1] <- 1

termM <- tdm_binary %*% t(tdm_binary)

# Create undirected graph
g_term <- graph_from_adjacency_matrix(termM, weighted = TRUE, mode = "undirected", diag = FALSE)
g_term <- simplify(g_term)

V(g_term)$label <- V(g_term)$name
V(g_term)$degree <- degree(g_term)

# Plot Histogram of Node Degrees
hist(V(g_term)$degree,
     breaks = 20,
     col = "limegreen",
     main = "Histogram of Term Node Degrees",
     ylab = "Frequency",
     xlab = "Degree of Vertices (Connections)")

# Plot Basic Network Diagram
set.seed(222)
plot(g_term,
     vertex.color = "lightgreen",
     vertex.size = 6,
     vertex.label.dist = 1.2,
     main = "Term Co-occurrence Network")

# ------------------------------------------------------------------------------
# Step 6: Community Detection Algorithms
# ------------------------------------------------------------------------------
par(mfrow = c(1, 3))

# Edge Betweenness
comm_eb <- cluster_edge_betweenness(g_term)
plot(comm_eb, g_term, main = "Edge Betweenness")

# Label Propagation
comm_lp <- cluster_label_prop(g_term)
plot(comm_lp, g_term, main = "Label Propagation")

# Fast Greedy
comm_fg <- cluster_fast_greedy(g_term)
plot(comm_fg, g_term, main = "Fast Greedy")

par(mfrow = c(1, 1))

# ------------------------------------------------------------------------------
# Step 7: Centrality Analysis (Hubs & Authorities)
# ------------------------------------------------------------------------------
hs <- hub_score(g_term, weights = NA)$vector
as <- authority_score(g_term, weights = NA)$vector

par(mfrow = c(1, 2))
plot(g_term, vertex.size = hs * 30, main = "Hubs (Term Importance)",
     vertex.label = V(g_term)$name, vertex.label.cex = 0.8,
     vertex.color = rainbow(length(V(g_term))))

plot(g_term, vertex.size = as * 30, main = "Authorities (Term Centrality)",
     vertex.label = V(g_term)$name, vertex.label.cex = 0.8,
     vertex.color = rainbow(length(V(g_term))))
par(mfrow = c(1, 1))

# ------------------------------------------------------------------------------
# Step 8: Highlighting High-Degree Term Nodes
# ------------------------------------------------------------------------------
max_deg <- max(V(g_term)$degree)
V(g_term)$label.cex <- (1.5 * V(g_term)$degree / max_deg) + 0.5
V(g_term)$label.color <- rgb(0, 0, 0.3, 0.9)
V(g_term)$frame.color <- NA

# Edge weight normalization for visual opacity
edge_weights <- E(g_term)$weight
if (is.null(edge_weights)) edge_weights <- rep(1, ecount(g_term))
egam <- (log(edge_weights + 1)) / max(log(edge_weights + 1))
E(g_term)$color <- rgb(0.5, 0.5, 0, egam)
E(g_term)$width <- egam * 2

plot(g_term,
     vertex.color = "lightgreen",
     vertex.size = V(g_term)$degree * 0.8,
     main = "Weighted Term Co-occurrence Network")

# ------------------------------------------------------------------------------
# Step 9: Network of Documents / Tweets
# ------------------------------------------------------------------------------
# Transpose TDM to find document-to-document similarity network
tweetM <- t(tdm_binary) %*% tdm_binary
g_tweet <- graph_from_adjacency_matrix(tweetM, weighted = TRUE, mode = "undirected", diag = FALSE)
g_tweet <- simplify(g_tweet)

V(g_tweet)$degree <- degree(g_tweet)

# Degree Histogram for Tweets
hist(V(g_tweet)$degree,
     breaks = 20,
     col = "lightblue",
     main = "Histogram of Tweet Node Degrees",
     ylab = "Frequency",
     xlab = "Degree")

# Plot Full Tweet Network
plot(g_tweet, 
     vertex.label = NA, 
     vertex.size = 3, 
     vertex.color = "skyblue", 
     main = "Tweet Similarity Network")

# Filter Low-Degree Nodes (Pruning disconnected or weak nodes)
degree_threshold <- quantile(V(g_tweet)$degree, 0.75) # Retain top 25% connected tweets
g2 <- delete.vertices(g_tweet, V(g_tweet)[degree(g_tweet) < degree_threshold])

plot(g2,
     vertex.size = 4,
     vertex.label = NA,
     vertex.color = "coral",
     main = paste("Filtered Tweet Network (Degree >=", round(degree_threshold), ")"))

cat("\nScript executed successfully!\n")
