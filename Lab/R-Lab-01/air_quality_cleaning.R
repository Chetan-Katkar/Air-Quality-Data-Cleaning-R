# ==============================================================================
# ASSIGNMENT 1: AIR-QUALITY DATA CLEANING USING R
# Dataset : PRSA_Data_Aotizhongxin_20130301-20170228.csv
# Author  : Chetan Katkar
# GitHub  : https://github.com/Chetan-Katkar/Air-Quality-Data-Cleaning-R
# ==============================================================================

# ==============================================================================
# TASK 1: Import and Inspect the Dataset
# Purpose: Read the CSV file with tryCatch() for graceful error handling,
#          then display head(), str(), dimensions, and total missing values.
# ==============================================================================
file_path <- "PRSA_Data_Aotizhongxin_20130301-20170228.csv"

air_data <- tryCatch({
  # Check if file exists before attempting to read
  if (!file.exists(file_path)) {
    stop("Error: The specified CSV file was not found in the working directory.")
  }
  df <- read.csv(file_path, stringsAsFactors = FALSE)
  cat(">>> File imported successfully.\n")
  df
}, error = function(e) {
  message("Import Error: ", e$message)
  return(NULL)
})

# Display inspection results only if import succeeded
if (!is.null(air_data)) {
  cat("\n--- First Six Records (head) ---\n")
  print(head(air_data))

  cat("\n--- Structure of the Dataset (str) ---\n")
  str(air_data)

  cat("\n--- Dimensions ---\n")
  cat("Rows:", nrow(air_data), " | Columns:", ncol(air_data), "\n")

  cat("\n--- Total Missing Values ---\n")
  cat("Contains missing values:", any(is.na(air_data)), "\n")
  cat("Total missing values across all columns:", sum(is.na(air_data)), "\n")
}

# ==============================================================================
# TASK 2: Understand the Difference Between NA, NULL, and NaN
# Purpose: Demonstrate each special value type and the functions used to
#          detect them: is.na(), is.null(), is.nan().
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 2: NA vs. NULL vs. NaN Demonstration\n")
cat("=====================================================================\n")

# --- NA (Not Available) ---
# Represents a missing or unknown observation within a data structure.
# NA preserves the position in a vector; it acts as a placeholder.
temp_na <- c(28, 30, NA, 32)
cat("NA vector       :", temp_na, "\n")
cat("is.na() result  :", is.na(temp_na), "\n\n")

# --- NULL ---
# Represents the absence of an object entirely. It has zero length and
# cannot occupy a position inside a vector.
missing_object <- NULL
cat("NULL object     : NULL\n")
cat("is.null() result:", is.null(missing_object), "\n")
cat("Length of NULL  :", length(missing_object), "\n\n")

# --- NaN (Not a Number) ---
# A special numeric value that results from undefined mathematical
# operations (e.g., 0/0). Note: is.na(NaN) also returns TRUE.
undefined_value <- 0 / 0
cat("NaN value       :", undefined_value, "\n")
cat("is.nan() result :", is.nan(undefined_value), "\n")
cat("is.na(NaN)      :", is.na(undefined_value), "\n")

# ==============================================================================
# TASK 3: Create a Missing-Value Summary Function
# Purpose: Build missing_summary(df, vars) that returns a data frame with
#          Variable, Total Records, Missing Values, and Missing Percentage.
#          A warning is raised if any variable exceeds 20% missing.
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 3: Missing-Value Summary Function\n")
cat("=====================================================================\n")

missing_summary <- function(data, variables) {
  summary_list <- list()

  for (var in variables) {
    if (var %in% names(data)) {
      total_records <- nrow(data)
      missing_count <- sum(is.na(data[[var]]))
      missing_pct   <- round((missing_count / total_records) * 100, 2)

      # Warn user if missing percentage exceeds 20%
      if (missing_pct > 20) {
        warning(paste("Variable", var, "has", missing_pct,
                       "% missing values (exceeds 20% threshold)."))
      }

      summary_list[[length(summary_list) + 1]] <- data.frame(
        Variable           = var,
        Total_Records      = total_records,
        Missing_Values     = missing_count,
        Missing_Percentage = missing_pct,
        stringsAsFactors   = FALSE
      )
    } else {
      warning(paste("Variable", var, "not found in the dataset."))
    }
  }

  do.call(rbind, summary_list)
}

# Apply the function to the seven specified variables
selected_vars <- c("PM2.5", "PM10", "SO2", "NO2", "TEMP", "WSPM", "wd")
summary_table_before <- missing_summary(air_data, selected_vars)

cat("\n--- Missing-Value Summary Table (Before Cleaning) ---\n")
print(summary_table_before)

# Store missing counts before cleaning for Task 8 & 9 comparison
missing_before <- setNames(summary_table_before$Missing_Values,
                           summary_table_before$Variable)

# ==============================================================================
# TASK 4: Identify Invalid Numerical Results (NA, NaN, Inf)
# Purpose: Create a pollution_ratio column (PM2.5 / PM10), detect invalid
#          values (NaN and Inf), and replace them with NA.
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 4: Pollution Ratio & Invalid-Value Handling\n")
cat("=====================================================================\n")

# Create the derived column
air_data$pollution_ratio <- air_data$PM2.5 / air_data$PM10

# Identify invalid values using dedicated checking functions
na_count  <- sum(is.na(air_data$pollution_ratio))
nan_count <- sum(is.nan(air_data$pollution_ratio))
inf_count <- sum(is.infinite(air_data$pollution_ratio))

cat("NA  values in pollution_ratio:", na_count, "\n")
cat("NaN values in pollution_ratio:", nan_count, "\n")
cat("Inf values in pollution_ratio:", inf_count, "\n")

# Replace NaN and Infinite values with NA for consistent handling
air_data$pollution_ratio[is.nan(air_data$pollution_ratio) |
                         is.infinite(air_data$pollution_ratio)] <- NA

cat("After cleanup — total NAs in pollution_ratio:",
    sum(is.na(air_data$pollution_ratio)), "\n")

# ==============================================================================
# TASK 5: Handle Missing Numerical Values Using a For Loop
# Purpose: Loop through numerical pollutant/weather columns, count NAs,
#          compute median(na.rm = TRUE), impute, and print before/after.
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 5: Loop-Based Median Imputation (Numerical Variables)\n")
cat("=====================================================================\n")

numeric_variables <- c("PM2.5", "PM10", "SO2", "NO2", "TEMP", "WSPM")

for (var in numeric_variables) {
  if (var %in% names(air_data)) {
    na_before <- sum(is.na(air_data[[var]]))
    med_val   <- median(air_data[[var]], na.rm = TRUE)

    # Impute: replace NAs with the column median
    air_data[[var]][is.na(air_data[[var]])] <- med_val
    na_after <- sum(is.na(air_data[[var]]))

    cat("Variable:", var, "\n")
    cat("  Missing Before :", na_before, "\n")
    cat("  Median Used    :", med_val, "\n")
    cat("  Missing After  :", na_after, "\n")
    cat("  ---\n")
  } else {
    cat("  [SKIP] Variable", var, "does not exist in the dataset.\n")
  }
}

# ==============================================================================
# TASK 6: Handle Missing Categorical Values (Mode Imputation)
# Purpose: Create a calculate_mode() function, apply it to the `wd` (wind
#          direction) column, and print before/after missing counts.
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 6: Mode Imputation for Categorical Variable (wd)\n")
cat("=====================================================================\n")

calculate_mode <- function(x) {
  # Remove NAs, tabulate frequencies, return the most frequent value
  x_clean    <- x[!is.na(x)]
  freq_table <- table(x_clean)
  names(freq_table)[which.max(freq_table)]
}

wd_missing_before <- sum(is.na(air_data$wd))
wd_mode           <- calculate_mode(air_data$wd)
air_data$wd[is.na(air_data$wd)] <- wd_mode
wd_missing_after  <- sum(is.na(air_data$wd))

cat("Variable: wd\n")
cat("  Missing Before :", wd_missing_before, "\n")
cat("  Mode Used      :", wd_mode, "\n")
cat("  Missing After  :", wd_missing_after, "\n")

# ==============================================================================
# TASK 7: Reusable Error-Handling Function (clean_variable)
# Purpose: Wrap median imputation inside tryCatch() so errors (non-existent
#          column, non-numeric type, 100% NA) are caught gracefully.
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 7: Reusable clean_variable() with Error Handling\n")
cat("=====================================================================\n")

clean_variable <- function(data, var_name) {
  tryCatch({
    # Step 1: Check if the variable exists in the data frame
    if (!var_name %in% names(data)) {
      stop(paste("Variable", var_name, "does not exist in the dataset."))
    }

    var_data <- data[[var_name]]

    # Step 2: Check that the variable is numerical
    if (!is.numeric(var_data)) {
      stop(paste("Variable", var_name, "is not a numerical variable."))
    }

    # Step 3: Check that the column is not 100% NA
    if (all(is.na(var_data))) {
      stop(paste("Variable", var_name, "is entirely NA — cannot impute."))
    }

    # Step 4: Compute median and impute
    med_val <- median(var_data, na.rm = TRUE)
    if (is.na(med_val)) {
      stop(paste("Median could not be computed for variable", var_name))
    }

    var_data[is.na(var_data)] <- med_val
    cat("  clean_variable(): Successfully cleaned", var_name, "\n")
    return(var_data)

  }, error = function(e) {
    message("  clean_variable() Error [", var_name, "]: ", e$message)
    return(NULL)
  })
}

# Demonstrate the function with three test cases
cat("\n--- Demonstration ---\n")
test_valid   <- clean_variable(air_data, "PRES")       # Valid numerical column
test_not_num <- clean_variable(air_data, "wd")          # Categorical → error
test_no_var  <- clean_variable(air_data, "FAKE_COL")    # Non-existent → error

# ==============================================================================
# TASK 8: Comparison Table — Missing Values Before vs. After Cleaning
# Purpose: Build and print a data frame showing Variable, Missing Before,
#          Missing After, and Values Replaced.
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 8: Comparison Table (Before vs. After Cleaning)\n")
cat("=====================================================================\n")

summary_table_after <- missing_summary(air_data, selected_vars)
missing_after <- setNames(summary_table_after$Missing_Values,
                          summary_table_after$Variable)

comparison_table <- data.frame(
  Variable        = selected_vars,
  Missing_Before  = as.integer(missing_before[selected_vars]),
  Missing_After   = as.integer(missing_after[selected_vars]),
  Values_Replaced = as.integer(missing_before[selected_vars] -
                               missing_after[selected_vars]),
  row.names       = NULL
)

cat("\n--- Comparison Table ---\n")
print(comparison_table)

# ==============================================================================
# TASK 9: Visualization — Grouped Bar Chart
# Purpose: Generate a side-by-side bar chart comparing missing values
#          before and after cleaning, with title, axis labels, and legend.
# ==============================================================================
cat("\n=====================================================================\n")
cat("TASK 9: Generating Grouped Bar Chart\n")
cat("=====================================================================\n")

# Prepare a matrix for the grouped barplot
plot_matrix <- rbind(
  Before = comparison_table$Missing_Before,
  After  = comparison_table$Missing_After
)
colnames(plot_matrix) <- comparison_table$Variable

# Draw the chart
barplot(
  plot_matrix,
  beside      = TRUE,
  col         = c("steelblue", "darkgreen"),
  main        = "Missing Values: Before vs. After Data Cleaning",
  xlab        = "Variables",
  ylab        = "Number of Missing Values",
  legend.text = c("Before Cleaning", "After Cleaning"),
  args.legend = list(x = "topright", bty = "n"),
  ylim        = c(0, max(plot_matrix, na.rm = TRUE) * 1.2)
)

cat(">>> Bar chart generated successfully.\n")

# ==============================================================================
# TASK 10: Export the Cleaned Dataset
# Purpose: Write the final cleaned data frame to a CSV file without row names.
# ==============================================================================
write.csv(air_data, file = "cleaned_air_quality_data.csv", row.names = FALSE)
cat("\n>>> Cleaned dataset exported as 'cleaned_air_quality_data.csv'.\n")
cat(">>> All 10 tasks completed successfully.\n")