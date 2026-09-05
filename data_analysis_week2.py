# ============================================================
# WEEK 2 TASK: EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# Dataset: Netflix Movies and TV Shows
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

df = pd.read_csv("netflix_titles.csv")

print("Dataset loaded successfully!")


# ============================================================
# 3. VIEW FIRST 5 ROWS
# ============================================================

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. CHECK DATASET SHAPE
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ============================================================
# 5. CHECK COLUMN NAMES
# ============================================================

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 6. CHECK DATASET INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive Statistics:")
print(df.describe(include="all"))


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
missing_values = df.isnull().sum()

print(missing_values)


# Visualize missing values
plt.figure(figsize=(12, 6))

sns.barplot(
    x=missing_values.values,
    y=missing_values.index
)

plt.title("Missing Values by Column")
plt.xlabel("Number of Missing Values")
plt.ylabel("Column")

plt.show()


# ============================================================
# 9. CHECK DUPLICATE VALUES
# ============================================================

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 10. HANDLE MISSING VALUES
# ============================================================

# Fill missing text values with "Unknown"

text_columns = [
    "director",
    "cast",
    "country",
    "rating",
    "duration"
]

for col in text_columns:
    df[col] = df[col].fillna("Unknown")


# Convert date_added into datetime format

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)


# Check missing values again

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ============================================================
# 11. REMOVE DUPLICATE ROWS
# ============================================================

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)


# ============================================================
# 12. DATA TRANSFORMATION
# ============================================================

# Extract year from date_added

df["year_added"] = df["date_added"].dt.year

# Extract month from date_added

df["month_added"] = df["date_added"].dt.month

print("\nNew Columns Added:")
print(["year_added", "month_added"])


# ============================================================
# 13. CHECK CLEANED DATASET
# ============================================================

print("\nCleaned Dataset:")
print(df.head())

print("\nCleaned Dataset Information:")
print(df.info())


# ============================================================
# 14. MOVIES VS TV SHOWS
# ============================================================

print("\nContent Type Counts:")
print(df["type"].value_counts())


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="type"
)

plt.title("Distribution of Movies and TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.show()


# ============================================================
# 15. MOVIE AND TV SHOW PERCENTAGE
# ============================================================

type_percentage = df["type"].value_counts(normalize=True) * 100

print("\nContent Type Percentage:")
print(type_percentage.round(2))


# ============================================================
# 16. RELEASE YEAR DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 6))

plt.hist(
    df["release_year"],
    bins=30
)

plt.title("Distribution of Netflix Titles by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.show()


# ============================================================
# 17. TOP 10 CONTENT RATINGS
# ============================================================

rating_counts = df["rating"].value_counts().head(10)

print("\nTop 10 Content Ratings:")
print(rating_counts)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=rating_counts.values,
    y=rating_counts.index
)

plt.title("Top 10 Content Ratings on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Rating")

plt.show()


# ============================================================
# 18. TITLES ADDED TO NETFLIX BY YEAR
# ============================================================

yearly_additions = (
    df["year_added"]
    .value_counts()
    .sort_index()
)

print("\nTitles Added by Year:")
print(yearly_additions)


plt.figure(figsize=(12, 6))

plt.plot(
    yearly_additions.index,
    yearly_additions.values,
    marker="o"
)

plt.title("Netflix Titles Added by Year")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.grid(True)

plt.show()


# ============================================================
# 19. TOP 10 COUNTRIES
# ============================================================

country_data = (
    df["country"]
    .str.split(", ")
    .explode()
)

top_countries = country_data.value_counts().head(10)

print("\nTop 10 Countries:")
print(top_countries)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries by Number of Netflix Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.show()


# ============================================================
# 20. TOP 10 CONTENT CATEGORIES
# ============================================================

category_data = (
    df["listed_in"]
    .str.split(", ")
    .explode()
)

top_categories = category_data.value_counts().head(10)

print("\nTop 10 Content Categories:")
print(top_categories)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_categories.values,
    y=top_categories.index
)

plt.title("Top 10 Netflix Content Categories")
plt.xlabel("Number of Titles")
plt.ylabel("Category")

plt.show()


# ============================================================
# 21. MOVIES AND TV SHOWS BY RELEASE YEAR
# ============================================================

plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x="release_year",
    hue="type",
    bins=30,
    multiple="stack"
)

plt.title("Movies and TV Shows by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.show()


# ============================================================
# 22. MOVIES VS TV SHOWS ADDED EACH YEAR
# ============================================================

type_year = (
    df.groupby(["year_added", "type"])
    .size()
    .reset_index(name="count")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=type_year,
    x="year_added",
    y="count",
    hue="type",
    marker="o"
)

plt.title("Movies and TV Shows Added to Netflix by Year")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.legend(title="Content Type")

plt.show()


# ============================================================
# 23. TOP DIRECTORS
# ============================================================

director_counts = (
    df[df["director"] != "Unknown"]["director"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Directors:")
print(director_counts)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=director_counts.values,
    y=director_counts.index
)

plt.title("Top 10 Directors by Number of Netflix Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Director")

plt.show()


# ============================================================
# 24. DURATION ANALYSIS
# ============================================================

# Separate movies and TV shows

movies = df[df["type"] == "Movie"].copy()

tv_shows = df[df["type"] == "TV Show"].copy()


# Extract numerical movie duration

movies["duration_minutes"] = (
    movies["duration"]
    .str.extract("(\d+)")
    .astype(float)
)


print("\nMovie Duration Statistics:")
print(movies["duration_minutes"].describe())


# Plot movie duration distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    movies["duration_minutes"].dropna(),
    bins=30
)

plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")

plt.show()


# ============================================================
# 25. MOVIE DURATION BOX PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=movies["duration_minutes"]
)

plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (Minutes)")

plt.show()


# ============================================================
# 26. CORRELATION ANALYSIS
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================================
# 27. RELEASE YEAR VS YEAR ADDED
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="release_year",
    y="year_added",
    hue="type",
    alpha=0.6
)

plt.title("Release Year vs Year Added to Netflix")
plt.xlabel("Release Year")
plt.ylabel("Year Added")

plt.legend(title="Content Type")

plt.show()


# ============================================================
# 28. FINAL DATASET SUMMARY
# ============================================================

print("\n========== FINAL DATASET SUMMARY ==========")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nContent Types:")
print(df["type"].value_counts())

print("\nTop Ratings:")
print(df["rating"].value_counts().head())

print("\nTop Countries:")
print(top_countries.head())

print("\nTop Categories:")
print(top_categories.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 29. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "netflix_titles_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully as:")
print("netflix_titles_cleaned.csv")


# ============================================================
# END OF WEEK 2 EDA PROJECT
# ============================================================