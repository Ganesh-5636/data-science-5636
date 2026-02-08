import pandas as pd

df = pd.read_csv("C:/Users/Ganesh/Downloads/top_rated_movies.csv")
# 1. head() command
print("\n1. head() - First 5 rows:")
print(df.head())
print()

# 2. info() command
print("2. info() - Dataset information:")
df.info()
print()

# 3. describe() command
print("3. describe() - Statistical summary:")
print(df.describe(include='all'))
print()

# 4. shape command
print("4. shape - Dataset dimensions:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print()

# Technique 1: Check for missing values
print("\n🔹 TECHNIQUE 1: Check missing values")
missing = df.isnull().sum()
print("Missing values in each column:")
print(missing)
print()

# Technique 2: Check for duplicates
print("🔹 TECHNIQUE 2: Check duplicate rows")
duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
print()

# Technique 3: Remove duplicates
print("🔹 TECHNIQUE 3: Remove duplicates")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"Removed {duplicates} duplicate rows")
else:
    print("No duplicates to remove")

print(f"Rows after cleaning: {df.shape[0]}")
print()

# Technique 4:Standardize column names (Renaming)
print("\n🔹 TECHNIQUE 4: Standardize column names (Renaming)")

print("Original columns:")
print(df.columns.tolist())
print()

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Cleaned columns:")
print(df.columns.tolist())
print()

# Technique  5: Fix data types
print("\n🔹 TECHNIQUE 5: Fix data types")

print("Before fixing - Data types:")
print(df.dtypes)
print()

# Fix release_date column
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Fix numeric columns
numeric_cols = ['vote_average', 'vote_count', 'popularity']

for col in numeric_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("After fixing - Data types:")
print(df.dtypes)
print()

#Technique  6: Extract date features
print("\n🔹 TECHNIQUE 6: Extract date features")

# Convert release_date to datetime (SAFETY FIX)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Extract features
df['year'] = df['release_date'].dt.year
df['month'] = df['release_date'].dt.month
df['day'] = df['release_date'].dt.day
df['day_of_week'] = df['release_date'].dt.day_name()

print("Added columns: year, month, day, day_of_week")
print()

#Technique 7: Extract exact release date
print("\n🔹 TECHNIQUE 7: Extract exact release date")

# Ensure datetime format (safety)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Create formatted date column (YYYY-MM-DD)
df['exact_release_date'] = df['release_date'].dt.strftime('%Y-%m-%d')

# Print sample dates
print("Sample exact release dates:")
print(df['exact_release_date'].head(10))

print("\nDate range in dataset:")
print("Earliest movie date:", df['exact_release_date'].min())
print("Latest movie date:", df['exact_release_date'].max())
print()

#Technique  8: Create release time categories
print("\n🔹 TECHNIQUE 8: Create release time categories")

# Step 1: Ensure release_date exists and is datetime
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
else:
    raise Exception("❌ Column 'release_date' not found in dataset")

# Step 2: Ensure year column exists
if 'year' not in df.columns:
    df['year'] = df['release_date'].dt.year
    print("✔ 'year' column created from release_date")

# Step 3: Create categories
def release_time_category(year):
    if pd.isna(year):
        return 'Unknown'
    elif year < 2000:
        return 'Classic Era'
    elif year < 2010:
        return 'Early 2000s'
    elif year < 2020:
        return 'Modern Era'
    else:
        return 'Recent Era'

df['release_time_category'] = df['year'].apply(release_time_category)

print("Created categories: Classic Era, Early 2000s, Modern Era, Recent Era")
print()

# Preview
print(df[['title', 'release_date', 'year', 'release_time_category']].head(10))
print()

#Technique  9: Binary encoding
print("\n🔹 TECHNIQUE 9: Binary encoding")

df['is_high_rated'] = df['vote_average'].apply(lambda x: 1 if x >= 7 else 0)

popularity_threshold = df['popularity'].median()
df['is_popular'] = df['popularity'].apply(lambda x: 1 if x >= popularity_threshold else 0)

if 'original_language' in df.columns:
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
else:
    df['is_english'] = 0   # fallback if column not present

print("Created binary columns:")
print(" - is_high_rated (rating >= 7)")
print(" - is_popular (above median popularity)")
print(" - is_english (English language movie)")
print()

print(df[['title', 'vote_average', 'popularity', 'is_high_rated', 'is_popular', 'is_english']].head(10))
print()


#Technique10: Create rating categories
print("\n🔹 TECHNIQUE 10: Create rating categories")

def rating_category(rating):
    if pd.isna(rating):
        return 'Unknown'
    elif rating < 5:
        return 'Low'
    elif rating < 7:
        return 'Medium'
    else:
        return 'High'
df['rating_category'] = df['vote_average'].apply(rating_category)

print("Rating categories created: Low, Medium, High")
print()

print(df[['title', 'vote_average', 'rating_category']].head(10))
print()

 #Technique  11: Create movie category groups
print("\n🔹 TECHNIQUE 11: Create movie category groups (fallback)")

def movie_category(rating):
    if rating >= 8:
        return "Premium Movies"
    elif rating >= 7:
        return "Popular Movies"
    elif rating >= 5:
        return "Average Movies"
    else:
        return "Low Rated Movies"

df['movie_category_group'] = df['vote_average'].apply(movie_category)
print("Movie category groups created:")
print(" - Premium Movies")
print(" - Popular Movies")
print(" - Average Movies")
print(" - Low Rated Movies")
print()
print(df[['title', 'vote_average', 'movie_category_group']].head(10))
print()

#Technique  12: Create peak popularity indicator
print("\n🔹 TECHNIQUE 12: Create peak popularity indicator")

popularity_threshold = df['popularity'].quantile(0.75)

df['is_peak_popularity'] = df['popularity'].apply(
    lambda x: 1 if x >= popularity_threshold else 0
)

print("Peak popularity threshold:", popularity_threshold)
print("Created binary column: is_peak_popularity")
print("1 = High demand movie, 0 = Normal demand movie")
print()

print(df[['title', 'popularity', 'is_peak_popularity']].head(10))
print()

#Technique  13: Create yearly movie aggregation
print("\n🔹 TECHNIQUE 13: Create yearly movie aggregation")
if 'year' not in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    print("✔ 'year' column created")

yearly_stats = df.groupby('year').agg(
    total_movies=('title', 'count'),
    avg_rating=('vote_average', 'mean'),
    avg_popularity=('popularity', 'mean'),
    total_votes=('vote_count', 'sum')
).reset_index()

print("Yearly aggregated statistics created")
print()
df = df.merge(yearly_stats, on='year', how='left')

print("Added yearly aggregated columns to main dataset")
print()
print(df[['title', 'year', 'total_movies', 'avg_rating', 'avg_popularity', 'total_votes']].head(10))
print()

#Technique  14: Rating distribution (Histogram)
print("\n🔹 TECHNIQUE 14: Rating distribution (Histogram)")

plt.figure(figsize=(10, 5))
plt.hist(df['vote_average'], bins=30, edgecolor='black')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Movie Rating (vote_average)')
plt.ylabel('Number of Movies')
plt.grid(True, alpha=0.3)
plt.show()

print()

Technique 15: Top 10 Most Popular Movies
print("\n🔹 TECHNIQUE 15: Top 10 Most Popular Movies")
top_popular = df.sort_values(by='popularity', ascending=False).head(10)

plt.figure(figsize=(14, 6))

plt.bar(
    top_popular['title'],
    top_popular['popularity'],
    edgecolor='black'
)
plt.title('Top 10 Most Popular Movies', fontsize=16, fontweight='bold')
plt.xlabel('Movie Title', fontsize=12)
plt.ylabel('Popularity Score', fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("Top 10 popular movies bar chart generated")
print()

#Technique  16: Movies released per year (Line Chart)
print("\n🔹 TECHNIQUE 16: Movies released per year (Line Chart)")

if 'year' not in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
yearly_movies = df.groupby('year')['title'].count()

plt.figure(figsize=(12, 5))
yearly_movies.plot(kind='line', marker='o')

plt.title('Movies Released Per Year (TMDB)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Yearly movie release trend plotted")
print()

Technique  17: Correlation analysis (Heatmap)
print("\n🔹 TECHNIQUE 17: Correlation analysis (Heatmap)")
numeric_cols = [
    'popularity',
    'vote_average',
    'vote_count',
    'runtime'
]
numeric_cols = [col for col in numeric_cols if col in df.columns]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('TMDB Movie Data Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Correlation heatmap generated successfully")
print()

 Technique 18: Movie title length analysis
 print("\n🔹 TECHNIQUE 18: Movie title length analysis")

df['title_length'] = df['title'].astype(str).str.len()

plt.figure(figsize=(10, 5))
plt.hist(df['title_length'], bins=20, alpha=0.7, edgecolor='black')
plt.title('Movie Title Length Distribution')
plt.xlabel('Title Length (characters)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

print("Title length feature created")
print()

Technique 19: Most common words in movie titles
print("\n🔹 TECHNIQUE 19: Most common words in movie titles")

all_words = ' '.join(df['title'].astype(str).str.lower()).split()


word_counts = Counter(all_words)

top_words = word_counts.most_common(15)

print("\nTop 15 most common words in movie titles:")
for word, count in top_words:
    print(f"  {word:15s} : {count} times")

print()


Technique  20 (Alt): One-hot encoding of genres
print("\n🔹 TECHNIQUE 20 (Alt): One-hot encoding of genres")

genre_col = None
for col in df.columns:
    if 'genre' in col.lower():
        genre_col = col
        break

if genre_col:
    genre_dummies = df[genre_col].astype(str).str.get_dummies(sep=',')
    df = pd.concat([df, genre_dummies], axis=1)

    print(f"Created {genre_dummies.shape[1]} genre dummy variables")
    print("Genre columns:", list(genre_dummies.columns))
else:
    print("No genre column found in dataset")
print()

