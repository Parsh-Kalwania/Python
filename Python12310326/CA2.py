import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset\\IMDB_top_1000.csv")  

# Basic overview
print(df.head())
print(df.info())
print(df.describe(include='all'))

# Cleaning & Preprocessing
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)
df['Year'] = df['Title'].str.extract(r'\((\d{4})\)').astype(float)
df['Votes'] = df['Info'].str.extract(r'Votes:\s([\d,]+)')[0].str.replace(',', '').astype(float)
df['Gross'] = df['Info'].str.extract(r'Gross:\s\$(\d+\.\d+)M')[0].astype(float)
df['Genre'] = df['Genre'].fillna('').apply(lambda x: x.split(', '))
df['Main Genre'] = df['Genre'].apply(lambda x: x[0] if x else None)
df['Rate'] = df['Rate'].astype(float)

# Objective 1: How strongly are IMDb rating and metascore related on a scale from -1 to 1

df_corr = df[['Rate', 'Metascore']].dropna()

correlation = df_corr['Rate'].corr(df_corr['Metascore'])
print(f"Correlation: {correlation:.2f}")

plt.figure(figsize=(8, 6))
sns.regplot(x='Metascore', y='Rate', data=df_corr, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title(f"IMDb Rating vs Metascore\nCorrelation: {correlation:.2f}", fontsize=14)
plt.xlabel("Metascore")
plt.ylabel("IMDb Rating")
plt.show()

# Objective 2: Genre-wise average IMDb rating — are horror movies rated better than action?

genre_rating = df[['Main Genre', 'Rate']].dropna()

genre_avg = genre_rating.groupby('Main Genre')['Rate'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_avg.index, y=genre_avg.values, palette='coolwarm')

for index, value in enumerate(genre_avg.values):
    plt.text(index, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

plt.title("Average IMDb Rating by Genre", fontsize=16)
plt.xlabel("Main Genre")
plt.ylabel("Average IMDb Rating")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Objective 3: Gross earnings vs IMDb rating — does quality equal money?

clean_df = df.dropna(subset=['Gross', 'Rate'])

plt.figure(figsize=(10,6))
sns.scatterplot(data=clean_df, x='Rate', y='Gross', hue='Certificate', alpha=0.7)

plt.title('Gross Earnings vs IMDb Rating: Does Quality Equal Money?')
plt.xlabel('IMDb Rating')
plt.ylabel('Gross Earnings (in Millions)')
plt.legend(title='Certificate')
plt.grid(True)
plt.tight_layout()
plt.show()

# Objective 4: Trend of Number of Movies in IMDb Top 1000 Over the Years

movies_per_year = df['Year'].value_counts().sort_index().reset_index()
movies_per_year.columns = ['Year', 'Number of Movies']

plt.figure(figsize=(14, 6))

sns.lineplot(
    x='Year', 
    y='Number of Movies', 
    data=movies_per_year, 
    marker='o', 
    color='royalblue',
    linewidth=2.5
)

plt.title('Trend of Number of Movies in IMDb Top 1000 Over the Years', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.grid(linestyle='--', alpha=0.6)

plt.xticks(rotation=45)

peak_year = movies_per_year.loc[movies_per_year['Number of Movies'].idxmax()]
plt.annotate(
    f'Peak: {int(peak_year["Year"])} ({int(peak_year["Number of Movies"])} movies)',
    xy=(peak_year['Year'], peak_year['Number of Movies']),
    xytext=(10, 10), 
    textcoords='offset points',
    arrowprops=dict(arrowstyle='->', color='red'))
    
plt.tight_layout()
plt.show()

# Objective 5: Analyze the covariance between IMDb ratings and Metascore to understand if higher-rated movies on IMDb also tend to get higher Metascores from critics.”

df_clean = df.dropna(subset=['Rate', 'Metascore'])

covariance = df_clean[['Rate', 'Metascore']].cov().iloc[0, 1]
print(f"Covariance between IMDb Rating and Metascore: {covariance:.2f}")

plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='Rate', y='Metascore', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('IMDb Rating vs Metascore')
plt.xlabel('IMDb Rating')
plt.ylabel('Metascore')
plt.grid(True)
plt.tight_layout()
plt.show()

# Objective 6: Top 10 Most Frequent Movie Certificates

top_certificates = df['Certificate'].value_counts().nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_certificates.values, y=top_certificates.index, palette='viridis')
plt.title('Top 10 Most Frequent Movie Certificates')
plt.xlabel('Number of Movies')
plt.ylabel('Certificate')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Objective 7: Identify which genres have the most number of movies: Are dramas really more liked than action

df_exploded = df.explode('Genre')

genre_counts = df_exploded['Genre'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Number of Movies']

top_genres = genre_counts.head(15)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Number of Movies', y='Genre', data=top_genres, palette='viridis')

for p in ax.patches:
    width = p.get_width()
    plt.text(
        width + 1,
        p.get_y() + p.get_height() / 2,
        f'{int(width)}',
        ha='left',
        va='center'
    )

plt.title('Top Genres by Number of Movies in IMDb Top 1000', fontsize=16)
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Objective 8: See if popularity (votes) correlates with commercial success (gross). 

df_votes_gross = df.dropna(subset=['Votes', 'Gross'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_votes_gross, x='Votes', y='Gross', alpha=0.6)
sns.regplot(data=df_votes_gross, x='Votes', y='Gross', scatter=False, color='red')  # Optional trendline

plt.title('Votes vs Gross Earnings')
plt.xlabel('Votes (Popularity)')
plt.ylabel('Gross Earnings (in Millions USD)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Objective 9: Understand if longer movies tend to have higher/lower ratings.

plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x='Duration', y='Rate', alpha=0.6)
sns.regplot(data=df, x='Duration', y='Rate', scatter=False, color='red')

plt.title('IMDb Rating vs Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('IMDb Rating')
plt.grid(True)
plt.tight_layout()
plt.show()

#Objective 10: Top 10 directors with highest number of movies in top 1000 IMDb ratings

df['Director'] = df['Cast'].str.extract(r'Director:\s([^\|]+)')

director_counts = df['Director'].value_counts().reset_index()
director_counts.columns = ['Director', 'Number of Movies']

top_directors = director_counts.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='Number of Movies', y='Director', data=top_directors, palette='viridis')
plt.title('Top 10 Directors with Most Movies in IMDb Top 1000', fontsize=16)
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Director', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Objective 11: Calculate correlations between numerical features

corr_matrix = df[['Rate', 'Duration', 'Year', 'Votes', 'Gross']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    center=0,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Strength'}
)
plt.title('Correlation Between Movie Features\n(Rating, Runtime, Year, Votes, Gross)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()