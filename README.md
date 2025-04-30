# Python

# 🎬 IMDb Top 1000 Movies - Exploratory Data Analysis (EDA)

This project presents an Exploratory Data Analysis (EDA) of the **IMDb Top 1000 Movies** dataset. It explores trends and insights from movies based on their ratings, genre, duration, gross earnings, and other features using Python libraries like Pandas, Matplotlib, and Seaborn.

---

## 📁 Dataset

- **Source**: [IMDb Top 1000 Movies dataset (CSV format)](https://data.world/melanithapa/movie-data/workspace/file?filename=IMDB_top_1000.csv)
- **Size**: 1000 rows × 10 columns
- **Features Include**:
  - `Title`: Name of the movie
  - `Certificate`: Age rating
  - `Duration`: Length of the movie
  - `Genre`: Movie genres
  - `Rate`: IMDb rating
  - `Metascore`: Metacritic score
  - `Gross`: Box office gross earnings
  - `Cast` and `Director` (extracted from text)

---

## 📊 Objective

- Analyze and visualize trends in top-rated IMDb movies.
- Understand relationships between IMDb rating, Metascore, and Gross earnings.
- Identify the most common genres, certificates, and directors.
- Detect outliers and explore data cleaning decisions.

---

## 📌 Key Analysis & Decisions

### 🔍 Data Cleaning
- **Dropped Nulls**: Chose to drop rows with missing values instead of filling them with mean/median/forward fill to avoid introducing bias or assumptions not supported by the data.

### 📈 Insights
- **Top Genres & Directors**: Visualized frequency of genres and directors with multiple top-rated movies.
- **Rating Analysis**: Found correlation between IMDb rating and Metascore.
- **Gross Earnings**: Detected outliers using a boxplot, showing a few high-grossing movies skewing the distribution.

### 📦 Outlier Detection
- Used **boxplot analysis** on Gross to identify high-grossing outliers.
- These outliers were **retained** as they represent genuinely popular/blockbuster films.

---

## 🛠️ Tools Used

- **Python 3.13**
- **Pandas**, **NumPy** for data manipulation
- **Seaborn**, **Matplotlib** for visualizations
- **VSCode** for development

---

## 📷 Visuals

- Barplots of Genre, Certificate, and Director distributions
- Boxplot of Gross earnings showing outliers
- Scatterplots for correlation analysis

---

## 📌 Conclusion

This EDA helped uncover interesting patterns in IMDb's top-rated films. The analysis highlights the dominance of certain genres and directors, the skewed nature of box office revenues, and the reasonable correlation between critic and user ratings.
