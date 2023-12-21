echo "# Book Recommendation Engine using K-Nearest Neighbors

This project implements a book recommendation algorithm using K-Nearest Neighbors (KNN) with the Book-Crossings dataset. The provided Google Colab notebook contains the code for data loading, preprocessing, model training, and a recommendation function.

## Getting Started

1. Open the [Google Colab notebook](<https://colab.research.google.com/drive/1BqdT8K2120ELJxX91xyzlMcvkpz2kTZs#scrollTo=ytu_Rx73ptin>).
2. Create a copy in your account or locally.
3. Run each cell sequentially.

## Project Structure

### Cell 1: Imports and Loading Data

```python
# Import libraries
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load and merge datasets
books = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv")
ratings = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")
df = pd.merge(ratings, books, on='book_id')

# Remove users with < 200 ratings and books with < 100 ratings
user_counts = df['user_id'].value_counts()
book_counts = df['book_id'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= 200].index)]
df = df[df['book_id'].isin(book_counts[book_counts >= 100].index)]
