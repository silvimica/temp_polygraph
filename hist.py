import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file into a DataFrame
file_path = 'token_similarity_for_hist.csv'  # Replace with your actual file path
df = pd.read_csv(file_path, header = None)

df.columns = ['removed token','context' , 'similarity score']

bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

df['score_bins'] = pd.cut(df['similarity score'], bins)

token_counts = df.groupby('score_bins')['removed token'].count()

plt.figure(figsize=(10, 6))
plt.bar(token_counts.index.astype(str), token_counts, width=0.8)
plt.xticks(rotation=90)
plt.xlabel('Similarity Score Range')
plt.ylabel('Count of Tokens')
plt.title('Histogram of Tokens by Similarity Score Range')
plt.tight_layout()

plt.savefig('token_similarity_histogram_unique.png')
plt.close()  #
