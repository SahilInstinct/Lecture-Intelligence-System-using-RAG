import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from read_chunks import create_embedding
import joblib

df = joblib.load('embeddings.joblib')

incoming_query = input("Enter your query:  ")
question_embedding = create_embedding([incoming_query])[0]

print("Question embedding shape:", np.array(question_embedding).shape)

similarity = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

top_results = 3
max_idx = similarity.argsort()[::-1][:top_results]

new_df = df.iloc[max_idx]
print(new_df)