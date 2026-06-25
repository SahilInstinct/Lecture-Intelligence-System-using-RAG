import numpy as np 
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from create_embeddings import create_embedding
import joblib

def Inference(prompt):
     r = requests.post("http://localhost:11434/api/generate", json={
        "model": 'llama3.2',
        "prompt": prompt,
        "Stream": False
    })
     response = r.json()
     print(response)
     return response["response"]


df = joblib.load('embeddings.joblib')

incoming_query = input("Enter your query:  ")
question_embedding = create_embedding([incoming_query])[0]

print("Question embedding shape:", np.array(question_embedding).shape)

similarity = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

top_results = 4
max_idx = similarity.argsort()[::-1][:top_results]

new_df = df.iloc[max_idx]
# print(new_df[['title', 'number', 'text']])


prompt = f"""
There are tutorials from web development from the Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[['title','number','start','end','text']].to_json(orient= "records")}
-----------------------------------------------------
"{incoming_query}" 
User asked this question related to the video chunks, you have to answer in a human way ( don't mention the above format that's just for you) where and how much content is taught in which video (in which video and at what timestamp convert that to mins and seconds) and guide the user to that particular video. If user asks anything irrelevant, tell him that you can ask only answers and questions related to the course. Don't ask any follow up questions.
"""

with open('prompt.txt', 'w') as f:
    f.write(prompt)
    
response = Inference(prompt)
print(response)

with open("response.txt","w") as f:
    f.write(response)
    
    
for index, row in new_df.iterrows():
    print(index, row['title'], row['number'], row['text'], row['start'], row['end'])

