import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": 'bge-m3',
        "input": text_list
    })
    print("Status:", r.status_code)
    print("Response:", r.text[:1000])
    
    data = r.json()

    if "embeddings" not in data:
        raise Exception(f"Unexpected API response: {data}")

    return data["embeddings"]


jsons = os.listdir('jsons') # Get all the json files in the jsons folder
my_dict = [] # Create an empty list to store the chunks with embeddings
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}","r") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    
    for i, chunk in enumerate(content['chunks']):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)
        
        
df = pd.DataFrame.from_records(my_dict)
print(df.head())