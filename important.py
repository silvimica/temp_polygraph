from sentence_transformers import CrossEncoder
import torch 

# Initialize the CrossEncoder with the specified model
model_name = "cross-encoder/stsb-roberta-large"
crossencoder = CrossEncoder(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

# List of important words
important_words = [
    'capital', 'Tamil Nadu', 'Chennai', 'rules', 'Narnia', 'High King Peter', 'Chronicles of Narnia', 'C.S. Lewis',
    'Prince Caspian', 'motor manufacturer', 'Movano', 'vans', 'mini-buses', 'Vauxhall', 'BBC', 'TV programme',
    'John Craven', 'Julia Bradbury', 'Matt Baker', 'Countryfile', 'Search for Extra-Terrestrial Intelligence', 'SETI',
    'extra-terrestrial life', 'scientific methods', 'electromagnetic transmissions', 'Vendredi', 'French', 'day',
    'week', 'Friday'
]


claim_level = [['capital', 'Indian state', 'Tamil Nadu', 'Chennai'] ,
               ['rules', 'Narnia', 'High King Peter', 'Chronicles of Narnia', 'C.S. Lewis', 'Prince Caspian'] ,
                ['motor manufacturer', 'Movano', 'vans', 'mini-buses', 'Vauxhall'] , 
                ['BBC', 'magazine style', 'TV programme', 'John Craven', 'Julia Bradbury', 'Matt Baker', 'Countryfile'],
                ['Search for Extra-Terrestrial Intelligence', 'SETI', 'extra-terrestrial life', 'scientific methods', 'electromagnetic transmissions'],
                ['Vendredi', 'French', 'day', 'week', 'Friday']]

# Concatenate the important words into a single string
original_string = ', '.join(important_words)

def compute_similarity(text1, text2):
    # Compute similarity score using the cross-encoder model
    score = crossencoder.predict([(text1, text2)])
    return score[0]

# Compute similarity scores by removing one important word at a time
results = []
for word in important_words:
    # Create a modified list with the current word removed
    modified_words = [w for w in important_words if w != word]
    modified_string = ', '.join(modified_words)
    
    # Compute similarity score between the full list and the modified list
    score = compute_similarity(original_string, modified_string)
    results.append((word, score))

# Display the results
for word, score in results:
    print(f"Word removed: {word}, Similarity score: {score:.4f}")

results2 = []

for claim in claim_level:
    original_string = ', '.join(claim)
    for word in claim:
        modified_words = [w for w in claim if w != word]
        modified_string = ', '.join(modified_words)
        score = compute_similarity(original_string, modified_string)
        results2.append((claim, word, score))


for claim, word, score in results2:
    print(f"Claim: {claim}, Word removed: {word}, Similarity score: {score:.4f}")


