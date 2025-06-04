from sklearn.neighbors import KNeighborsClassifier
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from word2number import w2n
import gensim.downloader 
import pandas as pd
import spacy
import os
import re

relative_path = os.getcwd()
nlp = spacy.load("en_core_web_md")

def p_input(user_input):
    print(f"==="*10)
    print(f"PREPROCESSING NL INPUT")
    print(f"==="*10)
    print(f"Extracting user_input information:\n    >{user_input}")
    # REGEX extracción de categoria texto
    text_query = re.findall(r"'([^']*)'", user_input)
    user_input = re.sub(r"'[^']*'", '', user_input)

    doc = nlp(user_input)

    # Extracción de nouns 
    nouns = []

    print("=== Tokens ===")
    for token in doc:
        print(f"Token: {token.text:4} - POS: {token.pos_:4} - Tag: {token.tag_:4} - Lemma: {token.lemma_:4} - Depth: {token.dep_:4} - Enttype: {token.ent_type_}")
        if token.pos_ == 'NOUN':
            nouns.append(token.text)

    tokens = [str(token) for token in doc]
    num_tokens = len(tokens)

    #Cardinalidad y lugares
    print("\n=== Named Entities ===")
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    for e, ent in enumerate(doc.ents):
        #if ent.label_ == 'GPE': 
        #    nouns.extend((ent.text).lower().split())
        if ent.label_ == 'CARDINAL':
            idx = tokens.index(ent.text) # Finds the index of cardinal token
            if idx+1 < num_tokens: # If not the last token
                print(tokens)
                print(nouns) 
                # get inmediate associated noun to the cardinal: two dark obscuire and whathell thunderstome
                for i in range(idx, len(token)):
                    if doc[i].pos_=='ADJ':
                        print(f"Adjetivo asociado: {doc[i]}")
                    if doc[i].pos_=='NOUN':
                        print(f"Inmediate noun: {doc[i]}")
                        number = w2n.word_to_num(ent.text)
                        print(f"> Aumenta {doc[i]} by {number}")
                        for j in range(int(number)-1):
                            nouns.append(str(doc[i]))
                        break
        
    # Useful for later
    #print("\n=== Noun Phrases ===")
    #for chunk in doc.noun_chunks:
    #    print(chunk.text)

    print(f"\nUncleaned processed input:    ")

    print(nouns)
    print(text_query)
    new_nouns = getClasses(nouns)
    print(new_nouns)
    key_map = {
        'entity': 'MoE_0',
        'object': 'MoE_1',
        'scene': 'MoE_2'
    }

    # Build renamed dictionary with counts
    tasks = {
        key_map[k]: dict(Counter(v))
        for k, v in new_nouns.items()
    }
    tasks['MoE_3'] = text_query

    print(f"\nFin del procesamiento. Resultados:")
    print(tasks)
    print("\n")
    return tasks

def getClasses(nouns):
    print(f"==="*10)
    print(f"MATCHING INFORMATION")
    print(f"==="*10)

    # Embedding space
    print(f"Modelos {list(gensim.downloader.info()['models'].keys())}: ")

    # so far -200 and -50 has worked fine fot this porpouses
    model = 'glove-twitter-200' #'glove-twitter-25' #'glove-wiki-gigaword-100' #'glove-twitter-50'
    print(f"\nSelecionado {model}", end="\n\n")
    
    try:    
        print(f"Cargando vectores:", end='')
        vectors = KeyedVectors.load(embeddig_path + model + '.bin')
        print(" Done")
    except:
        print("Descargandom vectores:", end='')
        #vectors = gensim.downloader.load(model)
        #vectors.save('./embeddings/' + model + '.bin')
        print("Done")

    print(os.getcwd())

    print(f"Entrenando KNN...")

    # Data preparation
    df = pd.read_csv(train_ex)
    x = [vectors[w] for w in df['word'] if w in vectors]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Training
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Tasks
    label_dict = {0: 'scene', 1: 'object', 2: 'entity'}
    new_nouns = {'scene': [], 'object': [], 'entity': []}

    for noun in nouns:
        if noun in vectors.key_to_index:
            prediction = model.predict([vectors[noun]])
            label = label_dict[prediction[0]]
            print(f"Resultados para palabra {noun}: {prediction} ({label})", end='')

            # Select the appropriate classlist of Vision Expert Model
            if prediction[0] == 0:
                known_classes = cvc_scene
            elif prediction[0] == 1:
                known_classes = cvc_objects
            elif prediction[0] == 2:
                known_classes = cvc_entity
            else:
                known_classes = []

            # Compute similarity
            # Filter the known classes to only include those in the vocabulary
            filtered_known_classes = [cls for cls in known_classes if cls in vectors.key_to_index]

            # Check if the input noun is also in the vocabulary
            if noun not in vectors.key_to_index:
                raise KeyError(f"Key '{noun}' not present in the embedding model.")
            
            # similarities
            similarities = [(cls, vectors.similarity(noun, cls)) for cls in filtered_known_classes]
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Most similar class to the new nouns
            if similarities:
                best_match = similarities[0][0]
                print(similarities[:4])
                new_nouns[label].append(best_match)
                print(f" -> {best_match}")
            else:
                print(" -> No similar class found.")
        else:
            print(f"{noun} not found in vector space!")
    return new_nouns

# COCO and ImageNet available classes
def preprocess_dataset(filepath):
    tokenized = []
    with open(filepath, "r") as f:
        lines = [s.strip() for s in f.readlines()]
        for line in lines:
            clean_line = line.split("/")[-1]
            clean_line = clean_line.lower()
            # Remove digits
            clean_line = re.sub(r'\d+', '', clean_line)
            # Replace underscores and multiple spaces 
            clean_line = re.sub(r'[_\s]+', ' ', clean_line) 
            words = clean_line.strip().split()
            tokenized.extend(words)
    return tokenized

# Places365 classes already clean
def preprocess_dataset_places(filepath):
    words = []
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            words.append(line)
    return words

# Define relative paths
availableClassesEntity = os.path.join(relative_path, "MoE_0", "moe0_classes.txt")
availableClassesObjects = os.path.join(relative_path, "MoE_1", "moe1_classes.txt")
availableClassesScenes = os.path.join(relative_path, "MoE_2", "all_places.txt")

# Load and tokenize classes
cvc_entity = preprocess_dataset(availableClassesEntity)
cvc_objects = preprocess_dataset(availableClassesObjects)
cvc_scene = preprocess_dataset_places(availableClassesScenes)

# Available computer vision model classes
#print("Entity tokens:", cvc_entity)
#print("Object tokens:", cvc_objects)
#print("Scene tokens:", cvc_scene)

embeddig_path = relative_path + "/nlp/embeddings/"
train_ex = relative_path + "/nlp/train_ex_vector/dataset.csv"