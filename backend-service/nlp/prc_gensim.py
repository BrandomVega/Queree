import gensim.downloader 
from gensim.models import KeyedVectors

def run_nlp_taks():
    # Gensim seems to have better semantics for our needs
    print(f"Iniciando {__name__}")

    # No es posible encontrar palabras que generalize para queries, toca usar ml
    known_classes = ['city', 'computer', 'dog']
    word = 'beach'

    # mode-<vector_size>
    print(f"Modelos {list(gensim.downloader.info()['models'].keys())}: ")
    model = 'glove-twitter-200'
    print(f"\nSelecionado {model}", end="\n\n")

    try:    
        print(f"Cargandom vectores:", end='')
        vectors = KeyedVectors.load('./nlp/embeddings/' + model + '.bin')
        print(" Done")
    except:
        print("Descargando vectores:", end='')
        vectors = gensim.downloader.load(model)
        vectors.save('./embeddings/' + model + '.bin')
        print(" Done")

    #print(vectors[word])  
    # Similares en todo el vocabulario.
    print(f"Palabra {word}")
    print(f"\nResultados\n")
    results = vectors.most_similar(word, topn=10)
    words = [w for (w,score) in results]
    #print(words)

    similarities = [(cls, vectors.similarity(word, cls)) for cls in known_classes]
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(similarities)
    return str(similarities)