import json
from argparse import ArgumentParser
from nltk.tokenize import sexpr_tokenize
import nltk
import math
import pprint

import sys
sys.path.insert(0, '../search_engine/')

import repository as se

#import search_engine.repository as se


def busca_and(index, query):
    query_terms = query.strip().split()
    if len(query_terms) == 0:
        return {}

    initial_term = query_terms[0]
    docids = set(index[initial_term]) if initial_term in index else set()
    for word in query_terms[1:]:
        result = set(index[word]) if word in index else set()
        docids &= result

    return docids


def busca_docids(index, query):
    result = [q.strip().strip('()') for q in sexpr_tokenize(query)]

    docids = set()
    for subquery in result:
        res = busca_and(index, subquery)
        docids |= res

    return docids

def busca(corpus, repo, index, query):
    # Parsing da query.
    # Recuperar os ids de documento que contem todos os termos da query.
    docids = busca_docids(index, query)

    # Retornar os textos destes documentos.
    return docids

# Implementacao simples do algoritmo levenshein, de distancia
# entre vetores de strings. Autor:  - https://stackoverflow.com/questions/47728069/sklearn-cosine-similarity-for-strings-python
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]/float(len(s1))

def ranking(corpus, repo, index, docids, query): #   

    query = query.replace("(", "")
    query = query.replace(")", "")
    query = query.split()
    print("q: ", query)
    debester = []
    points = 0
    for doc_id in range(len(list(docids))):
        fdisk = nltk.FreqDist(repo[list(docids)[doc_id]])
        for palavra in query:
            if fdisk[palavra] != 0:
                points += ( 1 + math.log2(fdisk[palavra])) * math.log2( len(corpus) / len(index[palavra]) )
 
        debester.append((list(docids)[doc_id], points))

    debester = sorted(debester, key=lambda x: x[1], reverse=True)

    return [x[0] for x in debester]

def main():
    parser = ArgumentParser()
    parser.add_argument('corpus', help='Arquivo do corpus')
    parser.add_argument('repo', help='Arquivo do repo.')
    parser.add_argument('index', help='Arquivo do index.')
    parser.add_argument('num_docs',
                        help='Numero maximo de documentos a retornar',
                        type=int)
    parser.add_argument('query', help='A query (entre aspas)')
    args = parser.parse_args()

    corpus = se.load_corpus(args.corpus)

    with open(args.repo, 'r') as file:
        repo = json.load(file)

    with open(args.index, 'r') as file:
        index = json.load(file)

    docids = busca(corpus, repo, index, args.query)
    docids_ranqueados = ranking(corpus, repo, index, docids, args.query)
    docs = [corpus[docid] for docid in docids_ranqueados[:args.num_docs]]
    
    for doc in docs:
        print(doc)
    print(f'Numero de resultados: {len(docids)}')

if __name__ == '__main__':
    main()
