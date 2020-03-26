import json
from argparse import ArgumentParser
from nltk.tokenize import sexpr_tokenize

import search_engine.repository as se


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


def ranking(corpus, repo, index, docids):
    # Ranquear os documentos.

    return list(docids)  # dummy por enquanto.

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
    docids_ranqueados = ranking(corpus, repo, index, docids)
    docs = [corpus[docid] for docid in docids_ranqueados[:args.num_docs]]

    print(docs)

    print(f'Numero de resultados: {len(docids)}')

if __name__ == '__main__':
    main()
