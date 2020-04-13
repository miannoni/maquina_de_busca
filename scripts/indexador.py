from argparse import ArgumentParser

import sys
sys.path.insert(0, '../search_engine/')

import repository as se


def main():
    parser = ArgumentParser()
    parser.add_argument('corpus',
                        help='Arquivo json com um dicionario docid para texto')
    parser.add_argument('repo_name',
                        help='Raiz do nome do arquivo de repositorio')
    args = parser.parse_args()

    corpus = se.load_corpus(args.corpus)
    repo = se.create_repo(corpus)
    index = se.create_index(repo)

    se.save_repo(args.repo_name + '_repo.json', repo)
    se.save_index(args.repo_name + '_index.json', index)


if __name__ == '__main__':
    main()
