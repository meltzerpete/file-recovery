import os

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smart_open
from sklearn.cluster import KMeans

# Set file names
lee_train_file = 'corpus.txt'
model_file = 'doc2vec-window80-min-max.model'

file_list = []


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if not str(line).startswith('/home/pete/raw_corpus'):
                continue
            file_name = str(line).split(" ")[0]
            file_list.append(file_name)
            line = line[len(file_name):]
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line, min_len=1, max_len=50),
                                                           [i])


corpus = list(read_corpus(lee_train_file))

if os.path.exists(model_file):
    print(f'Loading Model: {model_file}')
    model = gensim.models.Doc2Vec.load(model_file)
else:
    print('Building Model')
    model = gensim.models.doc2vec.Doc2Vec(vector_size=2, window=80, min_count=2, epochs=40)

    print('Building Vocab')
    model.build_vocab(corpus)

    print('Training Model')
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    print(f'Saving model: {model_file}')
    model.save(model_file)

print('Inferring Vectors')
vecs = np.array([model.infer_vector(line.words) for line in corpus])

df = pd.DataFrame(vecs)
df['file'] = file_list

print('Plotting')
plt.plot(df[0], df[1], '.')
plt.savefig('plot.pdf')
plt.show()


def get_middle(group):
    group = group.copy()
    group['single'] = group[0] + group[1]
    return group.sort_values('single').iloc[int(len(group) / 2)]


def get_others(group):
    group = group.copy()
    group['single'] = group[0] + group[1]
    mid_id = int(len(group) / 2)
    others = []
    if mid_id < len(group) - 1:
        others.append(group.sort_values('single').iloc[mid_id + 1])
    if mid_id > 0:
        others.append(group.sort_values('single').iloc[mid_id - 1])
    return others


def print_file(file, n=3):
    try:
        with open(file) as f:
            lines = f.readlines()
            for i in range(min(n, len(lines))):
                print(lines[i], end="")
    except Exception:
        print("file exception.")
    print()


def show_plot(df, label, title=None, print_examples=True):
    if title is None:
        title = label
    plt.title(title)

    grouped = df.groupby(label)
    for key, group in grouped:
        plt.plot(group[0], group[1], '.')

        median = get_middle(group)
        plt.text(median[0], median[1], str(key), fontdict={'size': 14})

        if print_examples:
            print(f'{label} {key}: --median--')
            print_file(median.file, n=5)

            for other in get_others(group):
                print(f'{label} {key}:')
                print_file(other.file, n=5)

    plt.savefig('label')
    plt.show()


class_range = range(1, 15)
scores = []
for num_classes in class_range:
    class_string = 'K-Means ' + str(num_classes) + '-classes'
    kmeans = KMeans(num_classes, max_iter=1000)
    score = kmeans.fit(vecs).inertia_
    scores.append(score)
    df[class_string] = kmeans.predict(vecs)
    show_plot(df, class_string)

# k-means score
plt.xlabel("No. of Clusters")
plt.ylabel("Inertia")
plt.plot(class_range, scores, 'x-')
plt.show()
