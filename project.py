import xml.etree.ElementTree as ET
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import fasttext
import sys
import re

severity_names = {'Unclassified', 'Minor', 'Normal', 'Major', 'Critical', 'Blocker'}
severities = dict((j,i) for i,j in enumerate(severity_names))
category_names = {'Unassigned', 'Fachteam', 'zLinux', 'eIP-Development', 'ELA', 'Maintenance', 'Nachlass', 'QS', 'eIP-Operations', 'Clearing', 'eIP-QS', 'GVP', 'eIP-Magdeburg', 'eIP-Afo', 'Anforderung', 'Release-Test', 'Documentation', 'PM', 'eIP-agile@BF', 'eIP-PM', 'ZENVG', 'Sensitive', 'eIP Tiger Team', 'KM', 'eKP', 'Test', 'eIP-Test', 'eIP-Clearing', 'Testautomatisierung', 'eIP'}
categories = dict((j,i) for i,j in enumerate(category_names))
type_names = {'Defect', 'ETU-Fehler', 'Internal Defect', 'Patch', 'Test Action', 'Fremd-Software', 'LB Action', 'Action', 'Anfrage'}
types = dict((j,i+1) for i,j in enumerate(type_names))

N = 1000

def clean(text):
    return re.sub("[^a-zA-Z0-9 ]+", '', text)

def fasttext_model(filename="model.bin"):
    model = fasttext.load_model(filename)
    return (model, len(model['test']))

def countvectorizer_model():
    class CountVectorizerModel:
        def __init__(self, cv):
            self.cv = cv
        def get_sentence_vector(self, s):
            return vectorizer.transform([s]).toarray()[0]
    vectorizer = CountVectorizer()
    with open("raw_text.txt") as file:
        vectorizer.fit(file.readlines())
    return (CountVectorizerModel(vectorizer), len(vectorizer.get_feature_names_out()))

def parse(input_file, output_file, processing=None, vectorizer='fasttext'):
    data = ET.parse(input_file)
    root = data.getroot()
    wis = root.findall("workItem")[:N]

    model, vec_length = None, None
    if vectorizer == 'fasttext':
        model, vec_length = fasttext_model()
    else:
        model, vec_length = countvectorizer_model()
    M = 4 + vec_length*2
    
    with open(output_file, 'w') as file:
        i = 0
        print(f"{N} {M} {vec_length}", file=file)
        for wi in wis:
            vec = []
            summary_vec = model.get_sentence_vector(clean(wi.findtext("summary")))
            description_vec = model.get_sentence_vector(clean(wi.findtext("description")))
            priority = re.compile('Prio ([0-9]+)').match(wi.findtext("priority/name"))

            if priority is None:
                vec.append(0)
            else:
                vec.append(int(priority.group(1)))
            vec.append(severities[wi.findtext("severity/name")])
            vec.append(categories[wi.findtext("category/name")])
            vec.append(types[wi.findtext("type/name")])
            vec = np.append(vec, summary_vec)
            vec = np.append(vec, description_vec)
            if processing == 'normalization':
                vec = normalized(vec)
            elif processing == 'standardization':
                vec = standardize(vec)

            vec = np.append(vec, int(wi.findtext("timeSpent"))//1000)
            print(' '.join(map(str, vec)), file=file)
            i += 1

def normalized(vec):
    mx, mn = np.max(vec), np.min(vec)
    vec = (vec - mn)/(mx - mn)
    return vec

def standardize(vec):
    mean = np.mean(vec)
    n = len(vec)
    deviation = np.sqrt(np.sum(vec-mean)**2/n)
    vec = (vec - mean)/deviation
    return vec

def load_data(filename):
  f = open(filename)
  n, m, lv = map(int, next(f).strip().split())

  X = []
  y = []
  i = 0
  for l in f:
    its = l.strip().split()
    if float(its[-1]) > 10**9 or str(np.inf) in its:
        n -= 1
        continue
    y.append(float(its[-1]))
    X.append(list(map(float, its[:-1])))
    i += 1

  tmp = np.concatenate((np.array(X), np.array(y).reshape(n, 1)), axis=1)
  np.random.shuffle(tmp)
  X = tmp[:, :-1]
  y = tmp[:, -1]
  return n, m, X, y

def feature_selection(X, y, C=0.1, max_iter=1000, dual=False):
    lsvc = LinearSVC(C=C, penalty="l1", dual=dual, max_iter=max_iter).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    return model.transform(X)

def fit_LR(X, y):
    return linear_model.LinearRegression().fit(X, y)

def fit_LCV(X, y, max_iter=1000):
    return linear_model.LassoCV().fit(X, y)

if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # parse('export.xml', 'input-cv.txt', vectorizer='cv')
    # parse('export.xml', 'input-fasttext.txt', vectorizer='fasttext')
    # parse('export.xml', 'input-standardized.txt', vectorizer='fasttext', processing='standardization')
    # data = [[1.0, -0.4552485319306063], [0.6224264722836523, 0.14830238677543284], [0.8958013765892773, -1.7840230056101155*10**(22)], [0.10167340999431684, 0.016287068833275464]]

    # n, m, X, y = load_data('input-fasttext.txt')
    # splt = int(0.8*n)

    # print(data)
    # data = np.array(data).T

    # columns = ('LR', 'LR+LSVC', 'LassoCV')
    # rows = ['Test', 'Train']

    # values = np.arange(-1.1, 1.1, 0.2)

    # colors = ['green', 'red']
    # n_rows = len(data)

    # index = np.arange(len(columns)) + 0.3
    # bar_width = 0.4

    # cell_text = []
    # for row in range(n_rows):
    #     plt.bar(index, data[row], bar_width, color=colors[row])
    #     cell_text.append(['%1.4f' % (x) for x in data[row]])

    # colors = colors[::-1]
    # cell_text.reverse()

    # the_table = plt.table(cellText=cell_text,
    #                     rowLabels=rows,
    #                     rowColours=colors,
    #                     colLabels=columns,
    #                     loc='bottom')

    # plt.subplots_adjust(left=0.2, bottom=0.2)

    # plt.ylabel("R2 score")
    # plt.yticks(values, ['%1.1f' % val for val in values])
    # plt.xticks([])
    # plt.title('Train and test score by model')

    # plt.show()

    # history = []
    # for max_iter in np.arange(1000, 11000, 1000):
    #     X_sel = feature_selection(X, y, C=0.3, max_iter=max_iter)
    #     print(X_sel.shape)
    #     if X_sel.shape[1] < 10:
    #         history.append([max_iter, -1, -1])
    #         continue
    #     X_train, X_test = X_sel[:splt], X_sel[splt:]
    #     y_train, y_test = y[:splt], y[splt:]
    #     model = fit_LR(X_train, y_train)

    #     history.append([max_iter, model.score(X_train, y_train), model.score(X_test, y_test)])

    # history = np.array(history)

    # plt.plot(history[:, 0], history[:, 1], color='g')
    # plt.plot(history[:, 0], history[:, 2], color='r')
    # # plt.show()


    n, m, X, y = load_data('input-fasttext.txt')
    splt = int(0.8*n)
    X_train, X_test = X[:splt], X[splt:]
    y_train, y_test = y[:splt], y[splt:]
    # model = fit_LR(X_train, y_train)
    # print(f"{model.score(X_train, y_train)}, {model.score(X_test, y_test)}")

    model = fit_LCV(X_train, y_train)
    print(f"{model.score(X_train, y_train)}, {model.score(X_test, y_test)}")
    for _ in range(5):
        i = np.random.randint(n-splt)
        print(f'got: {model.predict(X_test[i].reshape(1,-1))[0]:.1f}, expected: {y_test[i]}')

    # n, m, X, y = load_data('input-cv.txt')
    # splt = int(0.8*n)
    # X_train, X_test = X[:splt], X[splt:]
    # y_train, y_test = y[:splt], y[splt:]
    # model = fit_LR(X_train, y_train)
    # print(f"{model.score(X_train, y_train)}, {model.score(X_test, y_test)}")

    # model = fit_LCV(X_train, y_train)
    # print(f"{model.score(X_train, y_train)}, {model.score(X_test, y_test)}")