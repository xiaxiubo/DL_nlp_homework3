import numpy as np
import jieba
from gensim import corpora, models
import os
from sklearn.naive_bayes import MultinomialNB


# 统计所有的标点符号和英文字符
punctuation = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~！？｡。＂" \
           "＃＄％＆＇()＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝‘’" \
           "～｟｠｢｣､、〃《》「」『』【】〔〕（）〖〗〘〙〚〛〜〝〞“”〟〰\n\u3000" \
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
              "说道一个自己咱们什么不是他们一声心想心中知道只见还是却是甚么突然 "


def remove_punctuation(something):  # 移除列表中的标点符号
    new_l = []
    for s in something:
        if s not in punctuation:
            new_l.append(s)
    return new_l


def book_read():  # 将所有的book读取，使用jieba进行分词，再去除其中的标点
    if os.path.exists('books.npy'):
        a = np.load('books.npy', allow_pickle=True)
        books = a.tolist()
    else:
        books_name = open('source/inf.txt').read()
        book_list = books_name.split(",")

        books = []
        i = 0

        for book in book_list:
            i += 1
            f = open('source/'+book+'.txt', encoding="UTF-8")
            txt = f.read()
            # seg_list = jieba.lcut(txt, cut_all=False)
            seg_list = [w for w in jieba.cut(txt) if len(w) > 1]
            seg = remove_punctuation(seg_list)
            books.append(seg)
            print('读取进度{}/16'.format(i))
        m = np.array(books, dtype=object)
        np.save('books.npy', m)
    return books


if __name__ == '__main__':
    # 超参数
    para_lenth = 500
    para_num_each = 40
    para_num_for_test = 3
    topic_num = 150

    books = book_read()  # 读取所有书

    book_num = len(books)
    # 预处理数据
    train_test_data = []
    train_test_label = []
    for i in range(len(books)):
        for j in range(para_num_each):
            start = np.random.randint(0, len(books[i])-para_lenth-100)
            train_test_data.append(books[i][start:start+para_lenth])
            train_test_label.append(i)

    for i in range(len(books)):
        for j in range(para_num_for_test):
            start = np.random.randint(0, len(books[i])-para_lenth-100)
            train_test_data.append(books[i][start:start+para_lenth])
            train_test_label.append(i)

    train_data = train_test_data[:para_num_each * book_num]
    train_label = train_test_label[:para_num_each * book_num]
    test_data = train_test_data[para_num_each * book_num:]
    test_label = train_test_label[para_num_each * book_num:]

    # 利用lda包训练lda模型
    dictionary = corpora.Dictionary(train_test_data)
    lda_bow_data = [dictionary.doc2bow(x) for x in train_data]
    lda = models.LdaModel(corpus=lda_bow_data, id2word=dictionary, num_topics=topic_num)

    train_topic_distribution = lda.get_document_topics(lda_bow_data)
    train_topic_matrix = np.zeros((len(train_data), topic_num))
    for i in range(len(train_topic_distribution)):
        for j in train_topic_distribution[i]:
            train_topic_matrix[i][j[0]] = j[1]
    print(train_topic_matrix)

    print('模型训练结果：')
    print(lda.print_topics(num_topics=topic_num, num_words=10))
    lda.save('lda.model')

    # 对测试集进行lda主题分布求解
    lda_test_bow_data = [dictionary.doc2bow(x) for x in test_data]
    test_topic_distribution = lda.get_document_topics(lda_test_bow_data)
    test_topic_matrix = np.zeros((len(test_data), topic_num))
    for i in range(len(test_topic_distribution)):
        for j in test_topic_distribution[i]:
            test_topic_matrix[i][j[0]] = j[1]

    # 使用贝叶斯分类器对测试文本进行分类
    nav = MultinomialNB()
    nav.fit(train_topic_matrix, train_label)
    y_predict = nav.predict(test_topic_matrix)

    perfect = 0
    for i in range(len(y_predict)):
        if y_predict[i] == test_label[i]:
            perfect += 1
    print("预测的结果为: ", y_predict)
    print("实际的结果为: ", test_label)
    print("试验的准确率为：", perfect/len(y_predict))


