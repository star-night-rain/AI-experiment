import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(transform=True):
    file_path = './penguin.csv'
    data = pd.read_csv(file_path)
    '''数据预处理'''
    # 为每个物种分配一个从0开始的数字
    data['Species'], species_mapping = pd.factorize(data['Species'])
    print('species mapping:')
    for index, species in enumerate(species_mapping):
        print(f'{index} -> {species}')

    # 为每个岛屿分配一个从0开始的数字
    data['Island'], island_mapping = pd.factorize(data['Island'])
    print('island mapping:')
    for index, island in enumerate(island_mapping):
        print(f'{index} -> {island}')

    # 为性别信息缺失的企鹅随机分配性别
    for index in data.index:
        if pd.isna(data.loc[index, 'Sex']):
            if random.random() < 0.5:
                sex = 'MALE'
            else:
                sex = 'FEMALE'
            data.loc[index, 'Sex'] = sex

    # 为性别分配数字
    data['Sex'], sex_mapping = pd.factorize(data['Sex'])
    print('sex mapping:')
    for index, sex in enumerate(sex_mapping):
        print(f'{index} -> {sex}')

    # 用组的平均值填充缺失值
    # mean_values = data.groupby(['Species', 'Island']).transform('mean')
    # data.fillna(mean_values, inplace=True)

    # 用-1补充缺失值
    data.fillna(-1, inplace=True)

    # 对不同的特征进行归一化(均值为0，方差为1)
    if transform:
        scaler = StandardScaler()
        features = [
            'Island', 'Culmen Length (mm)', 'Culmen Depth (mm)',
            'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 'Age'
        ]
        data[features] = scaler.fit_transform(data[features])

    x = data.drop(columns=['Species'])
    y = data['Species']
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=3407)
    return x_train, x_test, y_train, y_test


def get_distances(x1, x2, metric):
    distances = None
    # 欧氏距离
    if metric == 'euclidean':
        distances = np.sqrt(np.sum((x1 - x2)**2))
    # 曼哈顿距离
    elif metric == 'manhattan':
        distances = np.sum(np.abs(x1 - x2))
    # 切比雪夫距离
    elif metric == 'chebyshev':
        distances = np.max(np.abs(x1 - x2))
    # 闵可夫斯基距离
    elif metric == 'minkowski':
        return np.power(np.sum(np.abs(x1 - x2)**2), 1 / 2)
    return distances


def knn_prediction(x_train, y_train, x_test, k, metric):
    predictions = []
    for test in x_test.values:
        distances = [get_distances(x, test, metric) for x in x_train.values]
        knn_indices = np.argsort(distances)[:k]
        labels = y_train.iloc[knn_indices]
        # (label,count)
        pred = Counter(labels).most_common(1)[0][0]
        predictions.append(pred)
    return predictions


def KNN(x_train, x_test, y_train, y_test):
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    best_accuracy = 0
    best_k = None
    best_metric = None
    best_pred = None
    for k in range(1, 20, 2):
        for metric in metrics:
            y_pred = knn_prediction(x_train, y_train, x_test, k, metric)
            # 使用sklearn的KNN
            # knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            # knn.fit(x_train, y_train)
            # y_pred = knn.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_metric = metric
                best_pred = y_pred
            print(f'k={k},metric={metric},test accuracy:{accuracy*100:.2f}%')
    show_confusion_matrix_of_KNN(y_train, y_test, best_pred)
    print('-----------------KNN-----------------')
    print(
        f'best k:{best_k}, best metric:{best_metric}, best accuracy:{best_accuracy*100:.2f}%'
    )


def show_confusion_matrix_of_KNN(y_train, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['Adelie', 'Gentoo', 'Chinstrap'])
    disp.plot(cmap='Blues')  # 设置颜色映射为蓝色
    plt.title("Confusion Matrix for KNN")
    plt.show()


def show_confusion_matrix_of_tree(tree, y_test, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Adelie', 'Gentoo', 'Chinstrap'],
                yticklabels=['Adelie', 'Gentoo', 'Chinstrap'])
    plt.xlabel('Predicted Species')
    plt.ylabel('Actual Species')
    plt.title('Confusion Matrix for Decision Tree')
    plt.show()


def show_tree(tree, x_train):
    # 可视化决策树
    plt.figure(figsize=(20, 10))
    plot_tree(tree,
              feature_names=x_train.columns,
              class_names=['Adelie', 'Gentoo', 'Chinstrap'],
              filled=True,
              rounded=True,
              fontsize=7)
    plt.title('Decision Tree')
    plt.show()


def decision_tree(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=None,
                                  random_state=3407)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('-----------------Decision Tree-----------------')
    print(f'accuracy:{accuracy*100:.2f}%')
    show_confusion_matrix_of_tree(tree, y_test, y_pred)
    show_tree(tree, x_train)

def main():
    x_train, x_test, y_train, y_test = load_data(transform=True)
    decision_tree(x_train, x_test, y_train, y_test)
    KNN(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
