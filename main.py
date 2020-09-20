import numpy as np
import scipy.io as scio
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import skimage.io


def load_data(src_fn):
    data = scio.loadmat(src_fn)
    for key in data.keys():
        if key[:2] != '__':
            return data[key]
    return None


def data_split(dataset, label):
    line, colm = label.shape
    x = []
    y = []
    for i in range(line):
        for j in range(colm):
            if label[i, j] != 0:
                x.append(np.ndarray.flatten(dataset[i,j,:]))
                y.append(label[i,j])
    return np.array(x), np.array(y)


def model_apply(model, dataset):
    line, colm, bands = dataset.shape
    x = []
    for i in range(bands):
        x.append(np.ndarray.flatten(dataset[:,:,i]))
    x = np.array(x).T
    predict = model.predict(x)
    return np.reshape(predict, (line,colm))


if __name__ == '__main__':
    dataset = load_data('./dataset/Indian_pines_corrected.mat')
    label = load_data('./dataset/Indian_pines_gt.mat')
    dst_fn = 'res_RF.tif'
    train_percent = 0.5
    model = 'RF'
    if dataset is None or label is None:
        raise Exception('failed to import data!')
    x, y = data_split(dataset, label)
    train_data, test_data, train_label, test_label = train_test_split(
        x, y, random_state=1, train_size=train_percent, test_size=1-train_percent)
    # model train
    if model == 'SVM':
        # SVM
        classifier = svm.SVC(C=2, kernel='linear', gamma=10, decision_function_shape='ovr')
    elif model == 'RF':
        # RF
        classifier = RandomForestClassifier()
    classifier.fit(train_data, train_label)
    print("训练集：%.4f" % classifier.score(train_data, train_label))
    print("测试集：%.4f" % classifier.score(test_data, test_label))
    # apply
    predict = model_apply(classifier, dataset)
    skimage.io.imsave(dst_fn, predict.astype(np.uint8))
