import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


def getPlot(dataset_n, ti, idx_,
            ugedan_x_train, ugedan_y_train,
            sgedan_x_train, sgedan_y_train,
            ugedan_x_test, ugedan_y_test,
            sgedan_x_test, sgedan_y_test):

    if ti == "KPCA":
        mds1 = KernelPCA(n_components=2, random_state=0)
        mds2 = KernelPCA(n_components=2, random_state=0)
    if ti == "PCA":
        mds1 = PCA(n_components=2, random_state=0)
        mds2 = PCA(n_components=2, random_state=0)
    if ti == "TNSE":
        perp = 30
        mds1 = TSNE(n_components=2, init="random", random_state=42, perplexity=perp)
        mds2 = TSNE(n_components=2, init="random", random_state=42, perplexity=perp)

    ssize = 5

    plt.subplot(2, 2, 1)
    plt.title("Train set U-GEDAN")
    coordinates = mds1.fit_transform(ugedan_x_train)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=ugedan_y_train, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)
    plt.subplot(2, 2, 2)

    plt.title("Train set S-GEDAN")
    coordinates = mds2.fit_transform(sgedan_x_train)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=sgedan_y_train, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)
    plt.subplot(2, 2, 3)

    plt.title("Test set U-GEDAN")
    if ti == "TNSE":
        coordinates = mds1.fit_transform(ugedan_x_test)
    else:
        coordinates = mds1.transform(ugedan_x_test)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=ugedan_y_test, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)
    plt.subplot(2, 2, 4)

    plt.title("Test set S-GEDAN")
    if ti == "TNSE":
        coordinates = mds2.fit_transform(sgedan_x_test)
    else:
        coordinates = mds2.transform(sgedan_x_test)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=sgedan_y_test, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_n}/plot/{idx_}.png")
    plt.close()


def getParam(model):

    if model == RandomForestRegressor:
        return {
            'n_estimators': [100],
            'max_depth': [10, 20],
        }

    if model == KNeighborsRegressor:
        return {
            'n_neighbors': [3, 5, 8, 16],
            'weights': ['uniform', 'distance'],
        }

    if model == MLPRegressor:
        return {
            'hidden_layer_sizes': [(128, 64), (64, 32)],
            'solver': ['adam'],
            'alpha': [0.001, 0.01],
            'max_iter': [200]
        }

    if model == SVR:
        return {
            'C': [1, 10, 100],
            'kernel': ["rbf"],
            'degree': [1, 2],
        }

    if model == RandomForestClassifier:
        return {
            'n_estimators': [100],
            'max_depth': [10, 20],
        }

    if model == KNeighborsClassifier:
        return {
            'n_neighbors': [3, 5, 8, 16],
            'weights': ['uniform', 'distance'],
        }

    if model == MLPClassifier:
        return {
            'hidden_layer_sizes': [(64, 32)],
            'solver': ['adam'],
            'alpha': [0.001, 0.01],
            'max_iter': [200]
        }

    if model == SVC:
        return {
            'C': [1, 10, 100],
            'kernel': ["rbf"],
            'degree': [1, 2],
        }


def testModel(data_train, label_train, data_test, label_test, model_c, params, metric_, trasf=None):

    stand = PowerTransformer()

    if trasf != None:
        data_train = stand.fit_transform(data_train)
        data_test = stand.transform(data_test)

    model_c = GridSearchCV(model_c, param_grid=params, scoring="neg_root_mean_squared_error", cv=5)

    model_c.fit(data_train, label_train)

    pred_test = model_c.predict(data_test)

    return metric_(label_test, pred_test)


def getLoad(type, dataset_name, pivot):

    if type == "UGEDAN":
        train_x = np.load(f"results/{dataset_name}/encoding/dati_UGEDAN_train_train_{pivot}.npy")
        train_y = np.load(f"results/{dataset_name}/encoding/label_UGEDAN_train_train_{pivot}.npy")

        test_x = np.load(f"results/{dataset_name}/encoding/dati_UGEDAN_test_train_{pivot}.npy")
        test_y = np.load(f"results/{dataset_name}/encoding/label_UGEDAN_test_train_{pivot}.npy")

    else:
        train_x = np.load(f"results/{dataset_name}/encoding/dati_SGEDAN_train_train_{pivot}.npy")
        train_y = np.load(f"results/{dataset_name}/encoding/label_SGEDAN_train_train_{pivot}.npy")

        test_x = np.load(f"results/{dataset_name}/encoding/dati_SGEDAN_test_train_{pivot}.npy")
        test_y = np.load(f"results/{dataset_name}/encoding/label_SGEDAN_test_train_{pivot}.npy")

    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    return train_x, train_y, test_x, test_y



def getPredictionEmbedding(dataset_name):

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    classificazione = False

    if dataset_name == "FreeSolv":
        metrics = r2_score
        print("R2")

    if dataset_name == "BBBP":
        metrics = roc_auc_score
        print("ROC-AUC")
        classificazione = True

    if dataset_name == "ZINC":
        metrics = mean_absolute_error
        print("MAE")

    u_m1, u_m2, u_m3, u_m4 = [], [], [], []
    s_m1, s_m2, s_m3, s_m4 = [], [], [], []

    for p in range(3):

        u_train_x, u_train_y, u_test_x, u_test_y = getLoad("UGEDAN", dataset_name, p)
        s_train_x, s_train_y, s_test_x, s_test_y = getLoad("SGEDAN", dataset_name, p)

        print(f"Pivots n.{p}")
        if p == 0:
            print("Size", u_train_x.shape, s_train_x.shape)
        # PCA
        # KPCA
        # TNSE

        getPlot(dataset_name, "KPCA", dataset_name+str(p),
                u_train_x, u_train_y,
                s_train_x, s_train_y,
                u_test_x, u_test_y,
                s_test_x, s_test_y)

        stand = PowerTransformer()

        if classificazione:
            model_ = KNeighborsClassifier()
            param = getParam(KNeighborsClassifier)
        else:
            model_ = KNeighborsRegressor()
            param = getParam(KNeighborsRegressor)

        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)

        u_m1.append(p1)
        s_m1.append(p2)

        if classificazione:
            model_ = SVC()
            param = getParam(SVC)
        else:
            model_ = SVR()
            param = getParam(SVR)

        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)

        u_m2.append(p1)
        s_m2.append(p2)

        if classificazione:
            model_ = RandomForestClassifier(random_state=0)
            param = getParam(RandomForestClassifier)
        else:
            model_ = RandomForestRegressor(random_state=0)
            param = getParam(RandomForestRegressor)
        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)

        u_m3.append(p1)
        s_m3.append(p2)

        if classificazione:
            model_ = MLPClassifier(max_iter=1000, random_state=0)
            param = getParam(MLPClassifier)
        else:
            model_ = MLPRegressor(max_iter=1000, random_state=0)
            param = getParam(MLPRegressor)
        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)

        u_m4.append(p1)
        s_m4.append(p2)

        print(f"KNN\t UGEDAN {np.array(u_m1).mean():.3f}\t SGEDAN {np.array(s_m1).mean():.3f}")
        print(f"SVC\t UGEDAN {np.array(u_m2).mean():.3f}\t SGEDAN {np.array(s_m2).mean():.3f}")
        print(f"RF\t UGEDAN {np.array(u_m3).mean():.3f}\t SGEDAN {np.array(s_m3).mean():.3f}")
        print(f"MLP\t UGEDAN {np.array(u_m4).mean():.3f}\t SGEDAN {np.array(s_m4).mean():.3f}")
        print()

    print()

    print(f"KNN\t UGEDAN {np.array(u_m1).mean():.3f} ({np.array(u_m1).std():.3f})\t "
                f"SGEDAN {np.array(s_m1).mean():.3f} ({np.array(s_m1).std():.3f})")

    print(f"SVM\t UGEDAN {np.array(u_m2).mean():.3f} ({np.array(u_m2).std():.3f})\t "
                f"SGEDAN {np.array(s_m2).mean():.3f} ({np.array(s_m2).std():.3f})")

    print(f"RF\t UGEDAN {np.array(u_m3).mean():.3f} ({np.array(u_m3).std():.3f})\t "
                f"SGEDAN {np.array(s_m3).mean():.3f} ({np.array(s_m3).std():.3f})")

    print(f"MLP\t UGEDAN {np.array(u_m4).mean():.3f} ({np.array(u_m4).std():.3f})\t "
                f"SGEDAN {np.array(s_m4).mean():.3f} ({np.array(s_m4).std():.3f})")