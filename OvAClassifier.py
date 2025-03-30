import numpy as np


class OvAClassifier:
    def __init__(self, feature_num: float, class_num: float, type: float):
        # type - 0 -> Linear, 1 -> SWM, 2 -> Logistic
        # define width with bias
        self.__weights: np.ndarray = \
            np.random.normal(0, 1, (feature_num + 1, class_num))
        self.__type: float = type
        self.__class_num: float = class_num

    # cals y_est -> <w,x>
    def __forward(self, x: np.ndarray, class_ind: float) -> np.ndarray:
        assert x.shape[0] == self.__weights[:, class_ind].shape[0], \
            "Invalid number of parameters. Should be " + self.weights.size - 1
        return self.__weights[:, class_ind] @ x

    # train (find weights)
    def train(self, x: np.ndarray, y_reference: np.ndarray, iterations: float,
              ll: float = 1e-3):
        # x - features
        # y_reference - targets
        # iterations - number of iteration for trains

        # pad x (for use width with bias)
        x_padded = np.concat([np.ones((1, x.shape[1])), x], 0)

        for i in range(iterations):
            # train 4 classifiers
            for ind_class in range(self.__class_num):
                # -1 - other classes, 1 - est class (for current classifier)
                y_class = 2 * (y_reference == ind_class) - 1
                # define M (<w,x>)
                y_estimate = self.__forward(x_padded, ind_class)
                # calc gradient depending on type
                gradient = self.__calc_gradient(ll, y_class, y_estimate,
                                                x_padded, ind_class)
                # define new weigths
                self.__weights[:, ind_class] -= gradient

    # calc gradient
    def __calc_gradient(self, ll: float, y_ref: np.ndarray, y_est: np.ndarray,
                        x: np.ndarray, class_ind: float) -> np.ndarray:
        # linear
        if (self.__type == 0):
            valid = y_est * y_ref <= 0  # y<w,x> <= 0
            gradient = 2 * ll * self.weights[:, class_ind] + \
                np.sum(-y_ref[valid] * x[:, valid], 1)
        # svm
        elif (self.__type == 1):
            valid = 1 - y_est * y_ref > 0
            gradient = 2 * ll * self.weights[:, class_ind] + \
                np.sum(-y_ref[valid] * x[:, valid], 1)
        # logistic
        else:
            gr = self.__calc_sigmoid(-y_ref*y_est) * (-y_ref) * x
            gradient = np.sum(gr, 1)
        return gradient

    # sigmoid for LogisticRegression
    def __calc_sigmoid(self, z):
        z = np.clip(z, -30, 30)
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    # define class for each object
    def predict(self, x, n_classes):
        y_est = np.zeros((x.shape[1], n_classes))
        for i in range(n_classes):
            y_est[:, i] = self.__forward(x, i)
        return np.argmax(y_est, axis=1)

    # getter for weights
    @property
    def weights(self) -> np.ndarray:
        return self.__weights
