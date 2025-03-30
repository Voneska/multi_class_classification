import numpy as np


class AvAClassifier:
    def __init__(self, feature_num: int, class_num: float, type: float):
        # type - 0 -> Linear, 1 -> SWM, 2 -> Logistic
        self.__class_num: float = class_num

        n_classifiers: int = int(class_num * (class_num - 1) / 2)

        # define width with bias
        self.__weights: np.ndarray = \
            np.random.normal(0, 1, (feature_num + 1, n_classifiers))

        self.__type: float = type

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
            w_ind = 0
            # train 4 classifiers
            for class1_ind in range(self.__class_num):
                for class2_ind in range(class1_ind + 1, self.__class_num):
                    # find indexes for needed classes
                    valid_classes = (y_reference == class1_ind) | \
                        (y_reference == class2_ind)

                    # take data for needed classes
                    x_taken = x_padded[:, valid_classes]
                    y_taken = y_reference[valid_classes]

                    # 1 - first classes, -1 - second class
                    y_class = 2 * (y_taken == class1_ind) - 1
                    # define M (<w,x>)
                    y_estimate = self.__forward(x_taken, w_ind)
                    # calc gradient depending on type
                    gradient = self.__calc_gradient(ll, y_class, y_estimate,
                                                    x_taken, w_ind)
                    # define new weigths
                    self.__weights[:, w_ind] -= gradient
                    w_ind = w_ind + 1

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
        z = np.clip(z, -50, 50)
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    # define class for each object
    def predict(self, x, n_classes):
        y_est = np.zeros((x.shape[1], self.__weights.shape[1]))
        w_ind = 0
        for class1_ind in range(self.__class_num):
            for class2_ind in range(class1_ind + 1, self.__class_num):
                y_est[:, w_ind] = np.where(self.__forward(x, w_ind) > 0,
                                           class1_ind, class2_ind)

                w_ind = w_ind + 1
        y_pred = np.apply_along_axis(self.__find_class, axis=1, arr=y_est)

        return y_pred

    def __find_class(self, y):
        unique_elements, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        most_frequent = unique_elements[counts == max_count]
        return most_frequent[0] if len(most_frequent) == 1 else None

    # getter for weights
    @property
    def weights(self) -> np.ndarray:
        return self.__weights
