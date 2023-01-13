import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
onehot_encoder = OneHotEncoder(sparse=False)

import torch
import torch.optim as optim


class LogisticRegression:
    def __init__(self):
        pass

    def fit(self, X, y, lr=0.4, momentum=0.1, niter=500):
        '''
        Train a multiclass logistic regression model on the given training set.

        Parameters
        ----------
        X: training examples, represented as an input array of shape (n_sample,
           n_features).
        y: labels of training examples, represented as an array of shape
           (n_sample,) containing the classes for the input examples
        lr: learning rate for gradient descent
        niter: number of gradient descent updates
        momentum: the momentum constant (see assignment task sheet for an explanation)

        Returns
        -------
        self: fitted model
        '''
        self.classes_ = np.unique(y)
        self.class2int = dict((c, i) for i, c in enumerate(self.classes_))
        y = np.array([self.class2int[c] for c in y])

        n = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.intercept_ = np.zeros(n_classes)
        self.coef_ = np.zeros((n_classes, n_features))

        # Implement your gradient descent training code here; uncomment the code below to do "random training"
        self.intercept_ = np.random.randn(*self.intercept_.shape)
        self.coef_ = np.random.randn(*self.coef_.shape)

        w = torch.from_numpy(self.coef_)
        b = torch.from_numpy(self.intercept_)
        w.requires_grad = True
        b.requires_grad = True
        self.w = w
        self.b = b

        X = torch.from_numpy(X)
        Y_onehot = onehot_encoder.fit_transform(y.reshape(-1, 1))
        Y_onehot = torch.from_numpy(Y_onehot)
        
        # subquestion (f)
        optimizer = optim.SGD([w, b], lr=lr, momentum=momentum)
        for i in range(niter):
            scores = X @ w.T + b
            score = scores - torch.max(scores, 1)[0].reshape(-1, 1)

            exp = score.exp()
            sum_exp = exp.sum(dim=1, keepdim=True)
            softmax = exp / sum_exp

            loss = -(1 / n) * torch.sum(torch.log(softmax + 1e-5) * Y_onehot)

            if w.grad is not None:
                w.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()

            # subquestion (f)
            optimizer.zero_grad()

            loss.backward()

            w.data.add_(-lr * w.grad.data)
            b.data.add_(-lr * b.grad.data)

            # subquestion (f)
            optimizer.step()
            print(loss)
        
        return self

    def predict_proba(self, X):
        '''
        Predict the class distributions for given input examples.

        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).

        Returns
        -------
        y: predicted class distributions, represented as an array of shape (n_sample,
           n_classes)
        '''

        X = torch.from_numpy(X)
        outputs = X @ self.w.T + self.b
        output = outputs - torch.max(outputs, 1)[0].reshape(-1, 1)
        exp = output.exp()
        sum_exp = exp.sum(dim=1, keepdim=True)
        softmax = exp / sum_exp
        return softmax

    def predict(self, X):
        '''
        Predict the classes for given input examples.

        Parameters
        ----------
        X: input examples, represented as an input array of shape (n_sample,
           n_features).

        Returns
        -------
        y: predicted class labels, represented as an array of shape (n_sample,)
        '''

        probs = self.predict_proba(X)
        labels = []
        for i in range(len(probs)):
            x = torch.argmax(probs[i]).item() + 1
            labels.append(x)

        return labels




if __name__ == '__main__':
    X, y = fetch_covtype(return_X_y=True)

    # normalize
    X = StandardScaler().fit_transform(X)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression()
    clf.fit(X_tr, y_tr)
    print(accuracy_score(y_tr, clf.predict(X_tr)))
    print(accuracy_score(y_ts, clf.predict(X_ts)))