import numpy as np


class DiyPCA(object):
    def __init__(self,X,K):
        self.X = X #原始数据
        self.K = K #维 K
        self.central_X = [] #中心化矩阵
        self.C = [] #协方差矩阵
        self.tran_X = [] #转换矩阵
        self.lst_X = [] #最终降维后的矩阵

        self.central_X =self._toCentralized()
        self.C  = self._getCov()
        self.tran_X = self._getTranMatric()
        self.lst_X = self._getDst()

    def _toCentralized(self):
        mean = np.array([np.mean(value) for value in self.X.T]) #T 列转行了，每个样本的维度的信息都在一行
        self.central_X = self.X - mean
        return self.central_X

    def _getCov(self):
        srcTotal = np.shape(self.central_X)[0]#上一步已经把所有行转列了，这里行的维度 = 所有样本数
        self.C = np.dot(self.central_X.T,self.central_X)/srcTotal -1
        return self.C

    def _getTranMatric(self):
        a, b = np.linalg.eig(self.C)# CALL METHON
        index = np.argsort(-1 * a)#SORT
        tran_XT = [b[:,index[i]] for i in range(self.K)]
        # self.tran_X =tran_XT.T tran_XT不是numpy数组，不能直接T
        self.tran_X = np.transpose(tran_XT)
        return self.tran_X

    def _getDst(self):
        self.lst_X = np.dot(self.X,self.tran_X)
        print(self.lst_X)
        return self.lst_X
    pass


if __name__=='__main__':
    X = np.array([
                  [10, 15, 29,2],
                  [15, 46, 13,3],
                  [23, 21, 30,1],
                  [11, 9, 35,33],
                  [42, 45, 11,11],
                  [9, 48, 5,7],
                  [11, 21, 14,8],
                  [8, 5, 15,9],
                  [11, 12, 21,3],
                  [21, 20, 25,8]
    ])

    K = np.shape(X)[1]-2
    pca = DiyPCA(X,K)