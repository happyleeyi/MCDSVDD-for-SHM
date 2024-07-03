import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class test_model:
    def __init__(self, net, R, c, test_loader, device):
        self.net = net
        self.R = R
        self.c = c
        self.test_loader = test_loader
        self.device = device

    def test(self):
        predict_test = []
        truevalue_test = []
        score = []
        score = np.array(score)

        self.net.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                check = [[] for i in range(x.shape[0])]

                for i in range(3):
                    outputs = self.net(x[:,:,:,(2-i)*8:(3-i)*8])
                    for j in range(outputs.shape[0]):
                        dist = torch.sum((self.c[i]-outputs[j]) ** 2)
                        err_ = np.sqrt(dist.clone().data.cpu().numpy())
                        check[j].append(err_)
                for i in range(y.shape[0]):
                    truevalue_test.append(y.clone().data.cpu().numpy().tolist()[i])
                check = np.array(check)

                if score.shape[0]==0:
                    score = check
                else:
                    score = np.append(score, check, axis = 0)

        score_idx = [score[i]>self.R.clone().data.cpu().numpy() for i in range(score.shape[0])]
        score_idx = np.array(score_idx)
        print(score_idx)
        for i in range(score_idx.shape[0]):
            if sum(score_idx[i]) == 0:
                predict_test.append(1)
            else:
                score[i][score_idx[i] == False] = 0
                predict_test.append(score[i].argmax()+2)


        predict_test = np.array(predict_test)
        truevalue_test = np.array(truevalue_test)
        return predict_test, truevalue_test

    def confusion_mat(self, rep_dim):
        predict_test, truevalue_test = self.test()
        class_names = ['normal', '1f dam', '2f dam', '3f dam']
        matrix1 = confusion_matrix(truevalue_test, predict_test)

        dataframe1 = pd.DataFrame(matrix1, index=class_names, columns=class_names)

        plt.figure(figsize=(6,6))
        sns.heatmap(dataframe1, annot=True, cbar=None, cmap="Blues")
        plt.title("Confusion Matrix_test"), plt.tight_layout()
        plt.ylabel("True Class"), plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig('confusion matrix with MCDSVDD'+'(repdim'+str(rep_dim)+')'+'.png')
        plt.show()

        print(predict_test==truevalue_test)
        accuracy = np.unique(predict_test==truevalue_test, axis=0, return_counts=True)[1][1]/len(predict_test)
        print("accuracy : ", accuracy)

        f = open("record.txt", "a+")
        f.write("repdim "+str(rep_dim)+" accuracy : %f\n" %accuracy)
        f.close()


