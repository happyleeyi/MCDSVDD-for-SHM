import numpy as np #선형대수 관련 함수 이용 가능 모듈
import matplotlib.pyplot as plt#시각화 모듈
import torch #파이토치 기본모듈
import torch.nn as nn #신경망 모델 설계 시 필요한 함수
import torch.nn.functional as F # 자주 이용되는 함수'F'로 설정
from torchvision import transforms, datasets #torchvision모듈 내 transforms, datasets함수 임포트
from Dataset import get_data
from Variables import path_damaged, path_undamaged, BATCH_SIZE, rep_dims, lr_pretrain, epochs_pretrain, weight_decay_pretrain, lr, epochs, weight_decay, num_class, eps, nu, rep_dims, data_saved, trained, pretrained, use_kde, bandwidth, lpf
from Trainer import train_model
from Tester import test_model

if torch.cuda.is_available():
    device = torch.device('cuda') #GPU이용

else:
    device = torch.device('cpu') #GPU이용안되면 CPU이용

print('Using PyTorch version:', torch.__version__, ' Device:', device)

Data = get_data(path_undamaged, path_damaged, lpf)
train_loader, test_loader = Data.load_data(BATCH_SIZE, data_saved)

#B = bandwidth
for rep_dim in rep_dims:
    SVDD_trainer = train_model(lr_pretrain, weight_decay_pretrain, epochs_pretrain, lr, weight_decay, epochs, device, train_loader, rep_dim, num_class, eps, nu, trained, pretrained)
    net, R, c = SVDD_trainer.train()

    SVDD_tester = test_model(net, R, c, train_loader, test_loader, device)
    SVDD_tester.confusion_mat(rep_dim, nu, bandwidth, use_kde)
