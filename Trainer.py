import time
import torch.optim as optim
import torch #파이토치 기본모듈
import numpy as np
from MultiClassDeepSVDD import Autoencoder,DeepSVDD

class train_model:
    def __init__(self, lr_pretrain, weight_decay_pretrain, epochs_pretrain, lr, weight_decay, epochs, device, train_loader, rep_dim, num_class, eps, nu):
        self.lr_pretrain = lr_pretrain
        self.weight_decay_pretrain = weight_decay_pretrain
        self.epochs_pretrain = epochs_pretrain
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.rep_dim = rep_dim
        self.num_class = num_class
        self.eps = eps
        self.nu = nu
        self.check_pretrained = False

    def pretrain(self):
        ae_net = Autoencoder(self.rep_dim).to(self.device)
        #ae_net.load_state_dict(torch.load('aefloormodel_state_dict.pt'))

        lr_milestones = tuple()
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr_pretrain, weight_decay=self.weight_decay_pretrain,
                                    amsgrad='adam'=='amsgrad')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

        print('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.epochs_pretrain):

            scheduler.step()
            if epoch in lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for X, Y in self.train_loader:
                X = X.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(X)
                scores = torch.sum((outputs - X) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.epochs_pretrain, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        print('Pretraining time: %.3f' % pretrain_time)
        print('Finished pretraining.')

        self.save_weight_to_model(ae_net)
        self.check_pretrained = True

    def save_weight_to_model(self, ae_net):

        ae_net.eval()
        net = DeepSVDD(self.rep_dim).to(self.device)
        net_dict = net.state_dict()
        ae_net_dict = ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        net.load_state_dict(net_dict)

        torch.save(net.state_dict(), 'floormodel_state_dict.pt')
        torch.save(ae_net.state_dict(), 'aefloormodel_state_dict.pt')


    def set_c(self, net):
        c = torch.zeros((self.num_class, net.rep_dim), device=self.device)
        n_samples = torch.zeros(self.num_class, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in self.train_loader:
                # get the inputs of the batch
                inputs, y = data
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.type(torch.IntTensor)
                outputs = net(inputs)
                for i in range(outputs.shape[0]):
                    n_samples[y[i]-1]+=1
                    c[y[i]-1]+=outputs[i]
        for i in range(self.num_class):
            c[i] = torch.div(c[i],n_samples[i])

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps
        return c

    def train(self):
        self.pretrain()
        net = DeepSVDD(self.rep_dim).to(self.device)
        net.load_state_dict(torch.load('floormodel_state_dict.pt'))

        c = self.set_c(net)
        R = torch.tensor([0.0,0.0,0.0], device=self.device)
        lr_milestones = tuple()
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                    amsgrad='adam'=='amsgrad')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
        print('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.epochs):

            scheduler.step()
            if epoch in lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            dist = [[],[],[]]

            epoch_start_time = time.time()
            for data in self.train_loader:
                inputs, y = data
                inputs = inputs.to(self.device)
                y = y.to(self.device)
                y = y.type(torch.IntTensor)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                loss = torch.zeros(3)

                c_ = c[y-1]

                loss = torch.sum((outputs-c_)**2, dim=1)

                loss = torch.mean(loss)

                for i in range(outputs.shape[0]):
                    dist[y[i]-1].append(torch.sum((outputs[i] - c[y[i]-1]) ** 2))

                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
            for i in range(self.num_class):
                d = torch.tensor(dist[i])
                R.data[i] = torch.tensor(np.quantile(np.sqrt(d), 1 - self.nu), device=self.device)

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.epochs, epoch_train_time, loss_epoch / n_batches))

        train_time = time.time() - start_time
        print('Training time: %.3f' % train_time)

        print('Finished training.')

        self.net = net
        self.R = R
        self.c = c

        torch.save(net.state_dict(), 'floormodel_state_dict.pt')
        np.save('R.npy', self.R.clone().data.cpu().numpy())
        np.save('c.npy',self.c.clone().data.cpu().numpy())

        return self.net, self.R, self.c



