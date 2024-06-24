path_undamaged = ['/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_udam1/udam1/',
        '/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_udam2/udam2/',
        '/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_udam3/udam3/',
        '/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_udam4/udam4/',
        '/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_udam5/udam5/',
        ]
path_damaged = ['/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L1C_A/Dam1/',
        '/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L1C_B/Dam4/',
        '/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L3A/Dam2/',
        '/content/drive/MyDrive/Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L13/Dam3/',
        ]       # path where the data files saved

BATCH_SIZE = 150
rep_dim = 32
lr_pretrain = 0.001
weight_decay_pretrain = 1e-6
epochs_pretrain = 500
lr = 0.001
weight_decay = 1e-6
epochs = 100
num_class = 3
eps = 0.001
nu = 0.1