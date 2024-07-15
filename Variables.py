path_undamaged = ['Bookshelf Frame Structure - DSS 2000/Book_2000_udam1/udam1/',
        'Bookshelf Frame Structure - DSS 2000/Book_2000_udam2/udam2/',
        'Bookshelf Frame Structure - DSS 2000/Book_2000_udam3/udam3/',
        'Bookshelf Frame Structure - DSS 2000/Book_2000_udam4/udam4/',
        'Bookshelf Frame Structure - DSS 2000/Book_2000_udam5/udam5/',
        ]
path_damaged = ['Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L1C_A/Dam1/',
        'Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L1C_B/Dam4/',
        'Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L3A/Dam2/',
        'Bookshelf Frame Structure - DSS 2000/Book_2000_Dam_L13/Dam3/',
        ]

BATCH_SIZE = 150
rep_dims = [4,8,16,24,32,40,48,56,64]
lr_pretrain = 0.001
weight_decay_pretrain = 1e-6
epochs_pretrain = 500
lr = 0.001
weight_decay = 1e-6
epochs = 200
num_class = 3
eps = 0.001
nu = [0.3,0.25,0.2,0.15,0.1,0.05,0]
bandwidth = [0.5,0.75,1,1.25,1.5,1.75]
data_saved = True
trained = True
pretrained = True
use_kde = True
lpf = False