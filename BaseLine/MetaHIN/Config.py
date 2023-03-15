


config_ml = {
    'dataset': 'movielens-1m',
    'mp': ['um'],
    'use_cuda': False,
    'file_num': 6,  # each task contains 12 files for movielens
    'first_embedding_dim': 32,
    'second_embedding_dim': 16,
    'top_k': 5,
    # 'item_dim': 3846,     # lastfm-20
    # 'user_dim': 1872,
    'item_dim': 3953,       # movielens-1m
    'user_dim': 6041,
    # 'item_dim': 8000,       # book_crossing
    # 'user_dim': 2947,

    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_fea_item': 2,
    'item_fea_len': 26,

    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'num_fea_user': 4,

    # model setting
    'embedding_dim': 32,
    'user_embedding_dim': 16,  # 1 features
    'item_embedding_dim': 16,  # 1 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 5e-4,
    'mp_lr': 5e-3,
    'local_lr': 5e-3,
    'batch_size': 32,  # for each batch, the number of tasks
    'num_epoch': 500,
    'neigh_agg': 'mean',
    'mp_agg': 'mean',
}


states = ["meta_training", "user_cold_testing"]