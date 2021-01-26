PARA = dict(
    GACNN_params=dict(
        epoch=1,
        num_layers=[5, 10],  # 5表示下限，10表示上限
        num_neurons=[16, 256],
        lr=[0.0001, 0.2],
        batch_size=[1, 2],  # 之后要重新乘以100
        save_data_txt = './cache/data/SaveData.txt',
        checkpoint_path = './cache/checkpoint/',
        # activate_func = ['Relu','sigmoid','tanh'],
    ),
    mnist_params=dict(
        root = './Mnist',
        in_dim = 28*28,
        out_dim = 10,
        hidden_layers = 3,
        hidden_neurons = [16,32,64,64],
    ),
    train=dict(
        epochs = 100,
        batch_size = 64,
        lr = 0.001,
        momentum=0.9,
        wd = 5e-4,
        num_workers = 2,
        divice_ids = [1],
        gpu_id = 0,
        num_classes=10,
    ),
    test=dict(
        batch_size=100
    ),
    cifar10_paths = dict(
        validation_rate = 0.05,

        root = '../../DATASET/cifar10/',

        original_trainset_path = '../../../DATASET/cifar10/cifar-10-python/',#train_batch_path
        original_testset_path = '../../../DATASET/cifar10/cifar-10-python/',

        after_trainset_path = '../../../DATASET/cifar10/trainset/',
        after_testset_path = '../../../DATASET/cifar10/testset/',
        after_validset_path = '../../../DATASET/cifar10/validset/',

        train_data_txt = '../../../DATASET/cifar10/train.txt',
        test_data_txt = '../../../DATASET/cifar10/test.txt',
        valid_data_txt = '../../../DATASET/cifar10/valid.txt',
    ),
    utils_paths = dict(
        checkpoint_path = './cache/checkpoint/',
        log_path = './cache/log/',
        visual_path = './cache/visual/',
        params_path = './cache/params/',
    ),
)