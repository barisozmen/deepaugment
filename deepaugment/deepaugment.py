







class DeepAugment():

    __init__(
        data="cifar10",
        labels=None,
        model="basiccnn",
        method="bayesian_optimization",
        train_set_size = 2000,
        opt_iterations = 300,
        opt_samples = 3,
        opt_last_n_epochs = 3,
        opt_initial_points = 10,
        child_epochs = 50,
        child_first_train_epochs = 0,
        child_batch_size = 64,
    ):

    if type(data)==str:
        _data, input_shape = DataOp.load(data, train_set_size)
        _data = DataOp.preprocess(_data)
    else:
        data[""]

    num_classes = find_num_classes(data["y_train"])

    if type(model)==str:
        child_model = ChildCNN(
            _model,
            input_shape,
            child_batch_size,
            num_classes,
            "initial_model_weights.h5",
            logging,
        )
    else:
        pass

    if method=="bayesian_optimization":
        run_bayesian_optimization()
    elif method=="random_search":

    else:
        raise ValueError

