def add_model_hparams(parser):
    parser.add_option('--optimizer', default='sgd', choices=('sgd', 'adagrad', 'momentum'))
    parser.add_option('--batch-size', type=int, default=1,
                      help='minibatch size')
    parser.add_option('--carry-len', type=int, default=32,
                      help='vector length for recurrent state')
    parser.add_option('--step-size', type=float, default=1e-4,
                      help='optimizer step size')
