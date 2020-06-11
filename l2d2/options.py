def add_model_hparams(parser):
    parser.add_option('--batch-size', type=int, default=128,
                      help='minibatch size')
    parser.add_option('--carry-len', type=int, default=1024,
                      help='vector length for recurrent state')
    parser.add_option('--step-size', type=float, default=1e-6,
                      help='optimizer step size')
