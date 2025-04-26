import torch



task = 'bbbp' # in ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp'] -------->  CHANGE <--------
data_path = 'test.txt' # txt file of smiles                         -------->  CHANGE <--------
save_path = 'test_results.txt'                                    # -------->  CHANGE <--------
batch_size = 128
ckpt_path = '../checkpoints'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
thresholds = {
    'bbbp': 0.5,
    'ames': 0.3,
    'dili': 0.4,
    'hia': 0.2,
    'pgp': 0.3,
    'cyp3a4': 0.55
}
