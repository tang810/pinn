'''
from moses.metrics.metrics import get_all_metrics

def moses_metrics(train_smiles, test_smiles, gen_smiles, device: str='cpu', njobs: int=1):
    return get_all_metrics(test=test_smiles, gen=gen_smiles, k=len(gen_smiles), 
                           device=device, n_jobs=njobs)
'''