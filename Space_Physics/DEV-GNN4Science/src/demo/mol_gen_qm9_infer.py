import networkx as nx
import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/mol_gen_qm9_infer.yaml')

    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)
    
    # load checkpoint
    ckpt_dict = gnn4s.utils.load_ckpt(f'{config.ckpt.dirt}/{config.ckpt.name}.pth')
    #ckpt_dict = torch.load(ckpt_path, map_location=config.device)
    ckpt_config = ckpt_dict['config']
    logger.log_config(vars(ckpt_config))
    
    # dataset
    dataset = gnn4s.data.Dataset(name=config.data.name, dirt=config.data.dirt, 
                                 load_fn=gnn4s.data.load_mol, process_fn=gnn4s.data.transform_qm9)
    train_idx, test_idx = gnn4s.data.load_idx(
        config.data.dirt, config.data.name, dataset.len())
    train_dataset, test_dataset = gnn4s.data.split_dataset(
        dataset=dataset, train_idx=train_idx, test_idx=test_idx)
    
    smiles_data = gnn4s.data.load_smiles(config.data.dirt, config.data.name)
    train_smiles = gnn4s.data.canonicalize_smiles(list(smiles_data.loc[train_idx]))
    test_smiles = gnn4s.data.canonicalize_smiles(list(smiles_data.loc[test_idx]))

    # model
    model_x = gnn4s.model.load_model_from_ckpt(
        ckpt_dict['params_x'], ckpt_dict['x_state_dict'], config.device)
    model_adj = gnn4s.model.load_model_from_ckpt(
        ckpt_dict['params_adj'], ckpt_dict['adj_state_dict'], config.device)
    
    # sde
    sde_x = gnn4s.diffusion.VESDE(sigma_min=ckpt_config.sde.x.beta_min, 
                                  sigma_max=ckpt_config.sde.x.beta_max,
                                  time_steps_num=ckpt_config.sde.x.num_scales)
    sde_adj = gnn4s.diffusion.VESDE(sigma_min=ckpt_config.sde.adj.beta_min,
                                    sigma_max=ckpt_config.sde.adj.beta_max,
                                    time_steps_num=ckpt_config.sde.adj.num_scales)
    
    # sampler
    sampler = gnn4s.diffusion.PCSampler(model_x=model_x, model_a=model_adj,
        sde_x=sde_x, sde_a=sde_adj, continuous=True, 
        probability_flow=config.sample.probability_flow, 
        snr=config.sample.signal_noise_ratio, scale_eps=config.sample.scale_eps, 
        time_steps_num=config.sample.time_steps_num_each, 
        eps=config.sample.eps, denoise=config.sample.noise_removal, 
        device=config.device)

    # generate samples
    train_graph_list = [nx.convert_matrix.from_numpy_array(
        gnn4s.data.edge_idx_to_adj(9,data.edge_index,data.edge_weight)) for data in train_dataset]
    test_graph_list = gnn4s.data.mol_to_nx(gnn4s.data.smiles_to_mol(test_smiles))    
    
    shape_x = (config.inference.sample_size, ckpt_config.data.max_node_num, ckpt_config.data.max_feat_num)
    shape_adj = (config.inference.sample_size, ckpt_config.data.max_node_num, ckpt_config.data.max_node_num)
    
    init_mask = gnn4s.data.init_mask(train_graph_list, ckpt_config, config.inference.sample_size).to(config.device)
    x, adj, _ = sampler.sample(init_mask, shape_x=shape_x, shape_a=shape_adj)
    x, adj = gnn4s.data.quantize_qm9(x, adj)
    chem_mol = gnn4s.data.ChemMol(x, adj, [6,7,8,9,0])
    gen_smiles = gnn4s.data.mol_to_smiles(chem_mol.mol)
    
    # save and plot generated samples
    gnn4s.data.save_smiles(gen_smiles, config.output.dirt, f"{config.data.name}-sample")
    gnn4s.postprocess.plot_smiles_rdkit(gen_smiles[:16], 
        save_path=f'{config.output.dirt}/{config.data.name}.png')
    
    # evaluation
    '''
    scores = gnn4s.evaluation.moses_metrics(train_smiles, test_smiles, gen_smiles, config.device, 8)
    scores_nspdk = gnn4s.evaluation.eval_graph_list(test_graph_list, 
        gnn4s.data.mol_to_nx(chem_mol.mol), methods=['nspdk'])['nspdk']
    
    logger.log_info(f'number of molecules: {chem_mol.mol_num}')
    logger.log_info(f'validity w/o correction: {chem_mol.no_correct_num / chem_mol.mol_num}')
    for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test']: # , 'Novelty'
        logger.log_info(f'{metric}: {scores[metric]}')
    logger.log_info(f'NSPDK MMD: {scores_nspdk}')
    '''
if __name__ == '__main__':
    main()