import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/airfoil_infer.yaml')
    
    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)

    # load checkpoint
    ckpt_dict = gnn4s.utils.load_ckpt(f'{config.ckpt.dirt}/{config.ckpt.name}.pth')
    ckpt_config = ckpt_dict['config']
    logger.log_config(vars(ckpt_config))

    # dataset
    var_dict = {'node_pos': ['node_pos_x','node_pos_y','node_pos_z'], 
                'node_attr': ['deg', 'mach'],
                'node_label': ['den', 'pre', 'tem', 'vel_x', 'vel_y', 'vel_z']}
    load_fn = gnn4s.data.load_h5_from_name_list
    process_fn = gnn4s.utils.partial(gnn4s.data.to_pyg_data, var_dict=var_dict)
    normalizer = gnn4s.data.Normalizer(var_dict)
    normalizer.load_state_dict(ckpt_dict['normalizer_state_dict'])
    
    test_dataset = gnn4s.data.Dataset(f'{config.data.name}_test', config.data.dirt,
                                      load_fn, process_fn, normalizer)
    test_dataloader = gnn4s.data.DataLoader(test_dataset)
    
    # model
    model = gnn4s.model.GCNTransformer(var_dict, ckpt_config.model.node_dim,
        ckpt_config.model.edge_dim, ckpt_config.model.output_dim, 
        ckpt_config.model.hidden_dim, ckpt_config.model.block_num)
    model.load_state_dict(ckpt_dict['model_state_dict'])
    model.to(config.device)
    
    # infer
    inferencer = gnn4s.infer.InferencerCat(dataloader=test_dataloader, normalizer=normalizer,
                                           model=model, device=config.device)
    pre_name_list = inferencer.inference(config.output.dirt)

    # interpolation
    gnn4s.postprocess.interpolation_data(pre_name_list, f'{config.output.dirt}',
        f'{config.data.dirt}/{config.data.sec_name}.h5', config.output.dirt, var_dict)
    
    # visualization
    gnn4s.postprocess.plot_airfoil_anime(pre_name_list, config.output.dirt)

if __name__=='__main__':
    main()