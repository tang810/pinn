import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/flag_simple_infer.yaml')

    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)

    # load checkpoint
    ckpt_dict = gnn4s.utils.load_ckpt(f'{config.ckpt.dirt}/{config.ckpt.name}.pth')
    ckpt_config = ckpt_dict['config']
    logger.log_config(vars(ckpt_config))

    # mask
    node_type_list = {'NORMAL': 0, 'OBSTACLE': 1, 'AIRFOIL': 2, 'HANDLE': 3,
                      'INFLOW': 4, 'OUTFLOW': 5, 'WALL_BOUNDARY': 6, 'SIZE': 9}
    mask_fn = gnn4s.data.NodeTypeMask(node_type_list, config.data.node_type)

    # datasets
    keys = ['mesh_pos', 'world_pos', 'cells', 'node_type']
    load_fn = gnn4s.utils.partial(gnn4s.data.load_npy, keys=keys)
    process_fn = gnn4s.utils.partial(gnn4s.data.euler_lagrange_to_pyg_data,
                                     sys_type='lagrange', mask_fn=mask_fn)
    dataset = gnn4s.data.Dataset(name=config.data.name, dirt=config.data.dirt,
                                 load_fn=load_fn, process_fn=process_fn)
    
    # dataloader
    dataloader = gnn4s.data.DataLoader(dataset=dataset, shuffle=False)
    
    # model
    model = gnn4s.model.MeshGraphNet(node_input_dim=ckpt_config.model.node_input_dim, 
                                     edge_input_dim=ckpt_config.model.edge_input_dim,
                                     output_dim=ckpt_config.model.output_dim,
                                     block_num=ckpt_config.model.block_num)
    model.load_state_dict(ckpt_dict['model_state_dict'], config.device)
    
    # evaluator
    predictor = gnn4s.infer.PredictorLagrange(model, config.device)
    evaluator = gnn4s.evaluation.Evaluator(dataloader, predictor)
    
    # evaluate
    error, pred = evaluator.evaluate()
    for i in range(len(error)):
        logger.log_info(error[i])
    
    # postprocess
    for data in dataloader:
        gnn4s.postprocess.plot_anime_trimesh3d(data, pred[0], config.output.name, config.output.dirt,
                                               ratio=0.7, intervel=200, dpi=config.output.dpi)
        break

if __name__=='__main__':
    main()