import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/cylinder_flow_train.yaml')

    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)
    
    # mask
    node_type_list = {'NORMAL': 0, 'OBSTACLE': 1, 'AIRFOIL': 2, 'HANDLE': 3,
                      'INFLOW': 4, 'OUTFLOW': 5, 'WALL_BOUNDARY': 6, 'SIZE': 9}
    mask_fn = gnn4s.data.NodeTypeMask(node_type_list, config.data.mask_node_type)
    mask_noise_fn = gnn4s.data.NodeTypeMask(node_type_list, config.data.mask_noise_node_type)

    # dataset
    keys = ['mesh_pos', 'cells', 'node_type', 'velocity']
    load_fn = gnn4s.utils.partial(gnn4s.data.load_h5, keys=keys)
    process_fn = gnn4s.utils.partial(gnn4s.data.euler_lagrange_to_pyg_data,
        sys_type='euler', mask_fn=mask_fn, mask_noise_fn=mask_noise_fn,
        noise_std=config.data.noise_std, noise_gamma=config.data.noise_gamma,
        label_update=True)
    dataset = gnn4s.data.Dataset(name=config.data.name, dirt=config.data.dirt, 
        load_fn=load_fn, process_fn=process_fn, split_time=True)
    
    # dataloader
    dataloader = gnn4s.data.DataLoader(dataset=dataset, batch_size=config.data.batch_size)

    # model
    model = gnn4s.model.MeshGraphNet(node_input_dim=config.model.node_input_dim, 
                                     edge_input_dim=config.model.edge_input_dim,
                                     output_dim=config.model.output_dim,
                                     block_num=config.model.block_num)
    model.to(config.device)
    
    # loss
    loss = gnn4s.train.LossMS(model)

    # train
    optimizer = gnn4s.train.AdamOptimizer(model.parameters(), lr=config.train.lr)
    scheduler = gnn4s.train.StepLR(optimizer, step_size=config.train.step_size,
                                   gamma=config.train.gamma)
    trainer = gnn4s.train.Trainer(train_dataloader=dataloader, model=model, loss=loss,
                                  optimizer=optimizer, scheduler=scheduler,
                                  epoch_num=config.train.epoch_num,
                                  ckpt_dirt=config.ckpt.dirt, ckpt_name=config.ckpt.name,
                                  logger=logger, device=config.device)
    trainer.train()

if __name__=='__main__':
    main()