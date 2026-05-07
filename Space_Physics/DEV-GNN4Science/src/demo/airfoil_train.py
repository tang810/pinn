import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/airfoil_train.yaml')

    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)
    
    # dataset
    var_dict = {'node_pos': ['node_pos_x','node_pos_y','node_pos_z'], 
                'node_attr': ['deg', 'mach'],
                'node_label': ['den', 'pre', 'tem', 'vel_x', 'vel_y', 'vel_z']}
    load_fn = gnn4s.data.load_h5_from_name_list
    process_fn = gnn4s.utils.partial(gnn4s.data.to_pyg_data, var_dict=var_dict)
    normalizer = gnn4s.data.Normalizer(var_dict)
    train_dataset = gnn4s.data.Dataset(f'{config.data.name}_train', config.data.dirt,
                                       load_fn, process_fn, normalizer)
    valid_dataset = gnn4s.data.Dataset(f'{config.data.name}_valid', config.data.dirt,
                                       load_fn, process_fn, normalizer)
    
    # dataloader
    train_dataloader = gnn4s.data.DataLoader(train_dataset)
    valid_dataloader = gnn4s.data.DataLoader(valid_dataset, shuffle=False)
    
    # model
    model = gnn4s.model.GCNTransformer(var_dict, config.model.node_dim,
        config.model.edge_dim, config.model.output_dim, 
        config.model.hidden_dim, config.model.block_num)
    model.to(config.device)
    
    # loss and error
    loss = gnn4s.train.LossLp(model=model, var_dict=var_dict)
    error = gnn4s.train.ErrorRelativeLp(model=model, var_dict=var_dict)
    
    # train
    optimizer = gnn4s.train.AdamOptimizer(model.parameters(), lr=config.train.lr)
    scheduler = gnn4s.train.StepLR(optimizer, step_size=config.train.scheduler_step,
                                   gamma=config.train.scheduler_gamma)
    
    trainer = gnn4s.train.Trainer(
        train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
        model=model, loss=loss, error=error, optimizer=optimizer, scheduler=scheduler, 
        epoch_num=config.train.epoch_num, print_interval=10, save_dirt=config.output.dirt,
        ckpt_dirt=config.ckpt.dirt, ckpt_name=config.ckpt.name,
        logger=logger, device=config.device)
    trainer.train()
    
if __name__=='__main__':
    main()