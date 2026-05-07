import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/mol_gen_qm9_train.yaml')
    
    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)
    
    # dataset
    dataset = gnn4s.data.Dataset(name=config.data.name, dirt=config.data.dirt, 
        load_fn=gnn4s.data.load_mol, process_fn=gnn4s.data.transform_qm9)
    train_idx, test_idx = gnn4s.data.load_idx(
        config.data.dirt, config.data.name, dataset.len())
    train_dataset, test_dataset = gnn4s.data.split_dataset(
        dataset=dataset, train_idx=train_idx, test_idx=test_idx)
    
    edge_to_adj = gnn4s.data.EdgeToAdj(node_num_max=config.data.node_num_max, mask=False, 
        channelization=config.data.channelization, centralization=config.data.centralization, 
        remove_diagonal=config.data.remove_diagonal)
    
    # dataloader
    train_dataloader = gnn4s.data.DataLoader(dataset=train_dataset, batch_size=config.data.batch_size)
    
    # model
    model_x = gnn4s.model.DenseScoreNetworkX(input_dim=config.model.x_input_dim).to(config.device)
    model_adj = gnn4s.model.DenseScoreNetworkAdj(input_dim=config.model.adj_input_dim).to(config.device)
    model = {'model_x': model_x, 'model_adj': model_adj}
    
    # sde
    sde_x = gnn4s.diffusion.VESDE(sigma_min=config.sde.beta_min, sigma_max=config.sde.beta_max,
        time_steps_num=config.diffusion.time_step_num)
    sde_adj = gnn4s.diffusion.VESDE(sigma_min=config.sde.beta_min, sigma_max=config.sde.beta_max,
        time_steps_num=config.diffusion.time_step_num)
    
    # loss
    score_x = gnn4s.train.Score(model=model_x, sde=sde_x, train=True, continuous=True)
    score_adj = gnn4s.train.Score(model=model_adj, sde=sde_adj, train=True, continuous=True)
    
    loss = gnn4s.train.LossDiffusion(sde_x, sde_adj, score_x, score_adj, edge_to_adj,
                                     mask_eps=True, reduce_mean=False)
    
    # train
    optimizer = gnn4s.train.AdamOptimizer(gnn4s.utils.chain(model_x.parameters(),
                                                            model_adj.parameters()),
                                          lr=config.train.lr)
    scheduler = gnn4s.train.ExponentialLR(optimizer, gamma=config.train.lr_decay)
    
    trainer = gnn4s.train.Trainer(train_dataloader=train_dataloader, model=model,
        loss=loss, optimizer=optimizer, lr=config.train.lr, scheduler=scheduler,
        epoch_num=config.train.epoch_num, logger=logger, device=config.device)
    trainer.train()

if __name__ == "__main__":
    main()