import torch
import json
import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/cardiovascular_train.yaml')
    
    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)

    # feature
    nodes_features = ['area', 'tangent', 'type', 'T', 'dip', 'sysp',
                      'resistance1', 'capacitance', 'resistance2', 'loading']
    edges_features = ['rel_position', 'distance', 'type']
    features = {'nodes_features': nodes_features, 'edges_features': edges_features}

    # types_to_keep: list of graph types to keep
    # 'synthetic' refers to the bcs, not the geometry
    dataset_info = json.load(open(f'{config.data.dirt}/dataset_info.json'))
    types_to_keep = ['synthetic_aorta_coarctation', 'synthetic_pulmonary', 
                     'synthetic_aortofemoral']
    types_to_keep = {'dataset_info': dataset_info, 'types_to_keep': types_to_keep}
    
    # dataset
    normalize_fn = gnn4s.utils.partial(gnn4s.data.generate_normalized_graphs, 
        norm_type=config.data.norm_type, bc_type=config.data.bc_type, 
        types_to_keep=types_to_keep, n_graphs_to_keep=-1, features=features)
    split_fn = gnn4s.utils.partial(gnn4s.data.split, split_type='train', dataset_info=dataset_info)
    process_fn = gnn4s.utils.partial(gnn4s.data.graph_to_lightgraph, stride=config.model.stride)
    noise_fn = gnn4s.utils.partial(gnn4s.data.noise.add_noise_to_lightgraph,
        noise_rate=config.data.noise_rate, noise_rate_feat=config.data.noise_rate_feat, 
        stride=config.model.stride)
    dataset = gnn4s.data.DglDataset(config.data.name, config.data.dirt, split_fn, 
        load_fn=gnn4s.data.load_graphs, normalize_fn=normalize_fn, 
        process_fn=process_fn, noise_fn=noise_fn)

    print(dataset.data_list[0].ndata.keys())
    print(dataset.data_list[0].edata.keys())
    
    # dataloader
    dataloader = gnn4s.data.DglDataLoader(dataset, batch_size=config.data.batch_size)

    # model
    model = gnn4s.model.DglMeshGraphNet(config.model.input_dim_node, config.model.input_dim_edge,
        config.model.output_dim, config.model.hidden_dim_gnn, config.model.hidden_dim_mlp, 
        config.model.layer_num_gnn, config.model.layer_num_mlp)
    
    # loss
    loss = gnn4s.train.LossMSE(model, config.model.stride, dataset.bc_type)

    # train
    optimizer = torch.optim.Adam(model.parameters(), config.train.lr, 
                                 weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epoch_num,
        eta_min=config.train.lr*config.train.lr_decay)

    trainer = gnn4s.train.Trainer(train_dataloader=dataloader, model=model, 
        loss=loss, optimizer=optimizer, lr=config.train.lr, scheduler=scheduler, 
        epoch_num=config.train.epoch_num, logger=logger, device=config.device)
    model, history = trainer.train()
    
    # postprocess
    for k in history:
        gnn4s.postprocess.plot_history(history[k], k, k, config.output.dirt)

if __name__ == "__main__":
    main()