import json
import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/cardiovascular_infer.yaml')

    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)
    
    # load checkpoint
    ckpt_dict = gnn4s.utils.load_ckpt(f'{config.ckpt.dirt}/{config.ckpt.name}.pth')
    ckpt_config = ckpt_dict['config']
    logger.log_config(vars(ckpt_config))
    
    # dataset
    dataset_info = json.load(open(f'{config.data.dirt}/dataset_info.json'))
    types_to_keep = ['synthetic_aorta_coarctation', 
                     'synthetic_pulmonary', 
                     'synthetic_aortofemoral']
    types_to_keep = {'dataset_info': dataset_info, 'types_to_keep': types_to_keep}

    normalize_fn = gnn4s.utils.partial(gnn4s.data.generate_normalized_graphs, 
        norm_type=ckpt_config.data.statistics['normalization_type'],
        bc_type=ckpt_config.data.bc_type,
        types_to_keep = types_to_keep,
        statistics=ckpt_config.data.statistics)
    split_fn = gnn4s.utils.partial(gnn4s.data.split, split_type='test', dataset_info=dataset_info)
    process_fn = gnn4s.utils.partial(gnn4s.data.graph_to_lightgraph, stride=ckpt_config.data.stride)
    noise_fn = gnn4s.utils.partial(gnn4s.data.noise.add_noise_to_lightgraph, 
        noise_rate=ckpt_config.data.rate_noise, 
        noise_rate_feat=ckpt_config.data.rate_noise_features, 
        stride=ckpt_config.data.stride)
    
    dataset = gnn4s.data.DglDataset(config.data.name, config.data.dirt,
        load_fn=gnn4s.data.load_graphs, normalize_fn=normalize_fn,
        split_fn=split_fn, process_fn=process_fn, noise_fn=noise_fn)
    
    # model    
    model = gnn4s.model.DglMeshGraphNet(ckpt_config.model.input_dim_node,
        ckpt_config.model.input_dim_edge, ckpt_config.model.output_dim,
        ckpt_config.model.hidden_dim_gnn, ckpt_config.model.hidden_dim_mlp, 
        ckpt_config.model.layer_num_gnn, ckpt_config.model.layer_num_mlp)
    model.load_state_dict(ckpt_dict['model_state_dict'])

    # evaluate
    predictor = gnn4s.infer.PredictorCardiovascular(model, ckpt_config.data.bc_type)
    evaluator = gnn4s.evaluation.EvaluatorCardiovascular(dataset, predictor,
        ckpt_config.data.bc_type, ckpt_config.data.statistics)
    pred, error_mean, time_mean, time_step_mean = evaluator.evaluate()
    print('average error')
    print(error_mean)
    
    # postprocess
    idx = 0
    gnn4s.postprocess.plot_anime_3d(dataset.data_list[idx].ndata['x'], pred[idx], 
                                    name='pressure (prediction)', dirt=config.output.dirt)
    gnn4s.postprocess.plot_anime_3d(dataset.data_list[idx].ndata['x'], 
                                    dataset.data_list[idx].ndata['nfeatures'],
                                    name='pressure (truth)', dirt=config.output.dirt)

if __name__ == '__main__':
    main()