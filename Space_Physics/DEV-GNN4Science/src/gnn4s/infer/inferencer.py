import os
import h5py
import torch

class InferencerCat():
    def __init__(self, dataloader, normalizer, model, device):
        """ initializaiton
        Args:
            dataloader: data loader
            normalizer: normalizer
            model: model
            device: device
        """
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.model = model
        self.device = device
    
    def inference(self, dirt):
        """ implement inference
        Args:
            dirt: directory for saving predicition
        """
        pre_dict = {}
        for data in self.dataloader:
            data = data.to(self.device)
            
            # data name
            data_name = data[0].name.split("/")[0]
            if data_name not in pre_dict:
                pre_dict[data_name] = {}
            
            # node position
            if 'node_pos_x' not in pre_dict[data_name]:
                pre_dict[data_name]['node_pos_x'] = []
                pre_dict[data_name]['node_pos_y'] = []
                pre_dict[data_name]['node_pos_z'] = []
            pre_dict[data_name]['node_pos_x'].append(data['node_pos'][:,0:1].detach().cpu())
            pre_dict[data_name]['node_pos_y'].append(data['node_pos'][:,1:2].detach().cpu())
            pre_dict[data_name]['node_pos_z'].append(data['node_pos'][:,2:3].detach().cpu())
            
            # predition
            pre = self.model(data)
            for key in pre:
                pre[key] = pre[key].detach().cpu()
            self.normalizer.inverse_normalize(pre)

            for key in pre:
                if key not in pre_dict[data_name]:
                    pre_dict[data_name][key] = []
                pre_dict[data_name][key].append(pre[key])
        
        # save prediction
        pre_name_list = []
        for data_name in pre_dict:
            for key in pre_dict[data_name]:
                pre_dict[data_name][key] = torch.cat(pre_dict[data_name][key]).numpy()
            
            os.makedirs(dirt, exist_ok=True)
            with h5py.File(f'{dirt}/{data_name}_pre.h5', "w") as f:
                for key, val in pre_dict[data_name].items():
                    f.create_dataset(key, data=val)
        
            pre_name_list.append(data_name)

        def custom_sort_key(s):
            return (len(s), s)
        pre_name_list = sorted(pre_name_list, key=custom_sort_key)  
        
        tmp = [pre_name_list[i:i+3] for i in range(0,len(pre_name_list),3)]
        pre_name_list = [row for col in zip(*tmp) for row in col]
        
        return pre_name_list