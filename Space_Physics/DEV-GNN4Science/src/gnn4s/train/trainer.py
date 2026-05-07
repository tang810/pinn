import os
import torch
import numpy as np
import time
import resource
import itertools
from types import SimpleNamespace
from torch_geometric.loader import DataLoader, DataListLoader

class ExponentialMovingAverage:
    """ Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters: torch.nn.Parameter, decay: float, use_num_updates: bool=True) -> None:
        """
        Args:
            parameters: iterable of 'torch.nn.Parameter'; usually the result of 'model.parameters()'.
            decay: the exponential decay.
            use_num_updates: whether to use number of updates when computing averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """ Update currently maintained parameters.
            Call this every time the parameters are updated, 
            such as the result of the 'optimizer.step()' call.
        Args:
            parameters: iterable of 'torch.nn.Parameter';
                usually the same set of parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_((1.0 - decay) * (s_param - param))

    def copy_to(self, parameters: torch.nn.Parameter) -> None:
        """ Copy current parameters into given collection of parameters.
        Args:
            parameters: Iterable of 'torch.nn.Parameter'; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters: torch.nn.Parameter) -> None:
        """ Save the current parameters for restoring later.
        Args:
            parameters: iterable of 'torch.nn.Parameter'; the parameters to be temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters: torch.nn.Parameter) -> None:
        """ Restore the parameters stored with the 'store' method.
            Useful to validate the model with EMA parameters without affecting the original optimization process.
            Store the parameters before the 'copy_to' method.
            After validation (or model saving), use this to restore the former parameters.
        Args:
            parameters: iterable of 'torch.nn.Parameter'; the parameters to be updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def save_state_dict(self):
        state_path = 'checkpoints/ema_state.pth'
        state_dict = {'decay': self.current_epoch,
                      'num_updates': self.num_updates,
                      'shadow_params': self.shadow_params}  
        torch.save(state_dict, state_path)
        
    def load_state_dict(self, state_path):
        state_dict = torch.load(state_path)
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']

class Trainer():
    """ Trainer for training model """
    def __init__(self, train_dataloader=None, valid_dataloader=None, 
                 model=None, loss=None, error=None,
                 optimizer=None, lr: float=0.001, scheduler=None, 
                 epoch_num: int=10000, warmup_step: int=0, grad_clip: float=0.0,
                 ema_rate: float=0.9999, use_ema: bool=False, 
                 print_interval: int=1, save_interval: int=1, save_dirt: str=None, 
                 ckpt_name: str=None, ckpt_dirt: str=None, logger=None, device='cpu') -> None:
        """ initialization
        Args:
            train_dataloader: train dataloader
            valid_dataloader: valid dataloader
            model: model
            loss: loss function
            error: error function
            optimizer: optimizer for minimizing the loss function
            lr: learning rate
            scheduler: learning rate scheduler
            epoch_num: number of epochs
            warmup_step: number of warmup steps for learning rate
            grad_clip: maximum norm of gradients
            ema_rate: smoothing coefficient
            use_ema: whether to use Exponential Moving Average
            print_interval: interval for print
            save_interval: interval for saving the model
            save_dirt: directory for saving the data
            ckpt_name: name of the checkpoint
            ckpt_dirt: directory for saving the checkpoint
            logger: logger
            device: computing device
        """
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.loss = loss
        self.error = error
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.epoch_num = epoch_num
        self.warmup_step = warmup_step
        self.grad_clip = grad_clip
        self.ema_rate = ema_rate
        self.use_ema = use_ema
        self.print_interval = print_interval
        self.save_interval = save_interval
        self.save_dirt = save_dirt
        self.ckpt_name = ckpt_name
        self.ckpt_dirt = ckpt_dirt
        self.logger = logger
        self.device = device
        
        self.current_epoch = 0
        self.current_step = 0
        self.batch_num = len(self.train_dataloader)

        #if not isinstance(self.model, dict):
        #    self.model = {'model': self.model}
        
        self.parameters = self.collected_params()
        if self.use_ema:
            self.ema = ExponentialMovingAverage(self.parameters, self.ema_rate)

    def collected_params(self):
        if not isinstance(self.model, dict):
            return self.model.parameters()
        else:
            parameters = []
            for key in self.model:
                parameters.append(self.model[key].parameters())
            return parameters
    
    def step(self, batch_data) -> torch.Tensor:
        """ Running one step of training
            For jax version: This function will undergo 'jax.lax.scan' so that 
            multiple steps can be pmapped and jit-compiled together for faster execution.
        Args:
            batch_data: batch data
        Returns:
            loss: value of loss function after training
        """
        # 1: zero gradient
        self.optimizer.zero_grad()

        # 2: forward propagation
        loss = self.loss(batch_data)
        loss_sum = sum(loss.values())
        
        # 3: backward propagation
        loss_sum.backward()
        
        # 4: optimization
        # learning rate warmup
        if self.warmup_step > 0:
            for param_groups in self.optimizer.param_groups:
                param_groups['lr'] = self.lr * np.minimum(self.current_step/self.warmup_step, 1.0)
        
        # gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=self.grad_clip)

        # optimization
        self.optimizer.step()

        # exponential moving average
        if self.use_ema:
            self.ema.update(self.parameters)

        self.current_step += 1
        return loss
    
    def train(self):
        """ Train model """
        info = 'start training'
        self.logger.log_info(info)
        self.loss_history = {}
        self.error_history = {}
        start_time = time.time()

        if not isinstance(self.model, dict):
            self.model.train()
        else:
            for key in self.model:
                self.model[key].train()
        
        # loop
        while self.current_epoch < self.epoch_num:
            i = 0
            for batch_data in self.train_dataloader:
                batch_data = batch_data.to(self.device)
                loss = self.step(batch_data)

                # print
                if i % self.print_interval==0:
                    info = f'epoch: {self.current_epoch} | batch: {i}'
                    for key in loss:
                        info += f' | {key}: {loss[key]:.3e}'
                    info += f' | time: {time.time()-start_time:.2f}'
                    self.logger.log_info(info)
                
                i += 1
                start_time = time.time()

            for key in loss:
                if key not in self.loss_history:
                    self.loss_history[key] = [[],[]]
                self.loss_history[key][0].append(self.current_epoch)
                self.loss_history[key][1].append(loss[key].item())
            
            if self.valid_dataloader is not None and self.error is not None:
                for batch_data in self.valid_dataloader:
                    batch_data = batch_data.to(self.device)
                    error = self.error(batch_data)
                    break
                
                for key in error:
                    if key not in self.error_history:
                        self.error_history[key] = [[],[]]
                    self.error_history[key][0].append(self.current_epoch)
                    self.error_history[key][1].append(error[key].item())
                
                # print
                info = f'epoch: {self.current_epoch}'
                for key in error:
                    info += f' | {key}: {error[key]:.3e}'
                self.logger.log_info(info)    

            # schedule learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            
            self.current_epoch += 1
            
            # save model
            if self.current_epoch % self.save_interval == 0:
                self.save_checkpoint()

        # save loss and error history
        if self.save_dirt is not None:
            self.save_history()

        return self.model, self.loss_history, self.error_history
    
    def save_history(self):
        os.makedirs(self.save_dirt, exist_ok=True)
        for key in self.loss_history:
            tmp = np.stack([np.array(self.loss_history[key][0]),
                            np.array(self.loss_history[key][1])],1)
            np.savetxt(f'{self.save_dirt}/loss_history_{key}.txt', tmp)
        for key in self.error_history:
            tmp = np.stack([np.array(self.error_history[key][0]),
                            np.array(self.error_history[key][1])],1)
            np.savetxt(f'{self.save_dirt}/error_history_{key}.txt', tmp)

    def save_checkpoint(self):
        """ save checkpoint """
        ckpt_dict = {}
        ckpt_dict['model_state_dict'] = self.model.state_dict()
        
        config_dict = {}
        config_dict['model'] = self.model.get_model_config()
        config_namespace = SimpleNamespace(**config_dict)
        ckpt_dict['config'] = config_namespace
        
        if hasattr(self.train_dataloader.dataset, 'normalizer') and (
            self.train_dataloader.dataset.normalizer is not None):
            ckpt_dict['normalizer_state_dict'] = (
                self.train_dataloader.dataset.normalizer.get_state_dict())

        os.makedirs(self.ckpt_dirt, exist_ok=True)
        torch.save(ckpt_dict, f'{self.ckpt_dirt}/{self.ckpt_name}.pth')

class TrainerDiffDock():
    def __init__(self, dataset, batch_size: int=16, worker_num: int=1, pin_memory: bool=False,
                 model=None, loss=None, optimizer=None, scheduler=None, 
                 epoch_num: int=10000, warmup_step: int=0, grad_clip: float=0.0,
                 ema_rate: float=0.9999, use_ema: bool=False, save_interval: int=1,
                 device='cpu', logger=None) -> None:
        """ Trainer for training model.
        Args:
            dataset: dataset
            batch_size: batch size
            worker_num: number of worker
            pin_memory: whether to use pin memory
            model: model 
            loss: loss funtion
            optimizer: optimizer for minimizing the loss function
            scheduler: learning rate scheduler
            epoch_num: number of epochs
            warmup_step: number of warmup steps for learning rate
            grad_clip: maximum norm of gradients
            ema_rate: smoothing coefficient
            use_ema: whether to use Exponential Moving Average
            save_interval: interval for saving the model
            device: computing device
            logger: logger 
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.pin_memory = pin_memory
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_num = epoch_num
        self.warmup_step = warmup_step
        self.grad_clip = grad_clip
        self.ema_rate = ema_rate
        self.use_ema = use_ema
        self.save_interval = save_interval
        self.device = device
        self.logger = logger

        self.loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
        self.loader = self.loader_class(dataset=self.dataset, batch_size=self.batch_size,
            num_workers=self.worker_num, shuffle=True,
            pin_memory=self.pin_memory)
        self.loader_iter = iter(self.loader)

        self.model.train()
        self.current_epoch = 0
        self.current_step = 0

        if self.use_ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), self.ema_rate)
        
    def step(self, batch):
        """ Running one step of training.
        Returns:
            loss: The average loss value of this state.
        """
        if self.device == 'cuda' and len(batch) == 1 or self.device == 'cpu' and batch.graph_num == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        
        try:
            # 1: zero gradient
            self.optimizer.zero_grad()

            # 2: forward propagation
            loss = self.loss(batch)

            # 3: backward propagation
            loss.backward()

            # 4: optimization
            # learning rate warmup
            if self.warmup_step > 0:
                for param_groups in self.optimizer.param_groups:
                    param_groups['lr'] = self.lr * np.minimum(self.current_step/self.warmup_step, 1.0)
        
            # gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=self.grad_clip)
            
            self.optimizer.step()

            # exponential moving average
            if self.use_ema:
                self.ema.update(self.model.parameters())
            
            self.current_step += 1

        except RuntimeError as e:
            self.optimizer.zero_grad()
            loss = None

            if 'out of memory' in str(e):
                warning = '| WARNING: ran out of memory, skipping batch'
                self.logger.log_warning(warning)
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

            elif 'Input mismatch' in str(e):
                warning = '| WARNING: weird torch_cluster error, skipping batch'
                self.logger.log_warning(warning)
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

            else:
                raise e
        
        return loss

    def train(self):
        """ Train model. """

        info = 'start training'
        self.logger.log_info(info)
        start_time = time.time()

        # evaluate inital loss
        batch = next(self.loader_iter)
        loss = self.loss(batch)

        # print
        if loss is not None:
            info = f'epoch: {self.current_epoch} | loss: {loss.item():.3e}'
            self.logger.log_info(info)

        # save model
        # self.save_checkpoint()

        # loop
        while self.current_epoch < self.epoch_num:
            i = 0
            for batch in self.loader:
                # train neural networks
                loss = self.step(batch)

                # print
                if loss is not None:
                    info = (f'epoch: {self.current_epoch} | batch: {i} | '+ 
                            f'loss: {loss.item():.3e} | time: {time.time()-start_time:.2f}')
                    self.logger.log_info(info)
                
                i += 1
                start_time = time.time()

            # schedule learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            self.current_epoch += 1

            # save model
            # if self.current_epoch % self.save_interval == 0:
            #     self.save_checkpoint()
        
    def save_checkpoint(self):
        """ state: A dictionary of training information, containing 
            the score model and number of optimization epochs.
        """
        os.makedirs('checkpoints', exist_ok=True)
        model_path = 'checkpoints/model.pth'
        state = {'current_epoch': self.current_epoch,
                 'model_state_dict': self.model.state_dict()}
        torch.save(state, model_path)
        
    def load_checkpoint(self, model_path):
        state = torch.load(model_path)
        self.current_epoch = state['current_epoch']
        self.current_step = self.current_epoch*self.batch_num
        self.model.load_state_dict(state['model_state_dict'])