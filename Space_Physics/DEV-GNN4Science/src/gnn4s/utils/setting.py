import torch
import resource

def set_sharing_strategy(strategy: str='file_system') -> None:
    """ Configuring file system resource sharing strategy
    Args:
        strategy: strategy to share memory between multiple processes.
        e.g. 'file_system', 'file_descriptor'
    """ 
    torch.multiprocessing.set_sharing_strategy(strategy)

def setrlimit(file_num_limit: int) -> None:
    """ Setting file descriptor limits in multi-process training.
    Args:
        file_num_limit: the maximum number of files that can be opened
    """
    # Retrieve the file descriptor limits of the current process
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

    # Set the soft limit of file descriptors to file_num_limit while keeping the hard limit unchanged. 
    resource.setrlimit(resource.RLIMIT_NOFILE, (file_num_limit, rlimit[1]))

def use_cudnn_benchmark(use: bool=True) -> None:
    """ Configuring optimization options for the CuDNN library in PyTorch, 
        allowing runtime selection of the optimal convolution algorithm 
        by measuring the performance of different algorithms, to enhance training performance.
    Args:
        use: whether to use cudnn benchmark
    """
    torch.backends.cudnn.benchmark = use