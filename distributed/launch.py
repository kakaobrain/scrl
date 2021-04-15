# Modified detectron2.engine.launch
import datetime
import logging
import os

import torch
import torch.distributed as dist
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch.multiprocessing as mp

from distributed import comm


__all__ = ["launch"]


def launch(main_func, num_gpus_per_machine=torch.cuda.device_count(),
           num_machines=1, machine_rank=0, dist_url='auto', args=()):
    """
    Spawn (num_gpus_per_machine - 1) processes and proceed as a worker itself
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1 and comm.is_local_main_process():
        # os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # os.environ["NCCL_IB_DISABLE"] = "1"
        # os.environ['NCCL_BLOCKING_WAIT'] = "1"
        # os.environ['NCCL_P2P_LEVEL'] = "5"
        # os.environ['NCCL_TREE_THRESHOLD'] = "0"
        # os.environ['NCCL_P2P_DISABLE'] = "1"
        # os.environ['NCCL_SHM_DISABLE'] = '1'

        spawn_ctx = mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine - 1,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args),
            daemon=False, join=False
        )

        if comm.is_local_main_process():
            _distributed_setup(0, world_size, num_gpus_per_machine, machine_rank, dist_url)

        # Need to keep spawn context when join=False: https://github.com/pytorch/pytorch/issues/30461
        return spawn_ctx
    else:
        return None


def _distributed_worker(
    local_rank, main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args
):
    # parent process is local_rank = 0
    local_rank = local_rank + 1
    _distributed_setup(local_rank, world_size, num_gpus_per_machine, machine_rank, dist_url)
    main_func(*args)


def _distributed_setup(
    local_rank, world_size, num_gpus_per_machine, machine_rank, dist_url
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank,
            timeout=datetime.timedelta(0, 60*30)
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

