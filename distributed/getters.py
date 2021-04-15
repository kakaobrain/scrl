import logging
import socket
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from distributed import comm

log = logging.getLogger(('main'))


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _maybe_gethostbyname(dist_url):
    """to be compatible with Braincloud on which one can access the nodes by their task names.
    Each node has to wait until all the tasks in the group are up on the cloud."""
    url = list(urlsplit(dist_url))
    if url[1][0].isdigit():
        # if TCP address consists of digits, do nothing
        return dist_url
    
    done = False
    retry = 0
    ind = url[1].find(':')
    log.info(f"[DDP] Get URL by the given hostname '{dist_url}' in Braincloud..")
    while not done:
        try:
            url[1] = socket.gethostbyname(url[1][:ind]) + url[1][ind:]
            dist_url = urlunsplit(url)
            # dist_url = dist_url_new
            done = True
        except:
            retry += 1
            log.info(f"[DDP] Retrying count: {retry}")
            time.sleep(3)
    log.info(f"[DDP] Found the host.")
    return dist_url


def get_dist_url(dist_url, num_machines):
    if dist_url == "local":
        assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
        port = _find_free_port()
        dist_url = f"tcp://127.0.0.1:{port}"
    elif dist_url == 'auto':
        ip = socket.gethostbyname(socket.gethostname())
        port = _find_free_port()
        dist_url = f"tcp://{ip}:{port}"
    else:
        dist_url = _maybe_gethostbyname(dist_url)
        
    if num_machines > 1 and dist_url.startswith("file://"):
        raise ValueError("Support dist_url that starts with 'tcp://' only")
    
    return dist_url


def scatter_values(dist_url, value):
    url = list(urlsplit(dist_url))[1]
    addr = url[:url.find(":")]
    port = int(url[url.find(":") + 1:]) + 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    def threaded(client_sock, i):
        client_sock.send(value.encode())
        client_sock.close()
    
    if comm.is_main_process():
        # server side
        sock.bind((addr, port))
        sock.listen()
        comm.synchronize()
        for i in range(comm.get_world_size() - 1):
            client_sock, addr = sock.accept()
            thread = threading.Thread(target=threaded, args=(client_sock, i))
            thread.daemon = True
            thread.start()
    else:
        # client side
        comm.synchronize()
        time.sleep(3)
        sock.connect((addr, port))
        value = sock.recv(1024).decode()
        
    sock.close()
    return value
