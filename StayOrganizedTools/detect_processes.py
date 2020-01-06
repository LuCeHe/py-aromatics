import logging

import psutil
import nvidia_smi
import time
# FIXME: idea unfinished

logger = logging.getLogger('mylogger')

def test(): 
    listOfProcessNames = list()
    # Iterate over all running processes
    for proc in psutil.process_iter():
        # Get process detail as dictionary
        pInfoDict = proc.as_dict(attrs=['pid', 'name', 'cpu_percent'])
        # Append dict of process detail in list
        listOfProcessNames.append(pInfoDict)
        if 'python' in pInfoDict['name']:
            print(pInfoDict)
        
    
    
    for _ in range(10):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
        
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f'mem: {mem_res.used / (1024**2)} (GiB)') # usage in GiB
        print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%') # percentage usage
        
        thing = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle)
        
        print(thing[0].__dict__['pid'])
        print(type(thing[0]))
        pid = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle)[0]
        print(pid)
    

def wait_for_free_GPU(GPU):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(GPU)
    print('Waiting for GPU {} to be free...'.format(GPU))
    
    GPU_occupied = True
    while GPU_occupied:
        thing = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle)        
        try:
            pid_on_GPU = thing[0].__dict__['pid']
            # check every 30 mins
            time.sleep(1800)
            #time.sleep(3)            
        except:
            GPU_occupied = False
        print('Is GPU {} busy? {}'.format(GPU, GPU_occupied))

if __name__ == '__main__':
    wait_for_free_GPU(1)
    wait_for_free_GPU(0)

    
    
    