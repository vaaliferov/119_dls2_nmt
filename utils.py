import os
import psutil
import zipfile
import subprocess

def zip_dir(path, exclude, zip_file_path):
    zf = zipfile.ZipFile(zip_file_path, mode='w')
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        for filename in files: zf.write(f'{root}/{filename}')
    zf.close()

def get_sys_info():
    d = psutil.disk_usage('/')
    m = psutil.virtual_memory()
    memory = f'{m.used >> 20} / {m.total >> 20} MiB'
    storage = f'{d.used >> 30} / {d.total >> 30} GiB'
    return f'memory: {memory}, storage: {storage}'

def get_gpu_info():
    try:
        params = ','.join(['name','memory.used','memory.total'])
        cmd = ['nvidia-smi','--format=csv',f'--query-gpu={params}']
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        if result.returncode != 0: return 'no gpu'
        output = result.stdout.decode('utf-8').split('\n')[1:-1]
        to_str = lambda toks: f'{toks[0]}: {toks[1]} / {toks[2]}'
        return ', '.join([to_str(line.split(', ')) for line in output])
    except OSError:
        return 'no gpu'