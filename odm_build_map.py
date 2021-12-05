import os
from pyodm import Node

n = Node('192.168.68.202', 3000)


path = 'dataset/p2'
arr = []
for f in os.listdir(path):
  name = '{}/{}'.format(path, f)
  print(name)
  arr.append(name)
  if len(arr) == 150:
    break

task = n.create_task(arr,
                     {'fast-orthophoto' : True,
                      'orthophoto-png' : True,
                       'dsm': False,
                      'skip-3dmodel' : True,
                      'skip-band-alignment' : True,
                      'texturing-skip-local-seam-leveling' : True,
                      'texturing-skip-global-seam-leveling' : True,
                      'matcher-neighbors' : 2,
                      'orthophoto-resolution' : 1
                      })


task.wait_for_completion()
task.download_assets("./results")




