# -*- coding: cp1251 -*-
from ftplib import FTP
import sys

PATH = '\\'.join(sys.argv[0].split('\\')[:-1])

ftp = FTP()
#HOST = '192.168.137.46'
HOST = '46.146.211.38'
PORT = 56721

ftp.connect(HOST, PORT)

print(ftp.login(user='alexandr', passwd='9'))
#print(ftp.login(user='pi', passwd='robo9'))

ftp.cwd('/home/alexandr/wildhack/WildHack/yoloact/output_images')
#ftp.cwd('/home/alexandr/wildhack/WildHack/yoloact/weights')
#ftp.cwd('out_stepa_videos')

for i in ['0000.png',
          '0001.png',
          '0002.png',
          '0003.png',
          '0004.png',
          '0005.png',
          '0006.png',
          '0007.png',
          '0008.png',
          '0009.png',]:
#if True:
  #fl = 'test_9.wav'
  fl = i
  out = fl

  with open(out, 'wb') as f:
      ftp.retrbinary('RETR ' + fl, f.write)
