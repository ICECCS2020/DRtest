import math
import numpy as np
from scipy import stats
import re

f=open('cifar_googlenet16_jsma_untarget.txt')
arr=[]
line_result=[]
line=f.readline()
if (line.find('[DEBUG][L1]')!=-1):
    words=line.split(',')
    scale=words[-2].split('=')
    if (scale[-1]==' -1'):
        arr.append((words[0].split('='))[-1].strip())
while line:
    line=f.readline()
    if (line.find('[DEBUG][L1]')!=-1):
        words=line.split(',')
        scale=words[-2].split('=')
        if (scale[-1]==' -1'):
            arr.append((words[0].split('='))[-1].strip())
            
    
    if (line.find('[STATS][L1]')!=-1):
        words=line.split(',')
        image_id=(words[-5].split('='))[-1].strip()
        #li_score=float((words[-1].split('='))[-1].strip())
        if image_id not in arr:
            line_result.append(line)
l1=[]
l2=[]
li=[]
for i in range(0,len(line_result)):
    print(line_result[i])
for i in range(0,len(line_result)):
    words=line_result[i].split(',')
    l1_score=float((words[-3].split('='))[-1].strip())
    l2_score=float((words[-2].split('='))[-1].strip())
    li_score=float((words[-1].split('='))[-1].strip())
    if l1_score>5.0:
        l1_score=5.0
    if l2_score>5.0:
        l2_score=5.0
    if li_score>5.0:
        li_score=5.0
    l1.append(l1_score)
    l2.append(l2_score)
    li.append(li_score)

l1_np=np.array(l1)
l2_np=np.array(l2)
li_np=np.array(li)

mean_l1=l1_np.mean()
mean_l2=l2_np.mean()
mean_li=li_np.mean()

std_l1=l1_np.std()
std_l2=l2_np.std()
std_li=li_np.std()

interval_l1=stats.t.interval(0.95,len(line_result)-1, mean_l1, std_l1)
interval_l2=stats.t.interval(0.95,len(line_result)-1, mean_l2, std_l2)
interval_li=stats.t.interval(0.95,len(line_result)-1, mean_li, std_li)

print('mean_l1: '+str(mean_l1))
print('interval: ')
print(interval_l1)
print('mean_l2: '+str(mean_l2))
print('interval: ')
print(interval_l2)
print('mean_li: '+str(mean_li))
print('interval: ')
print(interval_li)
f.close()



'''
yb=[]
for i in range(len(arr)-1):
    if float(arr[i])<5.0:
        yb.append(float(arr[i]))
    else:
        arr[i]=5.0
        yb.append(float(arr[i]))
 
yb_np=np.array(yb)
mean=yb_np.mean()
std=yb_np.std()

interval=stats.t.interval(0.95,len(yb)-1,mean,std)
print('mean: '+str(mean))
print('interval: ')
print(interval)
'''
