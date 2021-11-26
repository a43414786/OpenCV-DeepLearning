from record import record
import numpy as np                
import matplotlib.pyplot as plt   

data = record()
data = data.data
epoch = []
trainloss = []
trainacc = []
testacc = []
for i in data:
    epoch.append(i['epoch'])
    trainloss.append(i['loss'])
    trainacc.append(i['train accuracy'])
    testacc.append(i['test accuracy'])


plt.figure(figsize = (6, 4.5), dpi = 100)                 # 設定圖片尺寸
plt.subplot(2, 1, 1) 
plt.title("Accuracy",fontsize = 16)
plt.xlabel('epoch', fontsize = 12)                        # 設定坐標軸標籤
plt.ylabel('%', fontsize = 12)
plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
plt.yticks(fontsize = 12)
plt.ylim(0, 110)                                          # 設定y軸繪圖範圍
line1, = plt.plot(epoch, trainacc, color = 'red', linewidth = 2, label = 'Training')             
line2, = plt.plot(epoch, testacc, color = 'blue', linewidth = 2, label = 'Testing')
plt.legend(handles = [line1, line2], loc='lower right')

plt.subplot(2, 1, 2) 
plt.title("Loss",fontsize = 16)
plt.xlabel('epoch', fontsize = 12)                        # 設定坐標軸標籤
plt.ylabel('loss', fontsize = 12)
plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
plt.yticks(fontsize = 12)
plt.ylim(-0.5, 3.0)                                          # 設定y軸繪圖範圍
plt.plot(epoch, trainloss, color = 'blue', linewidth = 2)

plt.tight_layout()

plt.savefig("trainplot.png")

plt.show()



plt.show()

# 儲存圖檔
'''
plt.savefig("test.jpg",   # 儲存圖檔
            bbox_inches='tight',               # 去除座標軸占用的空間
            pad_inches=0.0)                    # 去除所有白邊
#plt.close()      # 關閉圖表     ''' 