from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

# 对样本进行预处理并画图
def plot_embedding(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    fig = plt.figure()		# 创建图形实例
    ax = plt.subplot(111)		# 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
    # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
            fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()		# 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    
    return fig

def visual(data, label, batch, sub, ppl):
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', random_state=0, perplexity=10)
    # t-SNE降维
    result = ts.fit_transform(data)
    # # 调用函数，绘制图像
    # fig = plot_embedding(result, label, 't-SNE Embedding of digits')
    # 自定义颜色列表
    colors = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿色
    '#17becf',  # 青色
    '#F433FF',  # 紫罗兰红
    '#000000',  # 黑色
    ]

    # 绘制每个标签的数据点
    # for i in range(60):
    #     plt.scatter(result[i, 0], result[i, 1], c=colors[i % 10], label=label[i])
    # for i, y in enumerate(np.unique(label)):
    #     plt.scatter(result[y == label, 0], result[y == label, 1], c=colors[i % len(colors)], label=label)
    # 用于跟踪已经添加到图例中的标签
    added_to_legend = set()
    # 创建一个图形和两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # 在第一个子图中绘制数据
    # for i, y in enumerate(np.unique(label)):
    #     mask = (y == label)
    #     ax1.scatter(result[mask, 0], result[mask, 1], c=colors[i], label=f'Label {y}')

    
    # 绘制散点图
    # for i, y in enumerate(np.unique(label)):
    #     if y not in added_to_legend:
    #         plt.scatter(result[y == label, 0], result[y == label, 1], 
    #                     c=colors[i], label=f'Label {y}')
    #         added_to_legend.add(y)
    #     else:
    #         plt.scatter(result[y == label, 0], result[y == label, 1], 
    #                     c=colors[i])
    for i, y in enumerate(np.unique(label)):
        mask = (y == label)
        sub_mask_0 = mask & (sub == 0)
        sub_mask_1 = mask & (sub == 1)
        
        # 如果标签还没有添加到图例中，则添加图例
        if y not in added_to_legend:
            ax1.scatter(result[sub_mask_0, 0], result[sub_mask_0, 1], 
                        c=colors[i], marker='o', label=f'Label {y}')
            ax1.scatter(result[sub_mask_1, 0], result[sub_mask_1, 1], 
                        c=colors[i], marker='x')
            added_to_legend.add(y)
        else:
            ax1.scatter(result[sub_mask_0, 0], result[sub_mask_0, 1], 
                        c=colors[i], marker='o')
            ax1.scatter(result[sub_mask_1, 0], result[sub_mask_1, 1], 
                        c=colors[i], marker='x')

    # 在第二个子图中显示图例
    ax2.legend(*ax1.get_legend_handles_labels(), loc='center')
    ax2.axis('off')  # 关闭第二个子图的坐标轴显示
    # # 添加图例
    # plt.legend()

    # 显示图表
    # # 设置X轴的刻度间距为2
    # x_ticks = np.arange(0, 11, 2)
    # ax1.xticks(x_ticks)

    # # 设置Y轴的刻度间距为0.5
    # y_ticks = np.arange(-1, 1.1, 0.5)
    # ax1.yticks(y_ticks)
    ax1.set_title('t-SNE Embedding of digits')
    ax1.tick_params(axis='both', which='both', 
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    # 在显示图像之前保存它
    path = 'resource\\t-sne\\' + 'ours_p50'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path +'\\t-SNE_Embedding_batch_'+ str(batch) +'.png', format='png')

    # 显示图像
    plt.show()


# import tkinter as tk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure

# # 创建一个新窗口
# root = tk.Tk()
# root.title("Save Plot Example")

# # 创建一个matplotlib图形
# fig = Figure(figsize=(5, 4), dpi=100)
# plot = fig.add_subplot(111)

# # 假设这里是你的绘图代码
# # plot.plot([1, 2, 3], [4, 5, 6])

# # 将matplotlib图形嵌入到Tkinter窗口中
# canvas = FigureCanvasTkAgg(fig, master=root)
# canvas.draw()
# canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# # 创建一个按钮，点击时保存图像
# def save_plot():
#     fig.savefig('saved_plot.png', format='png')
#     tk.messagebox.showinfo("Saved", "Plot saved as saved_plot.png")

# save_button = tk.Button(master=root, text="Save Plot", command=save_plot)
# save_button.pack(side=tk.BOTTOM)

# # 运行Tkinter事件循环
# root.mainloop()
'''
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:55:08 2021

@author: 1
"""
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.optimizers import SGD
from keras.layers import Dense,LSTM, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D,BatchNormalization
from keras.models import load_model
from sklearn import preprocessing
# 载入数据
df = pd.read_csv(r'C:/Users/1/Desktop/14改.csv')
X = np.expand_dims(df.values[:, 0:1024].astype(float), axis=2)
Y = df.values[:, 1024]
 
# 湿度分类编码为数字

 
# 划分训练集，测试集
X_train, X_test, K, y = train_test_split(X, Y, test_size=0.3, random_state=0)
K=K

encoder = LabelEncoder()
Y_encoded1 = encoder.fit_transform(K)
Y_train = np_utils.to_categorical(Y_encoded1)

Y_encoded2 = encoder.fit_transform(y)
Y_test = np_utils.to_categorical(Y_encoded2)

# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Convolution1D(16, 64,strides=16,padding='same', input_shape=(1024, 1),activation='relu'))#第一个卷积层
    model.add(MaxPooling1D(2,strides=2,padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    

    model.add(Convolution1D(32,3,padding='same',activation='relu'))
    model.add(MaxPooling1D(2,strides=2,padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    
    model.add(Convolution1D(64,3,padding='same',activation='relu'))#第二个卷积层
    model.add(MaxPooling1D(2,strides=2,padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    
    model.add(Convolution1D(64, 3,padding='same',activation='relu'))#第三个卷积层
    model.add(MaxPooling1D(2,strides=2,padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    
    model.add(Convolution1D(64, 3,padding='same',activation='relu'))#第四个卷积层
    model.add(MaxPooling1D(2,strides=2,padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    
    model.add(Convolution1D(64,3,padding='same',activation='relu'))#第五个卷积层
    model.add(MaxPooling1D(2,strides=2,padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))


    model.add(Dense(100,activation='relu'))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
 
# 训练分类器
estimator = KerasClassifier(build_fn=baseline_model, epochs=3000, batch_size=128, verbose=1)
history=estimator.fit(X_train, Y_train, validation_data=(X_test, Y_test))
import matplotlib.pyplot as plt

# 卷积网络可视化
def visual(model, data, num_layer=1):
     layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
     f1 = layer([data])[0]
     np.set_printoptions(threshold=np.inf)
     print(f1.shape)
     print(f1)
     f2=f1.reshape(6034,64)
     print(f2)
     num = f1.shape[-1]
     print(num)
     plt.figure(figsize=(6, 12), dpi=150)
     for i in range(num):
         plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
         plt.imshow(f1[:, :, i] * 255, cmap='prism')
         plt.axis('off')
     plt.show()
     def get_data():
	
	#digits = datasets.load_digits(n_class=10)
         digits=2
         data = f2#digits.data		# 图片特征
         label = K#digits.target		# 图片标签
         n_samples=6034
         n_features =64 #data.shape		# 数据集的形状
         return data, label, n_samples, n_features


# 对样本进行预处理并画图
     def plot_embedding(data, label, title):

         x_min, x_max = np.min(data, 0), np.max(data, 0)
         data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
         fig = plt.figure()		# 创建图形实例
         ax = plt.subplot(111)		# 创建子图
	# 遍历所有样本
         for i in range(data.shape[0]):
		# 在图中为每个数据点画出标签
             plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
				 fontdict={'weight': 'bold', 'size': 7})
         plt.xticks()		# 指定坐标的刻度
         plt.yticks()
         plt.title(title, fontsize=14)
	# 返回值
         return fig



     data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
     print('Starting compute t-SNE Embedding...')
     ts = TSNE(n_components=2, init='pca', random_state=0)
	# t-SNE降维
     reslut = ts.fit_transform(data)
	# 调用函数，绘制图像
     fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
	# 显示图像
     plt.show()
    


# 可视化卷积层
visual(estimator.model, X_train, 20)#在这里插入代码片



'''