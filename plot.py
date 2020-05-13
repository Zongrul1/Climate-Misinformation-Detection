import matplotlib.pyplot as plt

x1 = range(0, 10)
x2 = range(0, 10)
y1 = [0.4766,
0.1265,
0.0297,
0.0127,
0.0081,
0.0056,
0.0037,
0.003,
0.0026,
0.0029,
]
y2 = [0.3684,
0.2477,
0.3117,
0.2092,
0.1947,
0.2286,
0.2028,
0.2858,
0.2064,
0.2055,
]
y3 = [0.8424,
0.9883,
0.996,
0.9979,
0.9986,
0.9991,
0.9991,
0.9996,
0.9993,
0.9993,
]
y4 = [0.85,
0.89,
0.85,
0.91,
0.92,
0.91,
0.91,
0.9,
0.92,
0.92,
]
xx1 = ['Baseline','ClassSVM','LR','FFNN','textCNN']
yy1 = [0.53,0.72,0.84,0.91,0.80]
yy2 = [0.50,0.47,0.72,0.79,0.69]
plt.plot(xx1, yy1, 'o-',color='r')
plt.plot(xx1, yy1, 'o-',label="F1-Score in Dev Data")
plt.plot(xx1, yy2, 'o-',label="F1-Score in Test Data")
plt.title('Model vs. F1-Score')
plt.ylabel('F1-Score')
plt.legend(loc='best')
plt.show()
#plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-',color='r')
# plt.plot(x1, y3, 'o-',label="Train_Accuracy")
# plt.plot(x1, y4, 'o-',label="Valid_Accuracy")
# plt.title('Accuracy vs. epoches')
# plt.ylabel('Accuracy')
# plt.legend(loc='best')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y1, '.-',label="Train_Loss")
# plt.plot(x2, y2, '.-',label="Valid_Loss")
# plt.xlabel('Loss vs. epoches')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()