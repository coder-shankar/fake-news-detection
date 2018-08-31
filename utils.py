import sklearn.metrics as sm
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(y_test,pred):
  """
  Confusion matrix for two lables true and false
  :aram y_test: true label
  :aram pred: predicated lable from algorithm
  """
  cm = sm.confusion_matrix(y_test,pred)
  plt.clf()
  plt.imshow(cm,interpolation = 'nearest',  cmap=plt.cm.Blues)
  ClassNames = ['fake','true']
  plt.ylabel('expected')
  plt.xlabel('predicted')
  tick_marks = np.arange(len(ClassNames))
  plt.xticks(tick_marks,ClassNames,rotation = 45)
  plt.yticks(tick_marks,ClassNames)
  plt.imshow(cm,interpolation = 'nearest',  cmap=plt.cm.Blues)
  s = [['TN','FP'],['FN','TP']]
  for i in range(2):
      for j in range(2):
          plt.text(j,i,str(s[i][j]) +" = " +str(cm[i][j]))
  plt.show()