from utils import *
from sklearn.model_selection import train_test_split

# get Data
path="trainingData"
data=importData(path)

# print(data.size)

# balance data
balanceData(data)
# print(data.size)

# split data
imagepath,steering=splitData(path,data)
# print(imagepath[2],steering[0])

# split into training and testing

x_train,x_test,y_train,y_test=train_test_split(imagepath,steering,test_size=0.2 ,random_state=5)
# print(len(x_train))
# print(len(x_test))

# image augmentation
# img,steeering=ImageAug('test.jpg',steering[0])

#image preprocessing

#model creation
model=mymodel()
model.summary()

#training our model
modelhis=model.fit(batchGenerator(x_train,y_train,100,1),steps_per_epoch=200,epochs=5,validation_data=batchGenerator(x_test,y_test,100,0),validation_steps=100)

#save model
model.save("model.h5")

plt.plot(modelhis.history['loss'])
plt.plot(modelhis.history['val_loss'])
plt.legend(['training','validation'])

plt.ylim([0,1])
plt.title("loss plot")
plt.xlabel("epoch")
plt.show()
