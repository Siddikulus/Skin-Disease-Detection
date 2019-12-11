from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D, BatchNormalization, Conv2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Model
#
# base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Conv2D(1024, (4,4),activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Conv2D(1024,(3,3),activation='relu')(x) #dense layer 2
# # x=BatchNormalization()(x)
# x=Dense(512,activation='relu')(x) #dense layer 3
# preds=Dense(14,activation='softmax')(x) #final layer with softmax activation
#
# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
# train_generator=train_datagen.flow_from_directory('/home/sidharth/Python/ML Algos/Datasets/Dermnet Clean',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical', #If category more than 2. 'Binary' otherwise.
#                                                  shuffle=True
#                                                 )
#
# model=Model(inputs=base_model.input,outputs=preds)
# for i,layer in enumerate(model.layers):    #Structure of our model
#   print(i,layer.name)
#
# for layer in model.layers:                  #Do not train new weights to the model
#     layer.trainable=False
# # or if we want to set the first 20 layers of the network to be non-trainable
# # for layer in model.layers[:20]:
# #     layer.trainable=False
# # for layer in model.layers[20:]:
# #     layer.trainable=True
#
# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# # Adam optimizer
# # loss function will be categorical cross entropy
# # evaluation metric will be accuracy
#
# step_size_train=train_generator.n//train_generator.batch_size
# model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                     epochs = 10)
