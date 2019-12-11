from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, BatchNormalization
from keras import Sequential
from keras.preprocessing import image
from keras.models import Model

base_model = Xception(weights = 'imagenet', include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
pred = Dense(5749, activation='softmax')(x)

model = Model(inputs = base_model.input,outputs = pred)
for layer in model.layers:
    print(layer.name)

train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('/home/sidharth/Python/ML Algos/Datasets/lfw',
                                                    target_size = (299,299),
                                                    color_mode = 'rgb',
                                                    batch_size = 8,
                                                    class_mode = 'categorical',
                                                    shuffle = True
                                                    )
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10)