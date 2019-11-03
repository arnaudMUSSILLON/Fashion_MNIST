model = Sequential()

def conv_layer(model, filter):
  model.add(Conv2D(filters=filter, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28,28,1)))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=filter, kernel_size=(3, 3), activation='relu', padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=filter, kernel_size=(3, 3), activation='relu', padding='same'))
  model.add(BatchNormalization())
  model.add(AveragePooling2D(pool_size=(2, 2)))
  model.add(SpatialDropout2D(0.3))
  return model

model = conv_layer(model, 32)
model = conv_layer(model, 64)
model = conv_layer(model, 64)
model = conv_layer(model, 64)

model.add(Flatten())    #mettre sous forme de vecteur
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

checkpointer = ModelCheckpoint(filepath='model_weights.h5', verbose=1, save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3, min_lr=0.000001)

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)

train_datagen = ImageDataGenerator(horizontal_flip=True)
train_generator = train_datagen.flow(x_train, y_train, batch_size=256)

model.summary()
#model.load_weights('model_weights2.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=(len(x_train)/256), epochs=(50), shuffle=True, validation_data=(x_test, y_test), callbacks=[reduce_lr, checkpointer])