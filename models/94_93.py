model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))


model.add(Dropout(0.4))
model.add(Flatten())    #mettre sous forme de vecteur
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
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
model.load_weights('model_weights.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=(len(x_train)/256), epochs=40, shuffle=True, validation_data=(x_test, y_test), callbacks=[reduce_lr, checkpointer])