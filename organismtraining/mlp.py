from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np
from keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
import matplotlib.pyplot as plt

def ngram_vectorize(c, train_texts, train_labels, val_texts, ngram_range=(1,1)):
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'stop_words': "english",
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': c['TOKEN_MODE'], 
            'min_df': c['MIN_DOCUMENT_FREQUENCY'],
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)

    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k=min(c['TOP_K'], x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32').todense()
    x_val = selector.transform(x_val).astype('float32').todense()
    return x_train, x_val, vectorizer.get_feature_names_out(), vectorizer

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model

def train_ngram_model(
    c,
    data,
    learning_rate=1e-4,
    epochs=100,
    batch_size=128,
    layers=2,
    units=64,
    dropout_rate=0.4,
    ngram_range=(1, 1),
):
    # Get the data.
    (
        (train_texts, train_labels),
        (val_texts, val_labels),
        (test_texts, test_labels),
    ) = data

    # convert all inputs to np array
    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    val_texts = np.array(val_texts)
    val_labels = np.array(val_labels)
    test_texts = np.array(test_texts)
    test_labels = np.array(test_labels)

    # Verify that validation labels are in the same range as training labels.
    num_classes = 2
    unexpected_labels = [v for v in val_labels if v not in range(2)]
    if len(unexpected_labels):
        raise ValueError(
            "Unexpected label values found in the validation set:"
            f" {unexpected_labels}. Please make sure that the "
            "labels in the validation set are in the same range "
            "as training labels."
        )

    # Vectorize texts.
    x_train, x_val, name_train, vectorizer = ngram_vectorize(
        c, train_texts, train_labels, val_texts, ngram_range
    )

    x_train, x_test, name_test, vectorizer = ngram_vectorize(
        c, train_texts, train_labels, test_texts, ngram_range
    )

    # Create model instance.
    model = mlp_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
    )

    # Compile model with learning parameters.
    loss = "binary_crossentropy"
    # optimizer = adam_v2.Adam(learning_rate=learning_rate)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callback = [callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    classifier = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callback,
        # validation_split=0.2,
        validation_data=(x_val, val_labels),
        verbose=2,
        batch_size=batch_size,
    )

    # Print results.
    history = classifier.history
    print(
        "Validation accuracy: {acc}, loss: {loss}".format(
            acc=history["val_acc"][-1], loss=history["val_loss"][-1]
        )
    )

    # Save model.
    # model.save('VIP_mlp_model.h5')

    print("\n# Evaluate on test data")
    results = model.evaluate(x_test, test_labels, batch_size=batch_size)
    print("test loss, test acc:", results)

    predicted_test = model.predict(x_test)

    return (
        model,
        predicted_test,
        test_labels,
        history,
        history["val_acc"][-1],
        history["val_loss"][-1],
        results[1],
        results[0],
        classifier,
    )
