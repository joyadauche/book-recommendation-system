import numpy as np
import pandas as pd
import tensorflow.keras as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

books_df = pd.read_csv("books.csv")
ratings_df = pd.read_csv("ratings.csv")

print(books_df.head())
print(ratings_df.head())

unique_users = ratings_df.user_id.nunique()
unique_books = ratings_df.book_id.nunique()

train, val = train_test_split(ratings_df, test_size=0.2, random_state=1)
print(f"shape of train data: {train.shape}")
print(f"shape of test data: {val.shape}")

# define network
input_books = tf.layers.Input(shape=[1])
embed_books = tf.layers.Embedding(unique_books + 1, 10)(input_books)
books_out = tf.layers.Flatten()(embed_books)
input_users = tf.layers.Input(shape=[1])
embed_users = tf.layers.Embedding(unique_users + 1, 10)(input_users)
users_out = tf.layers.Flatten()(embed_users)

concatenate_layer = tf.layers.Concatenate()([books_out, users_out])
dense_layer = tf.layers.Dense(64, activation='relu')(concatenate_layer)
output = tf.layers.Dense(1, activation='relu')(dense_layer)

model = tf.Model([input_books, input_users], output)

optimizer = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

print(model.summary())

history = model.fit([train.book_id, train.user_id],
                    train.rating,
                    batch_size=64,
                    epochs=5,
                    verbose=1,
                    validation_data=([val.book_id, val.user_id], val.rating))

print(history.history)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()

# predict with for user 90
book_ids = np.array(list(ratings_df.book_id.unique()))
user = np.array([90 for i in range(len(book_ids))])
ratings = model.predict([book_ids, user])
print(ratings)
ratings = ratings.reshape(-1)
rec_book_ids = (-ratings).argsort()[0:5]
print(rec_book_ids)
print(books_df.iloc[rec_book_ids])
