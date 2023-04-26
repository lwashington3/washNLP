from .enumerations import *
from .models import *


__ALL__ = ["features_main", "embedding_main"]


def features_main(filename:str, model_folder:str = None, output_sequence_length=6, log_dir=None):
	from .tools import main_villain_type, get_vocab
	from datetime import datetime as dt
	import tensorflow as tf
	import numpy as np

	now = dt.now().strftime("%Y%m%d-%H%M%S")
	if model_folder is None:
		from time import time
		model_folder = f"models/features/villain_analysis_model_{now}"

	if log_dir is None:
		log_dir = f"logs/features/{now}"

	df = main_villain_type(filename)
	model = FeatureAnalysisModel(vec_inputs=output_sequence_length)
	model.create_library(get_vocab(df["Quote"]), output_mode="int", output_sequence_length=output_sequence_length)
	model.compile(loss=tf.keras.losses.MeanSquaredError(),
				  optimizer=tf.keras.optimizers.Adam(),
				  metrics=["accuracy"])

	results = [model.prepare(row["Quote"], row["VillainType"]) for _, row in df.iterrows()]
	features, labels = [], []
	for x, y in results:
		features.append(x)
		labels.append(y)

	features = np.array(features)
	labels = np.array(labels)

	callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

	history = model.fit(features, labels, epochs=10, verbose=2, callbacks=[callback])
	model.summary(line_length=200)
	model.save(model_folder)
	return model


def embedding_side(filename, model_folder=None, log_dir=None):
	from .tools import main_villain_type, main_villain_type_ohe, get_vocab
	from tensorflow.keras.callbacks import TensorBoard
	from tensorflow.keras.utils import to_categorical
	from datetime import datetime as dt
	import numpy as np

	now = dt.now().strftime("%Y%m%d-%H%M%S")
	if model_folder is None:
		from time import time
		model_folder = f"models/embedding/villain_analysis_model_{now}"

	if log_dir is None:
		log_dir = f"logs/embedding/{now}"
		# log_dir = f"logs/embedding/example"

	df = main_villain_type(filename)
	features = df["Quote"].to_numpy()
	labels = np.asarray([to_categorical(villain_type - 1, num_classes=len(VillainType), dtype="int") for villain_type in df["VillainType"]])

	vocab = get_vocab(df["Quote"])
	# model = EmbeddingAnalysisModel(len(vocab), 20, sequence_len=len(VillainType))

	vectorize = TextVectorization(
				max_tokens=len(vocab),
				output_mode="int",
				output_sequence_length=9,
				standardize=None,
				name="Text_Vectorizer"
			)
	vectorize.adapt(vocab)
	model = Sequential([
			vectorize,
			Embedding(len(vocab), 16, name="Embedding_Layer"),
			Dense(16, activation="relu", name="Hidden_Layer_1"),
			Dense(16 // 2, activation="sigmoid", name="Hidden_Layer_2"),
			Dense(len(VillainType), name="Output_Layer", activation="softmax"),
	])
	# model.adapt(vocab)
	model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
	model.fit(features, labels, epochs=15, callbacks=[tensorboard_callback])
	# model.log_values(log_dir)
	model.save(model_folder)


def embedding_main(filename, model_folder=None, log_dir=None):
	from .tools import main_villain_type, main_villain_type_ohe, get_vocab
	from tensorflow.keras.callbacks import TensorBoard
	from tensorflow.keras.utils import to_categorical
	from datetime import datetime as dt
	import numpy as np

	now = dt.now().strftime("%Y%m%d-%H%M%S")
	if model_folder is None:
		from time import time
		model_folder = f"models/embedding/villain_analysis_model_{now}"

	if log_dir is None:
		log_dir = f"logs/embedding/{now}"
		# log_dir = f"logs/embedding/example"

	df = main_villain_type(filename)
	features = df["Quote"].to_numpy()
	labels = np.asarray([to_categorical(villain_type - 1, num_classes=len(VillainType), dtype="int") for villain_type in df["VillainType"]])

	vocab = get_vocab(df["Quote"])
	model = EmbeddingAnalysisModel(len(vocab), 20, sequence_len=len(VillainType))
	model.adapt(vocab)
	model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
	model.fit(features, labels, epochs=15, callbacks=[tensorboard_callback])
	model.summary(line_length=200)
	model.log_values(log_dir, variable_value=now)
	model.save(model_folder)


if __name__ == '__main__':
	from sys import argv
	features_main(argv[-1])
	# embedding_main(argv[-1])
