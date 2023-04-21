from .enumerations import *
from numpy import ndarray
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer, Dense, TextVectorization

# Create a separate model containing the encoder from https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization#compile
# This model wouldn't need to be re-trained since the vectors wouldn't change from weights, so it can stay the same, and when a model is trained, the encoder
# can be saved and loaded as its own model. Then, in the model, it's called like predict model to create the vector and is passed to the actual VillainAnalysisModel
# InputLayer


__ALL__ = ["VillainAnalysisModel", "main"]


class VillainAnalysisModel(Model):
	def __init__(self, encoder_location:str=None):
		super(VillainAnalysisModel, self).__init__()
		from tensorflow.keras.layers import InputLayer, Dense

		self.text_vec = encoder_location
		self.input_layer = InputLayer(input_shape=(18,))
		self.hidden_layer1 = Dense(128, activation="relu", name="Hidden_Layer_1")
		self.hidden_layer2 = Dense(50, activation="sigmoid", name="Hidden_Layer_2")
		self.output_layer = Dense(len(VillainType), name="Output_Layer", activation="softmax")
		# Input layer layout will be: positive words score, negative words score, the 6 punctuation syntax features, and then the ratio of each POSTagCategory to the total number of words

	@property
	def text_vec(self) -> Model | None:
		return self._text_vectorization

	@text_vec.setter
	def text_vec(self, layer:Model | str | None):
		if isinstance(layer, Model):
			self._text_vectorization = layer
		elif isinstance(layer, str):
			from keras.models import load_model
			self._text_vectorization = load_model(layer)
		else:
			self._text_vectorization = None

	def create_library(self, dataset, **kwargs) -> Sequential:
		"""

		:param dataset: A list of all possible strings.
		:param kwargs:
		:type kwargs:
		:return:
		:rtype:
		"""
		model = Sequential()

		vectorize = TextVectorization(**kwargs)
		vectorize.adapt([str(i) for i in dataset])

		model.add(Input(shape=(1,), dtype=tf.string))
		model.add(vectorize)

		self.text_vec = model

	def call(self, x):
		x = self.input_layer(x)
		x = self.hidden_layer1(x)
		x = self.hidden_layer2(x)
		return self.output_layer(x)

	def predict(self, *x, batch_size=None, verbose="auto", steps=None, callbacks=None, max_queue_size=10, workers=1,
        use_multiprocessing=False, **kwargs) -> tuple[VillainType, ndarray]:
		from numpy import array as np_array, argmax
		if isinstance(x, str):
			x = np_array([self.prepare(x)])
		else:
			x = np_array([self.prepare(quote) for quote in x])
		results = super().predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
								  max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
								  **kwargs)
		return tuple([(VillainType(argmax(i)+1),i) for i in results])

	def compile(self,**kwargs):
		if self.text_vec is None:
			raise ValueError("The text vectorization layer cannot be None. A trained layer or save location must be given to `__init__`, or `create_library()` must be called to train the layer.")
		super().compile(**kwargs)

	def prepare(self, quote:str, villain_type:VillainType = None) -> tuple[ndarray, Tensor] | ndarray:
		from .tools import sentiment_based_features, punctuation_syntax_features, Pattern
		from numpy import asarray as np_array

		if self.text_vec is None:
			raise ValueError(
				"The text vectorization layer cannot be None. A trained layer or save location must be given to `__init__`, or `create_library()` must be called to train the layer.")

		dct = punctuation_syntax_features(quote)
		# pw, nw = sentiment_based_features(quote)
		# dct["positive_score"] = pw
		# dct["negative_score"] = nw

		dct["vectorized"] = self.text_vec.predict([quote])

		pattern = Pattern.get_tag_categories(quote, villain_type)
		for category in POSTagCategories:
			dct[category.name] = 0
		dct["unknown_cat"] = 0

		for category in pattern.pattern_vector:
			if not isinstance(category, POSTagCategories):
				dct["unknown_cat"] += 1
			else:
				dct[category.name] += 1

		for category in POSTagCategories:
			dct[category.name] /= len(pattern)
		dct["unknown_cat"] /= len(pattern)

		array = [
			*dct['vectorized'][0],
			dct['num_exclamation_marks'],
			dct['num_question_marks'],
			dct['num_dots'],
			dct['num_pauses'],
			dct['num_responses'],
			dct['num_capital'],
			dct['num_quotes'],
			dct['repeated_vowels'],
			# dct['positive_score'],
			# dct['negative_score'],
			dct['EI'],
			dct['CI'],
			dct['GFI'],
			dct['unknown_cat']
		]

		if villain_type is None:
			return array

		from tensorflow.keras.utils import to_categorical
		villain_type = to_categorical(villain_type - 1, num_classes=len(VillainType), dtype="int")
		return (array, villain_type)

	@staticmethod
	def get_datasets(filename:str):
		from .tools import main_villain_type
		import tensorflow as tf
		import numpy as np
		from tensorflow.keras.utils import to_categorical

		df = main_villain_type(filename)
		features, labels = df.pop("Quote"), df.pop("VillainType")
		# features = np.array([tf.constant(i) for i in features])
		# features = np.array(features, dtype=np.string_)
		labels = np.array(labels, dtype=np.int16)

		dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

		# villain_type = to_categorical(villain_type - 1, num_classes=len(VillainType), dtype="int")

	@classmethod
	def load(cls, folder_name:str):
		from keras.models import load_model
		return load_model(folder_name, custom_objects={cls.__name__: cls})


def main(filename:str, model_folder:str=None):
	from .tools import main_villain_type, get_vocab
	import tensorflow as tf
	import numpy as np

	if model_folder is None:
		from time import time
		model_folder = f"../models/villain_analysis_model_{time()}"

	df = main_villain_type(filename)
	model = VillainAnalysisModel()
	model.create_library(get_vocab(df["Quote"]), output_mode="int", output_sequence_length=6)
	model.compile(loss=tf.keras.losses.MeanSquaredError(),
				  optimizer=tf.keras.optimizers.Adam(),
				  metrics=tf.keras.metrics.CategoricalAccuracy())

	results = [model.prepare(row["Quote"], row["VillainType"]) for _, row in df.iterrows()]
	features, labels = [], []
	for x, y in results:
		features.append(x)
		labels.append(y)

	features = np.array(features)
	labels = np.array(labels)

	model.fit(features, labels, epochs=10, verbose=2)
	model.save(model_folder)
	return model


if __name__ == '__main__':
	from sys import argv
	main(argv[-1])
