__ALL__ = ["FeatureAnalysisModel", "EmbeddingAnalysisModel"]


from .enumerations import *
from numpy import ndarray
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Input, InputLayer, TextVectorization


class WashNLP(Sequential):
	def __init__(self, name=None):
		super(WashNLP, self).__init__(name=name)

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


class FeatureAnalysisModel(WashNLP):
	"""
	A model based on scanning features from Bouazizi et Al
	"""
	def __init__(self, encoder_location:str=None, vec_inputs:int=None, name="Feature_Analysis_Model"):
		super(FeatureAnalysisModel, self).__init__()

		self.text_vec = encoder_location
		if self.text_vec is None and vec_inputs is None:
			raise ValueError("If a TextVectorization model is not provided, you must provide the number of vectors the trained class will provide.")
		elif self.text_vec is None:
			self.vec_inputs = vec_inputs

		self.input_layer = InputLayer(input_shape=(12 + self.vec_inputs,))
		self.hidden_layer1 = Dense(128, activation="relu", name="Hidden_Layer_1")
		self.hidden_layer2 = Dense(50, activation="sigmoid", name="Hidden_Layer_2")
		self.output_layer = Dense(len(VillainType), name="Output_Layer", activation="softmax")

		self.add(self.input_layer)
		self.add(self.hidden_layer1)
		self.add(self.hidden_layer2)
		self.add(self.output_layer)
		# Input layer layout will be: positive words score, negative words score, the 6 punctuation syntax features, and then the ratio of each POSTagCategory to the total number of words

	@property
	def vec_inputs(self) -> int:
		return self._vec_inputs

	@vec_inputs.setter
	def vec_inputs(self, value:int | None):
		self._vec_inputs = value

	@property
	def text_vec(self) -> Model | None:
		return self._text_vectorization

	@text_vec.setter
	def text_vec(self, layer:Model | str | None):
		if isinstance(layer, str | Model):
			if isinstance(layer, str):
				from keras.models import load_model
				layer = load_model(layer)

			self._text_vectorization = layer
			self.vec_inputs = layer.output.shape[1]
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
		output_sequence_length = kwargs.get("output_sequence_length", self.vec_inputs)

		if self.vec_inputs is not None:
			if self.vec_inputs != output_sequence_length:
				raise ValueError("The output_sequence_length of this library does not match what the model expects.")

		model = Sequential()

		vectorize = TextVectorization(**kwargs)
		vectorize.adapt([str(i) for i in dataset])

		model.add(Input(shape=(1,), dtype=tf.string))
		model.add(vectorize)

		self.text_vec = model

	# def call(self, x):
	# 	x = self.input_layer(x)
	# 	x = self.hidden_layer1(x)
	# 	x = self.hidden_layer2(x)
	# 	return self.output_layer(x)

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


class EmbeddingAnalysisModel(WashNLP):
	"""
	A model based on scanning features from Bouazizi et Al
	"""
	def __init__(self, vocab_len:int, output_dim:int = 16, sequence_len:int = 10, standardize=None, name="Embedding_Analysis_Model", **kwargs):
		super().__init__(name=name)

		self.vectorize = TextVectorization(
			max_tokens=vocab_len,
			output_mode="int",
			output_sequence_length=sequence_len,
			standardize=standardize,
			name="Text_Vectorizer"
		)
		self.embedding = Embedding(vocab_len, output_dim, name="Embedding_Layer", **kwargs)
		self.hidden_layer1 = Dense(output_dim, activation="relu", name="Hidden_Layer_1")
		self.hidden_layer2 = Dense(output_dim//2, activation="sigmoid", name="Hidden_Layer_2")
		self.output_layer = Dense(len(VillainType), name="Output_Layer", activation="softmax")

		self.add(self.vectorize)
		self.add(self.embedding)
		self.add(self.hidden_layer1)
		self.add(self.hidden_layer2)
		self.add(self.output_layer)

	# def call(self, x):
	# 	x = self.vectorize(x)
	# 	x = self.embedding(x)
	# 	x = self.hidden_layer1(x)
	# 	x = self.hidden_layer2(x)
	# 	return self.output_layer(x)

	def adapt(self, data, batch_size=None, steps=None):
		self.vectorize.adapt(data, batch_size, steps)

	def log_values(self, log_dir, variable_value="VARIABLE_VALUE"):
		from tensorflow import Variable
		from tensorflow.train import Checkpoint
		from tensorboard.plugins import projector
		from os import makedirs
		from os.path import join, exists

		vocab = self.vectorize.get_vocabulary()
		weights = self.embedding.get_weights()[0]
		with open(join(log_dir, "metadata.tsv"), 'w', encoding="utf-8") as m, open(join(log_dir, "vectors.tsv"), 'w', encoding="utf-8") as v:
			for index, word in enumerate(vocab):
				if index < 1:
					continue
				vec = weights[index-2]
				v.write("\t".join([str(x) for x in vec]) + "\n")
				m.write(f"{word}\n")

		weights = Variable(weights[1:])
		checkpoint = Checkpoint(embedding=weights)
		checkpoint.save(join(log_dir, "embedding.ckpt"))

		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = f"embedding/.ATTRIBUTES/{variable_value}"
		embedding.metadata_path = join(log_dir, "metadata.tsv")
		projector.visualize_embeddings(log_dir, config)
