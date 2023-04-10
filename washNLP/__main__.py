from .enumerations import *
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.keras import Model


class VillainAnalysisModel(Model):
	def __init__(self):
		super(VillainAnalysisModel, self).__init__()
		from tensorflow.keras.layers import InputLayer, Dense

		self.input_layer = InputLayer(input_shape=(12,))
		self.hidden_layer1 = Dense(128, activation="relu", name="Hidden_Layer_1")
		self.hidden_layer2 = Dense(50, activation="sigmoid", name="Hidden_Layer_2")
		self.output_layer = Dense(len(VillainType), name="Output_Layer", activation="softmax")
		# Input layer layout will be: positive words score, negative words score, the 6 punctuation syntax features, and then the ratio of each POSTagCategory to the total number of words

	def call(self, x):
		x = self.input_layer(x)
		x = self.hidden_layer1(x)
		x = self.hidden_layer2(x)
		return self.output_layer(x)

	def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None, max_queue_size=10, workers=1,
        use_multiprocessing=False, get_probabilities=False, **kwargs):
		from numpy import array, argmax
		if isinstance(x, str):
			x = array([self.prepare(x)])
		else:
			x = array([self.prepare(quote) for quote in x])
		results = super().predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
								  max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
								  **kwargs)
		if get_probabilities:
			return results
		return (VillainType(argmax(i)+1) for i in results)

	@staticmethod
	def prepare(quote:str, villain_type:VillainType = None) -> tuple[ndarray, Tensor] | ndarray:
		from .tools import sentiment_based_features, punctuation_syntax_features, Pattern
		from numpy import array

		dct = punctuation_syntax_features(quote)
		pw, nw = sentiment_based_features(quote)
		dct["positive_score"] = pw
		dct["negative_score"] = nw

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

		array = array([
			dct['num_exclamation_marks'],
			dct['num_question_marks'],
			dct['num_dots'],
			dct['num_capital'],
			dct['num_quotes'],
			dct['repeated_vowels'],
			dct['positive_score'],
			dct['negative_score'],
			dct['EI'],
			dct['CI'],
			dct['GFI'],
			dct['unknown_cat']
		])

		if villain_type is None:
			return array

		from tensorflow.keras.utils import to_categorical
		villain_type = to_categorical(villain_type - 1, num_classes=len(VillainType), dtype="int")
		return (array, villain_type)


def main(filename:str):
	from .tools import main_villain_type
	import tensorflow as tf
	import numpy as np

	df = main_villain_type(filename)
	results = [VillainAnalysisModel.prepare(row["Quote"], row["VillainType"]) for _, row in df.iterrows()]
	features, labels = [], []
	for x, y in results:
		features.append(x)
		labels.append(y)

	features = np.array(features)
	labels = np.array(labels)
	model = VillainAnalysisModel()
	model.compile(loss=tf.keras.losses.MeanSquaredError(),
				  optimizer=tf.keras.optimizers.Adam())
	model.fit(features, labels, epochs=10, verbose=2)
	model.save("villain_analysis_model")
	return model


if __name__ == '__main__':
	from sys import argv
	main(argv[-1])
