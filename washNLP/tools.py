from .enumerations import *
import pandas as pd


__ALL__ = ["sentiment_based_features", "punctuation_syntax_features", "map_tag_category", "Pattern",
		   "resemblance_degree", "pattern_features", "villain_type", "main_villain_type"]


# region Bouazizi et Al "Sentiment Analysis: from Binary to Multi-Class Classification"
def sentiment_based_features(phrase:str) -> tuple[int, int]:
	"""
	Calculates the positive and negative strength of a phrase using the SentiStrength web tool as referenced by Bouazizi et al as feature family 1.
	:param str phrase:
	:return: The positive and negative strengths
	"""
	from requests import get
	from bs4 import BeautifulSoup
	from re import search

	# payload = {
	# 	"text": phrase,  # Tried having the text as a part of the payload, an error occurred every time.
	# 	"domain": "Film",
	# 	"submit": "Detect Sentiment in Domain",
	# }
	headers = {
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
		"Accept-Encoding": "gzip, deflate",
		"Accept-Language": "en-US,en;q=0.5",
		"Cache-Control": "max-age=0",
		"Connection": "keep-alive",
		"Host": "sentistrength.wlv.ac.uk",
		"Referer": "http://sentistrength.wlv.ac.uk/results.php?text=Good+morning%2C+you+filthy+animals.&submit=Detect+Sentiment&result=dual",
		"Upgrade-Insecure-Requests": "1",
		"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
	}

	response = get(f"http://sentistrength.wlv.ac.uk/results.php?text={phrase.replace(' ', '+')}&domain=Film&submit=Detect+Sentiment+in+Domain%27", headers=headers)
	if not response.ok:
		raise ValueError(f"The response code was not ok, content: {response.content}")

	soup = BeautifulSoup(response.text, "html.parser")
	summary = soup.select("span.ExecutiveSummary")[0]

	# match = search(fr"The text '{phrase}'\s*has positive strength (\d) and negative strength (-\d)", summary.text)
	match = search(fr"has positive strength (\d) and negative strength (-\d)", summary.text)
	if match is None:
		raise ValueError(f"The positive and negative strength results could not be found. `{summary.text}`")
	pw, nw = match.groups()
	return int(pw), int(nw)


def emotional_words_ratio(pw:int, nw:int) -> float:
	pw, nw = abs(pw), abs(nw)
	try:
		return (pw - nw) / (pw + nw)
	except ZeroDivisionError:
		return 0


def punctuation_syntax_features(phrase:str) -> dict:
	"""
	as referenced by Bouazizi et al as feature family 2.
	:param str phrase:
	:return:
	:rtype: dict
	"""
	from re import findall, IGNORECASE
	return {
		"num_exclamation_marks": phrase.count("!"),
		"num_question_marks": phrase.count("?"),
		"num_dots": phrase.count("."),
		"num_capital": len(findall("[A-Z]", phrase)),
		"num_quotes": len(findall(r"['\"]", phrase)),
		"repeated_vowels": bool(len(findall("a{3,}|e{3,}|i{3,}|o{3,}|u{3,}|y{3,}", phrase, IGNORECASE)))
	}


def map_tag_category(tag) -> POSTagCategories:
	"""Maps the given POS tag to the proper POS Tag Category given in Table III of Bouazizi et al"""
	match(tag):
		case "CC":
			return POSTagCategories.CI
		case "CD":
			return POSTagCategories.GFI
		case "DT":
			return POSTagCategories.CI
		case "EX":
			return POSTagCategories.CI
		case "FW":
			return POSTagCategories.GFI
		case "IN":
			return POSTagCategories.CI
		case "JJ":
			return POSTagCategories.EI
		case "JJR":
			return POSTagCategories.EI
		case "JJS":
			return POSTagCategories.EI
		case "LS":
			return POSTagCategories.GFI
		case "MD":
			return POSTagCategories.CI
		case "NN":
			return POSTagCategories.EI
		case "NNP":
			return POSTagCategories.GFI
		case "NNPS":
			return POSTagCategories.GFI
		case "NNS":
			return POSTagCategories.EI
		case "PDT":
			return POSTagCategories.CI
		case "POS":
			return POSTagCategories.CI
		case "PRP":
			return POSTagCategories.GFI
		case "PRP$":
			return POSTagCategories.GFI
		case "RB":
			return POSTagCategories.CI
		case "RBR":
			return POSTagCategories.CI
		case "RBS":
			return POSTagCategories.CI
		case "RP":
			return POSTagCategories.CI
		case "SYM":
			return POSTagCategories.GFI
		case "TO":
			return POSTagCategories.CI
		case "UH":
			return POSTagCategories.GFI
		case "VB":
			return POSTagCategories.EI
		case "VBD":
			return POSTagCategories.EI
		case "VBG":
			return POSTagCategories.EI
		case "VBN":
			return POSTagCategories.EI
		case "VBP":
			return POSTagCategories.EI
		case "VBZ":
			return POSTagCategories.EI
		case "WDT":
			return POSTagCategories.CI
		case "WP":
			return POSTagCategories.CI
		case "WP$":
			return POSTagCategories.CI
		case "WRB":
			return POSTagCategories.CI
		case _:
			from re import match
			if match("[,]", tag):
				return POSTagCategories.GFI	# These are symbols which I'm counting towards GFI like Table 3 does
			raise ValueError("The tag is not associated with a category in Table III of Bouazizi et al.")


class Pattern(object):
	def __init__(self, pattern, villain_type:VillainType=None, **kwargs):
		self._vector = pattern
		self.villain_type = villain_type

	@property
	def pattern_vector(self):
		return [i[-1] for i in self._vector]

	@property
	def villain_type(self) -> VillainType:
		return self._villain_type

	@villain_type.setter
	def villain_type(self, villain_type:VillainType):
		if isinstance(villain_type, int):
			villain_type = VillainType(villain_type)
		elif not isinstance(villain_type, VillainType | None):
			raise ValueError(f"The villain type must come from the VillainType enum, not {type(villain_type).__name__}")
		self._villain_type = villain_type

	def __len__(self):
		return len(self._vector) # TODO: Add a way to count the length of patterns in words

	@classmethod
	def get_tag_categories(cls, phrase:str, villain_type:VillainType = None, **kwargs):
		from nltk import word_tokenize, pos_tag
		from logging import warning

		text = word_tokenize(phrase)
		pattern = pos_tag(text)

		for i, tag in enumerate(pattern):
			try:
				mapped_pos = map_tag_category(tag[-1])
			except ValueError:
				mapped_pos = tag[-1]
				warning(f"The value \"{tag}\" could not be mapped using Table III of Bouazizi et al.")
			pattern[i] = (tag[0], tag[1], mapped_pos)
		return cls(pattern, villain_type, **kwargs)


def resemblance_degree(p:list[Pattern], t:Pattern, alpha=0.03) -> float:
	for pattern in p:
		if len(pattern) != len(t) or pattern.villain_type != t.villain_type:
			continue

		if pattern.pattern_vector == t.pattern_vector:
			return 1

		n = 0
		for p1, t1 in zip(pattern.pattern_vector, t.pattern_vector):
			if p1 == t1:
				n += 1

		return alpha * n / len(t)


def pattern_features(training_set:pd.DataFrame, num_occ=3, l_min=3, l_max=10, alpha=0.03, num_classes=10, K=None):
	from collections import defaultdict
	nl = l_max - l_min + 1
	patterns = defaultdict(int)
	if K is None:
		K = num_classes

	for phrase in training_set:
		pattern = Pattern.get_tag_categories(phrase["Quote"], phrase["Villain"])

		if not (l_min <= len(pattern) <= l_max):
			continue

		key = [pattern.villain_type]
		key.extend(pattern.pattern_vector)
		patterns[key] += 1

	patterns = [pattern for pattern, count in patterns.items() if count >= num_occ]
	nf = nl * num_classes
# endregion


def main_villain_type(input_file:str) -> pd.DataFrame:
	from numpy import argmax

	df = pd.read_csv(input_file)

	new_df = pd.DataFrame(columns=["Source", "Villain", "Quote", "Where", "VillainType"])
	for _, data in df.iterrows():
		row = data.filter(
			["Anti Villain Score", "Beasts Score", "Authority Figures Score", "Fanatics Score", "Machines Score",
			 "Personifications of Evil Score", "Masterminds Score", "Equals Score", "Corrupted Score", "Other Score"])
		villain = argmax(row) + 1
		series = pd.DataFrame({"Source": [data["Source"]], "Villain": [data["Villain"]], "Quote": [data["Quote"]],
		 								"Where": [data["Where"]], "VillainType": [VillainType(villain)]})
		new_df = pd.concat([new_df, series], ignore_index=True)

	return new_df


def main():
	# print(sentiment_based_features("I left early because the film was boring. My phone has a huge battery. There were mice in my room"))
	df = main_villain_type("sources.csv")
	print(df.head(len(df)))


if __name__ == '__main__':
	main()
