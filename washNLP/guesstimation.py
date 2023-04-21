from .enumerations import VillainType


def guesstimate(quote:str, villain_type:VillainType=None) -> float:
	pass


# New plan, use clustering with one known quote as the starting point and let DBScan sort them into one of the categories and anything unclustered will be others.
