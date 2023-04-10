from enum import StrEnum, IntEnum


__ALL__ = ["VillainType", "POSTagCategories"]


class VillainType(IntEnum):
	Anti_Villain = 1
	Beasts = 2
	Authority_Figures = 3
	Fanatics = 4
	Machines = 5
	Personifications_of_Evil = 6
	Masterminds = 7
	Equals = 8
	Corrupted = 9
	Other = 10


class POSTagCategories(StrEnum):
	EI = "emotional"
	CI = "non-emotional"
	GFI = "grammatical-function"
