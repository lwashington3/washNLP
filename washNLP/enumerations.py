from enum import StrEnum, IntEnum, auto


__ALL__ = ["VillainType", "POSTagCategories"]


class VillainType(IntEnum):
	Anti_Villain = auto()
	Authority_Figures = auto()
	Fanatics = auto()
	Machines = auto()
	Personifications_of_Evil = auto()
	Masterminds = auto()
	Equals = auto()
	Corrupted = auto()
	Other = auto()


class POSTagCategories(StrEnum):
	EI = "emotional"
	CI = "non-emotional"
	GFI = "grammatical-function"
