class rawJob:
	'''This is a class for interacting directly with the data provided
	by the competition csv file.

	Input is a parsed row from the csv module'''
	def __init__(self, parsedRow):
		values = [string.strip().lower() for string in parsedRow]
		categories = ["id", "title", "description", "rawLocation", "normalizedLocation",
						"contractType", "contractTime", "company", "category",
						"salaryRaw", "salaryNormalized","sourceName"]
		self.data = dict(zip(categories, values))
		
