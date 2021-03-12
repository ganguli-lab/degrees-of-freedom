import pickle
from pathlib import Path

def save_obj(obj, name ):
	Path("../lottery-subspace-data/").mkdir(parents=True, exist_ok=True)
	with open('../lottery-subspace-data/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
	Path("../lottery-subspace-data/").mkdir(parents=True, exist_ok=True)
	with open('../lottery-subspace-data/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def sizeof_fmt(num, suffix='B'):
	"""
	Provides human-readable string for an integer size in Byles
	https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
	"""
	for unit in ['','K','M','G','T','P','E','Z']:
		if abs(num) < 1024.0:
			return "%3.1f%s%s" % (num, unit, suffix)
		num /= 1024.0
	return "%.1f%s%s" % (num, 'Yi', suffix)