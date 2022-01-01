import os

from gensim.models import KeyedVectors


def load_single_vector(_name):
    return KeyedVectors.load(os.path.join(PATH_TO_MODELS, _name), mmap="r")


PATH_TO_MODELS = os.path.join("..", "crosstemporal_bias-master", "models", "vectors", "anticommunism")

_model_names = ['kaiserreich_1', 'kaiserreich_2', 'weimar', 'cdu_1', 'spd_1', 'cdu_2',  'spd_2', 'cdu_3']

keywords = ["frau", "frauen", "immigrant", "migrant", "migrantinnen", "migration", "zuwanderer", "einwanderer",
            "auslaender", "auslaenderin", "ansiedler", "aussiedler", "asylsuchender", "asylbewerber", "fluechtling",
            "zuwanderung"]
wvs = [load_single_vector(_name) for _name in _model_names]

for _keyword in keywords:
    for _i, _wv in enumerate(wvs):
        try:
            print(f"{_keyword} in {_model_names[_i]}: {_wv.most_similar(_keyword)}")
        except KeyError:
            print(f"{_keyword} not in {_model_names[_i]}")
    print("")




