import os
import pickle
import time

from os.path import join
from collections import Counter, defaultdict
from nltk import word_tokenize

PATH_TO_DATA_RT = os.path.join("..", "crosstemporal_bias-master", "data", "reichstag")
REIGN_ORDER_RT = ["kaiserreich_1_processed", "kaiserreich_2_processed", "weimar_processed", "ns_processed"]

PATH_TO_DATA_BT = os.path.join("..", "crosstemporal_bias-master", "data", "bundestag")
REIGN_ORDER_BT = ['cdu_1', 'spd_1', 'cdu_2', 'spd_2', 'cdu_3']


# Takes about 30 minutes
def main():
    keywords = {
        "migrant": {"migrant", "migrantinnen", "migration", "immigrant", "fluechtlinge", "vertriebener", "zuwanderung",
                    "zuwanderer", "zustrom", "einwanderer", "einwanderung", "auslaender", "auslaenderin", "ansiedler",
                    "aussiedler", "asylsuchender", "asylbewerber"},
        #"frau": {"frau", "frauen", "fran"},
        "frau_ext": {"frau", "frauen", "fran", "maedchen", "mutter", "fraeulein", "genossin", "ehefrau"},
        "frau_nofrau": {"frauen", "maedchen", "mutter", "fraeulein", "genossin", "ehefrau"}
                }
    dict_keywords = defaultdict(list)
    # This will be freqdists for each legislative (or 4 year) period
    all_word_freqs, word_totals = [], []

    time_start, last_time = time.time(), time.time()

    # Reichstag
    for _period in REIGN_ORDER_RT:
        # e.g. /reichstag/kaisserreich_1_processed
        _period_path = os.path.join(PATH_TO_DATA_RT, _period)
        for four_year_period in os.listdir(_period_path):
            # e.g. /reichstag/kaiserreich_1_processed/1867-1871
            four_year_period_path = os.path.join(_period_path, four_year_period)
            # _cnter is used to remember keyword frequencies, _all_words for wordclouds
            _cnter, _all_words = Counter(), Counter()
            _word_total = 0
            for _file in os.listdir(four_year_period_path):
                # e.g. /reichstag/kaisserreich_1_processed/1867-1871/1_sents.txt
                with open(os.path.join(four_year_period_path, _file), encoding='utf-8') as _f:
                    for _line in _f:
                        for _word in word_tokenize(_line):
                            for _keyword in keywords:
                                if _word in keywords[_keyword]:
                                    _cnter[_keyword] += 1
                            _all_words[_word] += 1
                            _word_total += 1
                #print(f"Finished {_file} in {four_year_period}s")
            print(f"Finished {four_year_period} in {time.time() - last_time}s")
            last_time = time.time()

            for _keyword in keywords:
                dict_keywords[_keyword].append(_cnter[_keyword])
            word_totals.append(_word_total)
            all_word_freqs.append(_all_words)

    time_mid = time.time()
    print(f"Building frequency dictionary for Reichstag took {time_mid-time_start}s")

    # Bundestag
    for _reigning_party in REIGN_ORDER_BT:
        # e.g. /cdu_1
        _party_path = os.path.join(PATH_TO_DATA_BT, _reigning_party)
        for _reign_period in os.listdir(_party_path):
            # e.g. /cdu_1/slice_processed1
            # _cnter is used to remember keyword frequencies, _all_words for wordclouds
            _cnter, _all_words = Counter(), Counter()
            _word_total = 0
            _period_path = os.path.join(_party_path, _reign_period)
            for _file in os.listdir(_period_path):
                # e.g. /cdu_1/slice_processed1/1_sents.txt
                with open(os.path.join(_period_path, _file), encoding='utf-8') as _f:
                    for _line in _f:
                        for _word in word_tokenize(_line):
                            for _keyword in keywords:
                                if _word in keywords[_keyword]:
                                    _cnter[_keyword] += 1
                            _all_words[_word] += 1
                            _word_total += 1
                #print(f"Finished {_file} in {_reigning_party}")
            print(f"Finished {_reign_period} in {time.time() - last_time}s")
            last_time = time.time()

            for _keyword in keywords:
                dict_keywords[_keyword].append(_cnter[_keyword])
            word_totals.append(_word_total)
            all_word_freqs.append(_all_words)

    time_end = time.time()
    print(f"Building frequency dictionary for Bundestag took {time_end-time_mid}s")

    # save files
    with open(join("dicts", 'dicts/keyword_freq.pkl'), 'wb') as _f:
        pickle.dump(dict_keywords, _f)

    with open(join("dicts", 'dicts/word_totals.pkl'), 'wb') as _f:
        pickle.dump(word_totals, _f)

    with open(join("dicts", 'dicts/all_words_freq.pkl'), 'wb') as _f:
        pickle.dump(all_word_freqs, _f)


if __name__ == "__main__":
    main()
