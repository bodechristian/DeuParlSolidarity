import re
from datetime import datetime
import pickle

from os import listdir
from os.path import join
from collections import Counter, defaultdict
from nltk import word_tokenize

months = {"januar": 1, "februar": 2, "märz": 3, "april": 4, "mai": 5, "juni": 6, "july": 7, "august": 8, "september":9,
          "oktober": 10, "november": 11, "dezember": 12}
pattern = re.compile("^[a-zA-Zäöü]*\s\d\d\d\d$")
keywords = {
    "migrant": {"migrant", "migrantinnen", "migration", "immigrant", "fluechtlinge", "vertriebener", "zuwanderung",
                "zuwanderer", "zustrom", "einwanderer", "einwanderung", "auslaender", "auslaenderin", "ansiedler",
                "aussiedler", "asylsuchender", "asylbewerber"},
    "frau_nofrau": {"frauen", "maedchen", "mutter", "fraeulein", "genossin", "ehefrau"},
}

PATH_REICHSTAG = join("..", "reichstag_corpora")
PATH_BUNDESTAG = join("..", "BRD Protokolle")


def main():
    # file: date
    dic_dates = {}
    # date: number of sentences
    dic_lengths = defaultdict(int)
    # date: number of occurrences
    dic_women = defaultdict(int)
    # date: number of occurrences
    dic_migrants = defaultdict(int)

    # REICHSTAG

    last_valid_date = datetime(year=1760, month=1, day=1)
    for period in ["kaiserreich_1", "kaiserreich_2", "weimar", "ns"]:
        for file_name in sorted([el for el in listdir(join(PATH_REICHSTAG, period)) if el.endswith(".txt")],
                                key=get_num):
            c = Counter()
            line_counter = 0
            for line in open(join(PATH_REICHSTAG, period, file_name), encoding="utf-8"):
                line_counter += 1
                for match in re.finditer(pattern, line):
                    _date = match.group().split(" ")
                    if (_month := _date[0].lower()) in months:
                        _temp = datetime(year=int(_date[1]), month=1, day=1)
                        c[_temp] += 1

            # in case there were no dates or the date is not in the timespan, take date from the most recent file
            if c and 1865 < c.most_common(1)[0][0].year < 2022:
                dic_dates[join(period, file_name)] = c.most_common(1)[0][0]
                last_valid_date = c.most_common(1)[0][0]
            else:
                dic_dates[join(period, file_name)] = last_valid_date
            dic_lengths[last_valid_date.year] += line_counter

            # get number of occurrences of keywords
            for line in open(join(PATH_REICHSTAG, period, file_name), encoding="utf-8"):
                tokenized_line = word_tokenize(line)
                for keyword in keywords:
                    for word in tokenized_line:
                        if word.lower() in keywords[keyword]:
                            if keyword == "migrant":
                                dic_migrants[last_valid_date.year] += 1
                            else:
                                dic_women[last_valid_date.year] += 1
                            break

            print(f"{join(period, file_name)}\t{last_valid_date}")

    # BUNDESTAG
    for period in listdir(PATH_BUNDESTAG):
        for file_name in listdir(join(PATH_BUNDESTAG, period, f"{period}_Wahlperiode_TXT")):
            if not file_name.endswith(".txt"):
                continue
            # Get the date
            date = file_name[-14:-4]
            # some files have weird duplicated names, they have the format xxxxxxxxx(1).txt
            if date.endswith(")"):
                date = file_name[-17:-7]

            # starting at three, cause I decided to ignore days (to fine-grained over 150 years)
            val = datetime.strptime(date[6:], "%Y")
            # 19th period as different file naming for some reason
            if period == "19":
                key = join(f"{period}_Wahlperiode_TXT_sents", file_name)
            else:
                key = join(f"{period}_Wahlperiode_TXT_sents", f'{file_name[1:4].lstrip("0")}_sents.txt')

            dic_dates[key] = val
            with open(join(PATH_BUNDESTAG, period, f"{period}_Wahlperiode_TXT", file_name), "r", encoding="utf-8") as f:
                dic_lengths[val.year] += len(f.readlines())

            print(key, val)

            # get number of occurrences of keywords
            for line in open(join(PATH_BUNDESTAG, period, f"{period}_Wahlperiode_TXT", file_name), encoding="utf-8"):
                tokenized_line = word_tokenize(line)
                for keyword in keywords:
                    for word in tokenized_line:
                        if word.lower() in keywords[keyword]:
                            if keyword == "migrant":
                                dic_migrants[val.year] += 1
                            else:
                                dic_women[val.year] += 1
                            break

    with open(join("dicts", "dicts/dates.pkl"), "wb") as f:
        pickle.dump(dic_dates, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join("dicts", "dicts/lengths.pkl"), "wb") as f:
        pickle.dump(dic_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join("dicts", "dicts/occurrences_women.pkl"), "wb") as f:
        pickle.dump(dic_women, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join("dicts", "dicts/occurrences_migrants.pkl"), "wb") as f:
        pickle.dump(dic_migrants, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_num(_str):
    _res = ""
    for _char in _str:
        if _char.isdigit():
            _res += _char
        else:
            break
    return int(_res)


if __name__ == "__main__":
    main()
