"""
script that receives a word and returns all the combinations of periodic table elements that can be combined to spell
that word, if any exist
"""
from typing import *
import fire

elements = [
    'H',
    'He',
    'Li',
    'Be',
    'B',
    'C',
    'N',
    'O',
    'F',
    'Ne',
    'Na',
    'Mg',
    'Al',
    'Si',
    'P',
    'S',
    'Cl',
    'Ar',
    'K',
    'Ca',
    'Sc',
    'Ti',
    'V',
    'Cr',
    'Mn',
    'Fe',
    'Co',
    'Ni',
    'Cu',
    'Zn',
    'Ga',
    'Ge',
    'As',
    'Se',
    'Br',
    'Kr',
    'Rb',
    'Sr',
    'Y',
    'Zr',
    'Nb',
    'Mo',
    'Tc',
    'Ru',
    'Rh',
    'Pd',
    'Ag',
    'Cd',
    'In',
    'Sn',
    'Sb',
    'Te',
    'I',
    'Xe',
    'Cs',
    'Ba',
    'La',
    'Ce',
    'Pr',
    'Nd',
    'Pm',
    'Sm',
    'Eu',
    'Gd',
    'Tb',
    'Dy',
    'Ho',
    'Er',
    'Tm',
    'Yb',
    'Lu',
    'Hf',
    'Ta',
    'W',
    'Re',
    'Os',
    'Ir',
    'Pt',
    'Au',
    'Hg',
    'Tl',
    'Pb',
    'Bi',
    'Po',
    'At',
    'Rn',
    'Fr',
    'Ra',
    'Ac',
    'Th',
    'Pa',
    'U',
    'Np',
    'Pu',
    'Am',
    'Cm',
    'Bk',
    'Cf',
    'Es',
    'Fm',
    'Md',
    'No',
    'Lr',
    'Rf',
    'Db',
    'Sg',
    'Bh',
    'Hs',
    'Mt',
    'Ds',
    'Rg',
    'Cn',
    'Nh',
    'Fl',
    'Mc',
    'Lv',
    'Ts',
    'Og',
]
elements_lower = [element.lower() for element in elements]


def user_word() -> str:
    while True:
        word = input('Enter word:').lower()
        if word.isalpha():
            break
        print(f'word must contain only alphabetic characters')
    return word


def find_all_matching_element_combinations(
        word: str,
        results: List,
        substring: Optional[str] = None,
        matches: Optional[List] = None
) -> None:
    substring = substring if substring is not None else word
    if substring == '':
        result = ''.join(matches)
        if result == word:
            print(f'match found: {result} from {matches}')
            results.append(matches)
        return None

    char_1 = substring[0]
    if char_1 in elements_lower:
        matches_1 = matches + [char_1] if matches else [char_1]
        find_all_matching_element_combinations(word=word, results=results, substring=substring[1:], matches=matches_1)

    char_2 = substring[:2]
    if (char_2 in elements_lower) and (len(char_2) == 2):
        matches_2 = matches + [char_2] if matches else [char_2]
        find_all_matching_element_combinations(word=word, results=results, substring=substring[2:], matches=matches_2)


def main(word: Optional[str] = None) -> Optional[bool]:
    """good examples: mtds nico nini"""
    word = str(word) if word else user_word()
    if not word.isalpha():
        print(f'word must contain only alphabetic characters')
        return None
    results = []
    find_all_matching_element_combinations(word=word, results=results)
    return bool(results)


if __name__ == '__main__':
    fire.Fire(main)
