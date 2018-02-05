from model.webis_text_pair import WebisTextPair
import os


"""
Webis corpus: 7859 text pairs (often multiple sentences, not 1:1)
Contain flag, wether one is paraphrase of other
"""

def read_file(path, metadata=False):
    with open(path, 'r') as f:
        file = f.read()
    if metadata:
        return True if file.split()[1] == 'Yes' else False
    else:
        return file



def read(webis_folder_path):
    text_pair_list = []
    number_of_text_pairs = int(len(os.listdir(webis_folder_path)) / 3)
    for i in range(1, number_of_text_pairs + 1):
        try:
            id = i
            path = webis_folder_path + '/' + str(i) + '-'

            is_paraphrase = read_file(path + 'metadata.txt', metadata=True)
            original = read_file(path + 'original.txt', metadata=False)
            paraphrase = read_file(path + 'paraphrase.txt', metadata=False)
            text_pair_list.append(WebisTextPair(id, paraphrase, original, is_paraphrase))
        except Exception as e:
            print(i, e)

    return text_pair_list
