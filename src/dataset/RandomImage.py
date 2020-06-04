from random import randint
from matplotlib.pyplot import imshow

name2id = {
    'alouatta_palliata': 0,
    'erythrocebus_patas': 1,
    'cacajao_calvus': 2,
    'macaca_fuscata': 3,
    'cebuella_pygmea': 4,
    'cebus_capucinus': 5,
    'mico_argentatus': 6,
    'saimiri_sciureus': 7,
    'aotus_nigriceps': 8,
    'trachypithecus_johnii': 9,
}


def get_random_monkey(data, random=True, cat_num=None, pic_num=None):
    maximum = len(data)
    # Pic at random
    if random:
        rand = randint(0, maximum)
        a, y = data[rand]
    elif cat_num is None or pic_num is None:
        return 'Fuck you'

    # Not Random and sb gave cat and pic number
    else:
        j = 1
        # I have to iterate through examples, It could be nicer if we have dataset_loader setup?
        for i in range(0, maximum):
            a, y = data[i]
            if y == cat_num and j == pic_num:
                break
            elif y == cat_num and j != pic_num:
                j += 1

    # Adding string label
    for name, idx in name2id.items():
        if idx == y:
            name_display = name
            print("y = " + str(y) + "-" + name_display)
            return imshow(a.numpy()[0])

    return 'What have just happened?'
