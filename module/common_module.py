

import os
import smtplib

BraTSshape = (32, 128, 128)

# make path
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


# Non-repetitive list
from random import randint
data = []
def unique_rand(inicial, limit, total):

        data = []

        i = 0

        while i < total:
            number = randint(inicial, limit)
            if number not in data:
                data.append(number)
                i += 1

        return data


# get random seed
def getmillisecond():
    from datetime import datetime
    return str(datetime.now()).split('.')[1]


# write history log
def history_log(path, history, write_stat):
    # print(history)
    the_file = open(path, write_stat)
    the_file.write(history)
    the_file.close()




