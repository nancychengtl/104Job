# -*- coding: utf-8 -*-
import csv

def setUpData():
    file = open("../104/user_log.csv", 'r')
    writer = csv.writer(open("../104/score_log.csv", 'w'))
    for row in csv.reader(file):
        if row[1] == "applyJob":
            row = [x.replace('applyJob', '5') if x == 'applyJob' else x for x in row]
            writer.writerow(row)
        elif row[1] == "saveJob":
            row = [x.replace('saveJob', '3') if x == 'saveJob' else x for x in row]
            writer.writerow(row)

        elif row[1] == "viewJob":
            row = [x.replace('viewJob', '2') if x == 'viewJob' else x for x in row]
            writer.writerow(row)

        elif row[1] == "viewCust":
            row = [x.replace('viewCust', '1') if x == 'viewCust' else x for x in row]
            writer.writerow(row)
        else:
            writer.writerow(row)



if __name__ == '__main__':
    setUpData()