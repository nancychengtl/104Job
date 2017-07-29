# -*- coding: utf-8 -*-
import csv

def setUpData():
    #amazon
    file = open("../user_log_testset/user_log_testset.csv", 'r')
    writer = csv.writer(open("../104/score_log.csv", 'w'))

    #local
    # file = open("data/user_log.csv", 'r')
    # writer = csv.writer(open("data/score_log.csv", 'w'))

    for row in csv.reader(file, delimiter='|'):
        if row[1] == "applyJob":
            row = [x.replace('applyJob', '5') if x == 'applyJob' else x for x in row]
            writer.writerow((row[0], row[2], row[1], row[4]))
        elif row[1] == "saveJob":
            row = [x.replace('saveJob', '3') if x == 'saveJob' else x for x in row]
            writer.writerow((row[0], row[2], row[1], row[4]))

        elif row[1] == "viewJob":
            row = [x.replace('viewJob', '2') if x == 'viewJob' else x for x in row]
            writer.writerow((row[0], row[2], row[1], row[4]))

        elif row[1] == "viewCust":
            row = [x.replace('viewCust', '1') if x == 'viewCust' else x for x in row]
            writer.writerow((row[0], row[2], row[1], row[4]))
        else:
            writer.writerow((row[0], row[2], row[1], row[4]))
def deleteUselessColumn():
    writer = csv.writer(open("../104/job/jobs_nancy.csv", 'w'))
    with open("../104/job/job_structured_info.csv","rb") as source:
        for r in csv.reader(source, delimiter='|'):
            writer.writerow((r[1], r[2], r[3], r[4]))
            print r[1]

def zeroApply():
    writer = csv.writer(open("../user_log_testset/user_log_testset_result.csv", 'w'), delimiter='|')
    with open("../user_log_testset/user_log_testset_sample.csv", "r") as source:
        reader = csv.reader(source)
        headers = next(reader, None)
        headers = str(headers).replace("\"", "")

        writer.writerow(headers)

        for r in csv.reader(source, delimiter='|'):

            r[2] = 1
            writer.writerow((r[0], r[1], r[2]))
            print r[1]




if __name__ == '__main__':
    setUpData()
    # deleteUselessColumn()
    # zeroApply()
