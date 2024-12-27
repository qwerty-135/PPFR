import csv
import os

from tqdm import tqdm


def disk_smart_extract(row, necessary_list):
    temp_row = [None for _ in range(len(necessary_list))]
    p = 0
    for i in necessary_list:
        if not row[i]:
            raise Exception
        temp_row[p] = row[i]
        p += 1
    return temp_row


def pre_process(input_path_source, input_dir, output_path, select_disk, capacity_bytes, disk_smart_parameter):
    # csv_output = csv.writer(open(output_path+"/"+"test.csv","w",encoding="utf-8",newline=""))
    output_dict = {}
    for dir in input_dir:
        input_path = input_path_source + dir
        for file in tqdm(os.listdir(input_path), position=0):
            if "csv" not in file:
                continue
            csv_file = csv.DictReader(open(input_path + "/" + file, "r", encoding="utf-8"))
            for row in csv_file:
                if row['model'] == select_disk:
                    if row['capacity_bytes'] == "-1" or row['capacity_bytes'] != capacity_bytes:
                        continue
                    sn = row["serial_number"]
                    try:
                        temp_row = ([row['date'], sn, row['failure']] +
                                    disk_smart_extract(row, disk_smart_parameter))
                    except Exception as e:
                        print("File:", file, " Row:", sn, "is an Exception, please notice \n")
                        continue
                    output_dict.setdefault(sn, [])
                    output_dict[sn].append(temp_row)
    for sn in tqdm(output_dict):
        output = open(output_path + "/" + sn + ".csv", "w", encoding="utf-8", newline="")
        csv.writer(output).writerows(output_dict[sn])
        output.close()


def failure_filter(input_path, output_path):
    for file in tqdm(os.listdir(input_path)):
        csv_file = csv.reader(open(input_path + "/" + file, "r", encoding="utf-8"))
        counter = 0
        temp_rows = []
        for row in csv_file:
            counter += 1
            if row[2] == "1":
                target_file = csv.writer(open(output_path + "/" + file, "w", encoding="utf-8", newline=""))
                target_file.writerows(temp_rows)
                break
            temp_rows.append(row)


def min_max(input_path="./failure-disk-data", smart_number=21):
    smart_value = [[1000000000000, -1] for _ in range(smart_number)]
    for file in tqdm(os.listdir(input_path)):
        csv_file = csv.reader(open(input_path + "/" + file, "r", encoding="utf-8"))
        for row in csv_file:
            p = 0
            for item in row[3:]:
                smart_value[p][0] = min(smart_value[p][0], int(item))
                smart_value[p][1] = max(smart_value[p][1], int(item))
                p += 1
    return smart_value
