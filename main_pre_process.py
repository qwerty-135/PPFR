from disk_smart_parameter import ST4000DM000_REALIST, ST12000NM0008_NECESSARY_LIST, ST12000NM0008_REALIST
from pre_process import pre_process, failure_filter, min_max

DISK_MODEL, DISK_CAPACITY_BYTES = "ST12000NM0008", "12000138625024"

# "ST12000NM0008","12000138625024",
# "ST4000DM000", "4000787030016",
input_path_source = "D:/Blackbaze/"
input_dir = [
    # "2021/data_Q1_2021",
    # "2021/data_Q2_2021",
    # "2021/data_Q3_2021",
    # "2021/data_Q4_2021",
    "2022/data_Q1_2022",
    "2022/data_Q2_2022",
    "2022/data_Q3_2022",
    "2022/data_Q4_2022",

]

output_path = "./data"
pre_process(input_path_source, input_dir, output_path, DISK_MODEL,DISK_CAPACITY_BYTES, ST12000NM0008_REALIST)

input_path = "./data"
output_path = "./failure_disk_data"
failure_filter(input_path, output_path)

# input_path = "./failure_disk_data"
# print(min_max(input_path, len(ST12000NM0008_NECESSARY_LIST)))
