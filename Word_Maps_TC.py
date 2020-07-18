article_text_folders = "./train-articles"
article_maps =  "./data/train-task2-TC.labels"
import io
import glob
import os
import pandas as pd
f_in = io.open(article_maps, mode="r", encoding="utf-8")
key = "article{}.txt"
data={}
counter = 0
fl_id_lst = []
start_range_lst = []
end_rage_lst = []
text_ = []
TC_ = []
for ln in f_in:
    '''
     for each line map the actual text file 
    '''
    fl_id = 0
    start_range = 0
    end_rage = 0
    tab_seperated = ln.split('\t')
    print(tab_seperated)
    fl_id = tab_seperated[0].strip()
    tc_text = tab_seperated[1].strip()
    start_range = int(tab_seperated[2].strip())
    end_rage = int(tab_seperated[3].strip().replace('\n',''))
    tmp = key.format(fl_id)
    tmp_fl = os.path.join(article_text_folders,tmp)
    artice_text = io.open(tmp_fl, mode="r", encoding="utf-8")
    data = artice_text.read()
    propaganda_ = data[start_range:end_rage]
    artice_text.close()
    fl_id_lst.append(fl_id)
    TC_.append(tc_text)
    start_range_lst.append(start_range)
    end_rage_lst.append(end_rage)
    text_.append(propaganda_)
f_in.close()
frame_ = pd.DataFrame({'File_ID': fl_id_lst, 'Classification':TC_,'Start_IDX':start_range_lst, 'End_IDX':end_rage_lst, 'Associated_Propaganda':text_})
writer = pd.ExcelWriter('mapping_TC.xlsx', engine='xlsxwriter')
frame_.to_excel(writer, sheet_name='task-2')
writer.save()
