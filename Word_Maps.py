article_text_folders = "./tc-articles"
article_maps =  "./data/test-new.labels"
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
    start_range = int(tab_seperated[1].strip())
    end_rage = int(tab_seperated[2].strip().replace('\n',''))
    tmp = key.format(fl_id)
    tmp_fl = os.path.join(article_text_folders,tmp)
    artice_text = io.open(tmp_fl, mode="r", encoding="utf-8")
    data = artice_text.read()
    propaganda_ = data[start_range:end_rage]
    artice_text.close()
    fl_id_lst.append(fl_id)
    start_range_lst.append(start_range)
    end_rage_lst.append(end_rage)
    text_.append(propaganda_)
f_in.close()
frame_ = pd.DataFrame({'File_ID': fl_id_lst, 'Start_IDX':start_range_lst, 'End_IDX':end_rage_lst, 'Associated_Propaganda':text_})
writer = pd.ExcelWriter('test-mapping.xlsx', engine='xlsxwriter')
frame_.to_excel(writer, sheet_name='task-1')
writer.save()
