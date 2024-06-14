import pandas as pd
import json

def gen_train_data(file_path, save_path):
    """
    file_path: 通过Label Studio导出的csv文件
    save_path: 保存的路径
    """
    data = pd.read_csv(file_path)
    for idx, item in data.iterrows():
        text = item['text']
        if pd.isna(text):
            text = ''
        text_list = list(text)
        label_list = []
        labels = item['label']
        label_list = ['O' for i in range(len(text_list))]
        if pd.isna(labels):
            pass
        else:
            labels = json.loads(labels)
            for label_item in labels:
                start = label_item['start']
                end = label_item['end']
                label = label_item['labels'][0]
                label_list[start] = f'B-{label}'
                label_list[start + 1:end - 1] = [f'I-{label}' for i in range(end - start - 2)]
                label_list[end - 1] = f'I-{label}'
        assert len(label_list) == len(text_list)
        with open(save_path, 'a') as f:
            for idx_, line in enumerate(text_list):
                if text_list[idx_] == '\t' or text_list[idx_] == ' ':
                    text_list[idx_] = '，'
                line = text_list[idx_] + ' ' + label_list[idx_] + '\n'
                f.write(line)
            f.write('\n')


if __name__ == '__main__':
    file_path = 'C:\\Users\\M_YoungBW\\Desktop\\label.csv'
    save_path = 'C:\\Users\\M_YoungBW\\Desktop\\OUT_label.txt'
    gen_train_data(file_path, save_path)
