
import riva.client
import os
from datetime import datetime
from datetime import date
import pandas as pd
from evaluate import load
import glob
from jiwer import wer

RIVA_HOST  = '192.168.5.18:50051'
CSV_FILE = ""
SAMPLE = 10
AUDIO_FILE_DIR = 'D:/sp/stt_test/riva_test/chunk-en-2023-5-22/'
REPORTS_DIR = 'D:/sp/stt_test/reports/'
CLIENT_NAME = 'amc'


positive_boost_words =['mutual', 'fund', 'prudential', 'investment', 'icici', 'scheme', 'sip', 'time', 'right', 'number', 'thank', 'morning', 'afternoon', 'funds', 'speaking', 'website', 'amount', 'market', 'actually', 'may', 'related', 'help', 'read', 'documents', 'need', 'carefully', 'option', 'rupees', 'trying', 'redemption', 'charges', 'risk', 'request', 'day', 'invest', 'completed', 'form', 'speak', 'given', 'please', 'dot', 'term', 'process', 'regarding', 'investments', 'subject', 'check', 'transaction', 'hundred', 'observed', 'date', 'inform', 'send', 'folio', 'login', 'visited', 'account', 'details', 'service', 'october', 'application', 'evening', 'subjected', 'month', 'last', 'years', 'received', 'mail', 'new', 'audible', 'means', 'successfully', 'period', 'problem', 'password', 'redeem', 'register', 'information', 'email', 'app', 'invested', 'valuable', 'holding', 'registered', 'year', 'thirty', 'bank', 'issue', 'mention', 'procedure', 'visit', 'equity', 'user', 'manager', 'online', 'banking', 'point', 'contact', 'kumar', 'complete']
positive_lm_score = 100.0
negative_boost_words = ["find"]
negative_lm_score = -100.0
def get_encoding(audio_file):
    file_extension = audio_file.split('.')[-1]
    if file_extension == 'wav':
        encoding = riva.client.AudioEncoding.LINEAR_PCM
    elif file_extension == 'flac':
        encoding = riva.client.AudioEncoding.FLAC
    elif file_extension == 'alaw':
        encoding = riva.client.AudioEncoding.ALAW
    elif file_extension == 'mulaw':
        encoding = riva.client.AudioEncoding.MULAW
    else:
        raise Exception(f'Audio format ".{file_extension}" not supported.')
    return encoding
def run_inference(audio_file, server=RIVA_HOST, print_full_response=False):
    with open(audio_file, 'rb') as fh:
        data = fh.read()
    auth = riva.client.Auth(uri=server)
    client = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        encoding=get_encoding(audio_file),
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=False,
    )
    riva.client.add_word_boosting_to_config(config, positive_boost_words, positive_lm_score)
    riva.client.add_word_boosting_to_config(config, negative_boost_words, negative_lm_score)
    riva.client.add_audio_file_specs_to_config(config, audio_file)
    response = client.offline_recognize(data, config)
    if print_full_response:
        return response
    else:
        return response.results[0].alternatives[0].transcript


def calculate_wer():
    df_first = pd.read_csv('D:/sp/stt_test/riva_test/ChunkedCallsData-en-2023-05-22.csv')
    df_first['format_in_wav'] = df_first['filename'].str.replace('mp3', 'wav')
    df_sample = pd.DataFrame()
    df_inference = pd.DataFrame()
    df_sample = df_first.sample(SAMPLE)
    reference_text_list = df_sample['modeified_text'].to_list()
    predicted_text_list = []
    for audio in df_sample['format_in_wav']:
        full_audio = AUDIO_FILE_DIR + audio
        print(full_audio)
        try:
            predicted_text_value = run_inference(full_audio, server=RIVA_HOST, print_full_response=False)
            predicted_text_list.append(predicted_text_value)
        except(IndexError):
            predicted_text_list.append("Not detected")
            print(f"error at {full_audio}")
    
    df_inference['filename'] = df_sample['filename']
    df_inference['reference_text'] = reference_text_list
    df_inference['predicted_text'] = predicted_text_list
    df_inference['wer'] = df_inference.apply(lambda row: wer(row['reference_text'], row['predicted_text']), axis = 1)

    print("reference text length = ", len(reference_text_list), "predicted text length = ", len(predicted_text_list))
    wer_1 = load("wer")
    wer_score = wer_1.compute(references=reference_text_list, predictions=predicted_text_list)
    print("wer score = ", wer_score)
    return wer_score, df_inference


def main():
    #df = pd.read_csv(CSV_FILE)
    wer_value, df_inference = calculate_wer()
    report_dict = {
        'wer':wer_value,
        'sample':SAMPLE,
        'client_project':'amc_english_audio'
    }
    print("report dictionary = ", report_dict)
    report_df = pd.DataFrame(report_dict, index = [0])

    date_folder_name = date.today().strftime("%Y-%m-%d")
    client_directory_path = os.path.join(REPORTS_DIR, CLIENT_NAME)
    os.makedirs(client_directory_path, exist_ok=True)
    date_folder = os.path.join(client_directory_path, date_folder_name)
    os.makedirs(date_folder, exist_ok=True)
    date_directory_path = os.path.join(client_directory_path, date_folder)
    print("date directory = ", date_directory_path)


    # Get a list of files in the directory
    files = glob.glob(os.path.join(date_directory_path, '*'))

    files.sort(key=os.path.getmtime, reverse=True)

    # Check if any files exist in the directory
    if files:
        last_file_name = os.path.basename(files[0])
        print(f"The last file in the directory is: {last_file_name}")
        last_file_iteration = last_file_name.split('_')[-1]
        last_file_iteration_no = int(last_file_iteration)
        last_file_iteration_no = last_file_iteration_no +1
        last_file_iteration_no = str(last_file_iteration_no)
        client_file_name = CLIENT_NAME + "_" + last_file_iteration_no
        client_file_path = os.path.join(date_directory_path, client_file_name)
        os.makedirs(client_file_path, exist_ok=True)
    else:
        client_file_name = f"{CLIENT_NAME}_1"
        client_file_path = os.path.join(date_directory_path, client_file_name)
        os.makedirs(client_file_path, exist_ok=True)


    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    report_file_name = f"report_{datetime_string}.csv"
    inference_file_name = f"inference_{datetime_string}.csv"
    report_file_path = os.path.join(client_file_path, report_file_name)
    inference_file_path = os.path.join(client_file_path, inference_file_name)
    report_df.to_csv(report_file_path)
    df_inference.to_csv(inference_file_path)

if __name__ == "__main__":
    main()
