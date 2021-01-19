import sys, datetime, csv, os, pickle, spacy, argparse
from os import listdir
from os.path import isfile, join
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
nlp = spacy.load("en_core_web_lg")

## Loads embedding model and bert tokenizer
def load_imports(args):
    ## Imports necessary packages for embedding a document
    global torch
    import torch
    ## Loads BERT model (if necessary)
    if args.model_flag == "BERT":
        from transformers import BertModel
        model = BertModel.from_pretrained('bert-base-uncased')
        if args.gpu:
            model = model.cuda()
    ## Loads SBERT model (if necessary)
    elif args.model_flag == "SBERT":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    ## Loads BERT tokenizer (always necessary)
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, bert_tokenizer

## Tokenizes a text based on a given spacy object
def tokenize(text):
    tokens = [tok.text for tok in nlp.tokenizer(text)]
    return tokens

## Flattens a two-dimensional list   
def flatten(listoflists):
    list = [item for sublist in listoflists for item in sublist]
    return list

## Gets name of files in a list from a directory
def get_all_files(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f)) \
             and ".txt" in f]
    files = sorted(files)
    return files

## Reads in all files
def load_interviews(args):
    ## Retrieves all files from a directory 
    files = get_all_files(args.input_folder)
    filepaths = [args.input_folder + f for f in files]
    data = dict()
    ## Loads transcripts of each interview
    no_sent, short_sent, changed_sent, good_sent = 0, 0, 0, 0
    for i, path in tqdm(enumerate(filepaths), desc='Loading files...'):
        idy = int(files[i].split(".")[0].split(" ")[1])
        data[idy] = {'sents': [], 'speaker': []}
        with open(path, encoding='utf8') as f:
            for line in f:
                ## Splits each dialogue turn into sentences
                line_sents = sent_tokenize(line.strip()[4:])
                for orig_sent in line_sents:   
                    ## Cleans each sentence
                    sent = orig_sent.replace('[long pause] ', '')
                    sent = orig_sent.replace('[long pause]', '')
                    sent = sent.replace('…', '...')
                    new_sent = sent.replace("’", "'")
                    if new_sent[:2] in [", ", ". "]:
                        new_sent = new_sent[2:]
                    if new_sent[:4] == "... ":
                        new_sent = new_sent[4:]
                    ## Ignotes empty sentences
                    if new_sent == "" or new_sent == "[long_pause]":
                        no_sent += 1
                        continue
                    elif new_sent != orig_sent:
                        changed_sent += 1
                    else:
                        good_sent += 1
                    ## Filters out short sentences
                    if len(tokenize(new_sent)) > args.min_length:
                        data[idy]['sents'].append(new_sent)
                        data[idy]['speaker'].append(line.strip()[0])
    #print("BAD SENTENCES: " + str(no_sent))
    #print("SHORT SENTENCES: " + str(short_sent))
    #print("CHANGED SENTENCES: " + str(changed_sent))
    #print("GOOD SENTENCES: " + str(good_sent))
    return data

## Reads in metadata file
def read_csv(metadata_file):
    metadata = dict()
    with open(metadata_file, 'rt', encoding='utf8') as f:
        csvreader = csv.reader(f)
        for i, row in enumerate(csvreader):
            if i > 0:
                metadata[int(row[0])] = row[3]
    return metadata

## Generates a text's average embedding, also saves the length of the text
def embed_text(cur_text, model, bert_tokenizer, args):
    indices = bert_tokenizer.encode(cur_text, max_length=512, truncation=True)
    if args.model_flag == "BERT":
        ## Move to GPU if flag is set
        if args.gpu:
            tindices = torch.tensor(indices).unsqueeze(0).to(torch.device("cuda"))
            with torch.no_grad():
                embed = model(tindices)
            word_embeds = embed[0][0][1:-1,:].cpu().numpy()
        else:
            tindices = torch.tensor(indices).unsqueeze(0)
            with torch.no_grad():
                embed = model(tindices)
            word_embeds = embed[0][0][1:-1,:].numpy()
        avg_embed = np.mean(word_embeds, axis=0)
    elif args.model_flag == "SBERT":
        avg_embed = np.asarray(model.encode([cur_text])[0])
    return avg_embed

## Embeds each turn and computes embedding differences
def calc_diffs(data, model, bert_tokenizer, args):
    diffs = dict()
    for idy, data_dict in tqdm(data.items(), desc='Generating embeddings and computing differences...'):
        diffs[idy] = {"diffs": [], "turns": []}
        interviewer_turn = ""
        cur_diffs, subject_sents, interviewer_embed = [], [], []
        for i, sent in enumerate(data_dict['sents']):
            speaker = data_dict['speaker'][i]
            ## Middle of interviewer turn
            if speaker == "I" and interviewer_embed == []:
                interviewer_turn += " " + sent
            ## Start of new interviewer turn
            elif speaker == "I":
                ## Saves previous subject turn diffs
                diffs[idy]['diffs'].append(cur_diffs)
                diffs[idy]['turns'].append({"interviewer": interviewer_turn, "subject": subject_sents})
                cur_diffs, subject_sents, interviewer_embed = [], [], []
                interviewer_turn = sent
            ## Start of new subject turn (assuming there is a current interviewer turn)
            elif speaker == "S" and interviewer_embed == [] and interviewer_turn != "":
                ## Creates an embedding of the interviewer turn
                interviewer_embed = embed_text(interviewer_turn, model, bert_tokenizer, args)
                ## Compare subject embedding to interviewer embedding
                subject_embed = embed_text(sent, model, bert_tokenizer, args)
                cur_diffs.append(np.mean(np.absolute(np.subtract(interviewer_embed, subject_embed))))
                subject_sents.append(sent)
            ## Middle of subject turn (assuming there is a current interviewer turn)
            elif speaker == "S" and interviewer_turn != "":
                ## Compare subject embedding to interviewer embedding
                subject_embed = embed_text(sent, model, bert_tokenizer, args)
                cur_diffs.append(np.mean(np.absolute(np.subtract(interviewer_embed, subject_embed))))
                subject_sents.append(sent)
            ## If there is a subject turn but no interviewer turn
            else:
                print(str(i) + "\t" + str(data_dict['sents'][i]) + "\t" + str(data_dict['speaker'][i]))
    return diffs

def print_diffs(diffs, metadata, args):
    ## Saves each individual (turn, sentence) pair to an output file
    with open(args.output_file, 'w') as f:
        f.write("\t".join(["Interviewer_Prompt", "Subject_Sentence", "Group", "Distance_from_Prompt", "Embedding_Difference"]) + "\n")
        ## Splits into control and scz groups, and
        ## also by distance away from original sentence
        scz_diffs, control_diffs = dict(), dict()
        for idy, id_dict in diffs.items():
            group = metadata[idy]
            for i, turn_diffs in enumerate(id_dict['diffs']):
                prompt = id_dict['turns'][i]['interviewer']
                for j, diff in enumerate(turn_diffs):
                    #print("NUMBER OF TURNS: " + str(len(id_dict['turns'])))
                    #print("NUMBER OF DIFFS: " + str(len(turn_diffs)))
                    #print("NUMBER OF SUBJECT SENTENCES: " + str(len(id_dict['turns'][i]['subject'])))
                    response = id_dict['turns'][i]['subject'][j]
                    if group == "scz":
                        try:
                            scz_diffs[j+1].append(diff)
                        except KeyError:
                            scz_diffs[j+1] = [diff]
                        f.write("\t".join([prompt, response, "SCZ", str(j+1), str(round(diff, 4))]) + "\n")
                    else:
                        try:
                            control_diffs[j+1].append(diff)
                        except KeyError:
                            control_diffs[j+1] = [diff]
                        f.write("\t".join([prompt, response, "HC", str(j+1), str(round(diff, 4))]) + "\n")

    ## Prints out average difference
    print("Turn\tControl\tNum_Control\tScz\tNum_Scz\tDiff")
    max_turns = max(max(list(scz_diffs.keys())), max(list(control_diffs.keys())))
    total_scz, total_control = [], []
    for turn_num in range(1, max_turns+1):
        ## Computes SCZ stats
        try:
            slength = len(scz_diffs[turn_num])
            avg_scz = round(sum(scz_diffs[turn_num])/slength, 3)
            total_scz += scz_diffs[turn_num]
        except KeyError:
            avg_scz = "N/A"
            slength = 0
        ## Computes HC stats
        try:
            clength = len(control_diffs[turn_num])
            avg_control = round(sum(control_diffs[turn_num])/clength, 3)
            total_control += control_diffs[turn_num]
        except KeyError:
            avg_control = "N/A"
            clength = 0
        ## Computes difference between HC and SCZ
        try:
            avg_diff = round(avg_scz - avg_control, 3)
        except TypeError:
            avg_diff = 0
        print("\t".join([str(turn_num), str(avg_control), str(clength), str(avg_scz), str(slength), str(avg_diff)]))
    ## Computes overall averages
    avg_total_c = float(round(sum(total_control)/len(total_control), 3))
    avg_total_s = float(round(sum(total_scz)/len(total_scz), 3))
    avg_total_diff = round(avg_total_s - avg_total_c, 3)
    print("\t".join(["Overall", str(avg_total_c), str(len(total_control)), \
        str(avg_total_s), str(len(total_scz)), str(avg_total_diff)]))

def main(args):
    ## Read in metadata
    metadata = read_csv(args.metadata_file)
    ## Embeds all interviews + computes all differences
    intermediate_file = args.intermediate_folder + "worden_embed_diffs.pkl"
    try:
        with open(intermediate_file, 'rb') as f:
            diffs = pickle.load(f)
    except FileNotFoundError:
        model, bert_tokenizer = load_imports(args)
        ## Loads in files
        data = load_interviews(args)
        ## Calculates differences of all sentence pairs
        diffs = calc_diffs(data, model, bert_tokenizer, args)
        with open(intermediate_file, 'wb') as f:
            pickle.dump(diffs, f)
    ## Prints out differences by sentence distance
    print_diffs(diffs, metadata, args)


if __name__ == '__main__':
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Finding similar documents...')
    parser.add_argument('-input_folder', type=str, required=True, help='File containing input documents')
    parser.add_argument('-metadata_file', type=str, required=True, help='File containing metadata for documents')
    parser.add_argument('-intermediate_folder', type=str, required=True, help='Folder for storing intermediate data')
    parser.add_argument('-output_file', type=str, required=True, help='File for storing the difference of each sentence pair')
    parser.add_argument('-model_flag', type=str, required=True, help='Contextual embedding model to use')
    parser.add_argument('-min_length', type=str, required=False, default=3, help='Minimum sentence length')
    parser.add_argument('-gpu', action='store_true', help='Boolean for using a GPU or not')
    args = parser.parse_args()
    main(args)
    finish = datetime.datetime.now()
    print("Total time: " + str(finish-start))

'''
python code/bert_random_walk.py \
-input_folder data/Schizophrenia_data/Worden_fixed/ \
-metadata_file data/Worden_Transcripts/Worden_metadata.csv \
-intermediate_folder intermediate_data/ \
-output_file output/bert_difference_output.txt \
-model_flag BERT \
-gpu
'''