#!/usr/bin/env python3

"""
Topic Modeling with gensim: Split text into segments.
The Size of chunks is controlled by the parameter chunksize (modify in roman18_run_pipeline.py).
Chunks that are smaller than 500 words are eliminated.

It also creates a metadata list combining the metadata with the
corresponding text chunks.

"""


# == Imports == 

import os
import glob
from os.path import join
from os.path import basename
import re
import pandas as pd


# == Functions ==


def load_text(textfile):
    """
    Loads a single plain text file. 
    Provides the content as a string.
    """
    with open(textfile, "r", encoding="utf8") as infile:
        text = infile.read()
        return text


def load_metadata(paths):
    """
    Loads the metadata file from disk.
    Provides it as a pandas DataFrame.
    """
    with open(paths["metadatafile_full"], "r", encoding="utf8") as infile:
        metadata = pd.read_csv(infile, sep='\t')
        return metadata



def split_text(text, params):
    """
    Takes text string and splits it into chunks.
    """
    chunksize = params["chunksize"]
    text = re.split("\W+", text)
    num_chunks = len(text) // chunksize
    chunks = [text[i:i + chunksize] for i in range(0, len(text), chunksize)]
    if len(chunks[-1]) < 500:   # last chunk is kept if >= 500 words
        chunks.pop()
    return chunks
    
    

def save_chunks(workdir, dataset, chunks, textid):
    counter = 0
    filenames = []
    for chunk in chunks: 
        filename = textid + "_" + "{:03d}".format(counter) + ".txt"
        filepath = filepath = join(workdir, "datasets", dataset, "txt")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = join(filepath, filename)
        chunk = " ".join(chunk)
        with open(filepath, "w", encoding="utf8") as outfile:
            outfile.write(chunk) 
        filenames.append(filename)
        counter +=1
    return filenames


def create_df_metadata(allfilenames, metadata):
    """
    Creates a pandas DataFrame in which the metadata information is assigned to the individual text chunks.
    """
    df_split = pd.DataFrame(columns=['filename', 'id', 'author', 'decade', 'gender', 'narration'])

    for filename in allfilenames:
        textid = basename(filename).split(".")[0]
        textid = re.sub(r'(_[a-zA-Z0-9]*(\[\d?\])?)(_\d{3})', r'\1', textid)  #RegEx ohne [1] hinter IDs: r'(_[a-zA-Z0-9]*(_\d{3})'

        key = metadata[metadata['id'] == textid].index.item()
        
        author = metadata.loc[key, 'author']
        
        decade = metadata.loc[key, 'decade']
            
        gender = metadata.loc[key, 'gender']
            
        narration = metadata.loc[key, 'narration']

        #df_split = df_split.append({'filename': filename, 'id': textid, 'author': author, 'decade': decade, 'gender': gender, 'narration': narration}, ignore_index=True)
        df_split.loc[len(df_split)] = pd.Series({'filename': filename, 'id': textid, 'author': author, 'decade': decade, 'gender': gender, 'narration': narration})
    
    return df_split
            

def write_metadata(df_split, paths):
    """
    Saves the DataFrame to a CSV file.
    """
    df_split.to_csv(paths["metadatafile_split"], sep='\t', columns=['filename', 'id', 'author', 'decade', 'gender', 'narration'], encoding="utf-8")




# == Coordinating function ==

def main(paths, params): 
    print("\n== splitting texts ==")
    workdir = paths["workdir"]
    dataset = paths["dataset"]
    allfilenames = []
    textpath = join(workdir, "datasets", dataset, "full", "*.txt")
    metadata = load_metadata(paths)
    for textfile in sorted(glob.glob(textpath)):
        textid = basename(textfile).split(".")[0]
        print(textid)
        text = load_text(textfile)
        chunks = split_text(text, params)
        filenames = save_chunks(workdir, dataset, chunks, textid)
        allfilenames.extend(filenames)
    df_split = create_df_metadata(allfilenames, metadata)
    write_metadata(df_split, paths)
        
        
    print("== done splitting texts ==")
                       


