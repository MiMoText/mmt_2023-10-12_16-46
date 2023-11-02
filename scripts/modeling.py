#!/usr/bin/env python3

"""
Doing Topic Modeling on Eighteenth-Century French Novels with gensim using mallet in the context of MiMoText:

Modeling.

Performs the main topic modeling step using the gensim library and mallet. 
This requires properly prepared forms of the corpus to be used. 

See:
http://mallet.cs.umass.edu/
https://radimrehurek.com/gensim/
https://radimrehurek.com/gensim/models/wrappers/ldamallet.html
"""

# == Imports == 

import pickle
from os.path import join, realpath, split
from gensim import corpora
from gensim import models
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
import shutil
import helpers


# == Functions ==

def build_model(dictcorpus, vectorcorpus, paths, params): 
    """
    Creates the actual topic model from the data. 
    Key parameters are number of topics (numtopics) 
    and number of iterations (passes). 
    Other parameters can be set here.
    It is important to specify where mallet binary is found on your computer
    """
    
    # path to mallet binary
    mallet_path = paths["mallet_path"]

    model = LdaMallet(
        mallet_path,
        corpus = vectorcorpus,
        id2word = dictcorpus,
        num_topics = params["numtopics"],
        prefix = paths["identifier"] + "_",
        optimize_interval = params["optimize_interval"],  # choosing an optimal interval alpha and beta are generated automatically
        iterations = params["passes"],
        )

    
    return model

def get_coherence_score(model, vectorcorpus, textscorpus):

    # Doku: https://radimrehurek.com/gensim/models/coherencemodel.html
    # Links: https://markroxor.github.io/gensim/static/notebooks/topic_coherence_tutorial.html
    # https://medium.com/@kurtsenol21/topic-modeling-lda-mallet-implementation-in-python-part-3-ab03e01b7cd7
    
    # in Variable corpus muss Korpus im BoW-Format -> ist vectorcorpus
    # dictionary -> dictcorpus
    
    cm_umass = CoherenceModel(model=model, corpus=vectorcorpus, coherence='u_mass')   
    #cm_cv = CoherenceModel(model=model, corpus=vectorcorpus, texts = textscorpus, coherence='c_v')   # fkt nicht, bracuht gensim.state
    coherence_umass = cm_umass.get_coherence()  # get coherence value
    #coherence_cv = cm_cv.get_coherence()
    
    print("coherence score u_mass:", coherence_umass)
    #print("coherence score u_cv:", coherence_cv)


def move_output(workdir, identifier):
    '''
    Moves mallet output from the script directory into the results directory.
    '''
    destination = join(workdir, "results", identifier)
    files = ['_corpus.mallet', '_corpus.txt', '_doctopics.txt', '_inferencer.mallet', '_state.mallet.gz', '_topickeys.txt']
    full_path = realpath(__file__)
    path, filename = split(full_path)
    
    for file in files:
        name = identifier + file
        source = join(path, name)
        shutil.move(source, destination)       
    

# == Coordinating function ==

def main(paths, params):
    
    print("\n== modeling ==")
    workdir = paths["workdir"]
    identifier = paths["identifier"]
    dictcorpus = helpers.load_pickle(paths, "dictcorpus.pickle")
    vectorcorpus = helpers.load_pickle(paths, "vectorcorpus.pickle")
    textscorpus = helpers.load_pickle(paths, "allprepared.pickle")  # notwendig für c_v coherence
    model = build_model(dictcorpus, vectorcorpus, paths, params)
    get_coherence_score(model, vectorcorpus, textscorpus)
    helpers.save_model(paths, model)
    move_output(workdir, identifier)
    print("==", helpers.get_time(), "done modeling", "==")   
    return model