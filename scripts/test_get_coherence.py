from gensim import models
from gensim.models.coherencemodel import CoherenceModel
from os.path import join
import pickle

workdir = ".."
datasets = ["roman18_205_10t_2000i_200opt", "roman18_205_20t_2000i_200opt", "roman18_205_30t_2000i_200opt", "roman18_205_30t_2000i_400opt", "roman18_205_30t_3000i_300opt", "roman18_205_30t_4000i_400opt", "roman18_205_40t_2000i_200opt"]
#dataset = "roman18_205_40t_2000i_200opt"  # Name des datasets (z.B. mmt_2020-11-19_11-38)
#model_path = join(workdir, "results", dataset, "model", dataset + ".gensim")
#dictcorpus_path = join(workdir, "results", dataset, "pickles", "dictcorpus.pickle")
#vectorcorpus_path = join(workdir, "results", dataset, "pickles", "vectorcorpus.pickle")
#tokenzied_texts_path = join(workdir, "results", dataset, "pickles", "allprepared.pickle")


for dataset in datasets:
    
    model_path = join(workdir, "results", dataset, "model", dataset + ".gensim")
    dictcorpus_path = join(workdir, "results", dataset, "pickles", "dictcorpus.pickle")
    vectorcorpus_path = join(workdir, "results", dataset, "pickles", "vectorcorpus.pickle")
    tokenzied_texts_path = join(workdir, "results", dataset, "pickles", "allprepared.pickle")
    
    model = models.ldamodel.LdaModel.load(model_path)
    with open(vectorcorpus_path, "rb") as filehandle:
            vectorcorpus = pickle.load(filehandle)

    cm_umass = CoherenceModel(model=model, corpus=vectorcorpus, coherence='u_mass')     
    coherence_umass = cm_umass.get_coherence()  # get coherence value

    print(dataset, " coherence score u_mass:", coherence_umass)
