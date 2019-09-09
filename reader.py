from drqa.reader import Predictor

def process(document, question, candidates=None,top_n=1):
    predictor = Predictor(None,'spacy',num_workers=0,normalize=True)
    predictions = predictor.predict(document,question,candidates,top_n)
    #table = prettytable.PrettyTable(['Rank','Span','Score'])
    val = []
    for i,p in enumerate(predictions,1):
        val.append(p[0])
    return val[0]
