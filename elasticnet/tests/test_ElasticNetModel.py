import csv
import numpy 
from elasticnet.models.ElasticNet import ElasticNetModel

# import matplotlib.pyplot as plt
def test_predict():
    model = ElasticNetModel()
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[float(v) for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[float(v) for k,v in datum.items() if k=='y'] for datum in data]).flatten()
    # print(X.shape)
    results = model.fit(X,y)
    preds = results.predict(X)
    rmse = numpy.sqrt(numpy.mean((preds - y) ** 2))

    print("Root Mean Square Error (RMSE):", rmse)
    
    tss = numpy.sum((y - numpy.mean(y)) ** 2)
    rss = numpy.sum((y - preds) ** 2)

    r_2 = 1 - (rss / tss)
    print("R2 score: ",r_2)
    # plt.plot( y, color='blue')
    # plt.plot( preds, color='orange')
    # plt.show()
    return preds

    plt.show()
predss=test_predict()
