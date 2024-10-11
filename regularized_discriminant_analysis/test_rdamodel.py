import matplotlib
import matplotlib.pyplot as plt
import numpy

from models.RegularizedDiscriminantAnalysis import RDAModel

# Any number of data sets (list of centroids & sigmas)
# sizes are len(centroids)
def generate_classifier_data(centroids, sigmas, sizes, seed):
    rng = numpy.random.default_rng(seed=seed)
    
    # Initialize lists for classes and labels
    xs = []
    ys = []
    n_classes = 0 
    
    # Move through data and generate classes and labels
    for i, (c, sigma, size) in enumerate(zip(centroids, sigmas, sizes)):
        classes = rng.multivariate_normal(c, sigma, size=size)
        # List in one column of labels asigned to each class
        labels = numpy.full(shape=(size, 1), fill_value=i)
        
        xs.append(classes)  
        ys.append(labels)
    
    # List in two columns of vectors with label 1 and lable 2 (component x, component y)
    xs = numpy.vstack(xs)
    ys = numpy.vstack(ys)
    n_classes = len(centroids)

    return xs, ys, n_classes

def test_RDAModel_2D():
    # Generate 2D (2 features) data to visualize a simple example
    c_1=numpy.array([-1,1])
    c_2=numpy.array([2,1])
    c_3=numpy.array([0,-10])
    c_4=numpy.array([-3,-3])
    centroid_list = [c_1,c_2,c_3,c_4]
    sigma_1 = numpy.array([[1, 0.5], [0.5, 1]])
    sigma_2 = numpy.array([[0.7, 0.5], [0.5, 1.2]])
    sigma_3 = numpy.array([[1.0, -0.4], [-0.4, 1.0]])
    sigma_4 = numpy.array([[1.1, 0.3], [0.3, 0.9]]) 
    sigma_list = [sigma_1,sigma_2,sigma_3,sigma_4]
    nsamples_1 = 500
    nsamples_2 = 500
    nsamples_3 = 500
    nsamples_4 =350
    nsamples_list = [nsamples_1,nsamples_2,nsamples_3,nsamples_4]
    xs, ys, n_classes = generate_classifier_data(centroid_list,sigma_list,nsamples_list, 73784) #Train data
    xs_test, ys_test, n_classes = generate_classifier_data(centroid_list,sigma_list,nsamples_list, 73784) # Test data

    # Plot 2D example

    model=RDAModel()
    alpha = 0.8
    result = model.fit(xs, ys, n_classes, alpha)
    #result.predict(numpy.array([-1,1]))

    x_p, predictions = result.viz_everything(xs, ys)

    plt.scatter(x_p[:,0], x_p[:, 1], c=predictions)
    plt.title("2D example")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    print("predictions 2D", result.predict_LL([-4,-3]))

    # Metrics
    y_p=result.predict(xs_test)
    resulting=numpy.column_stack((ys_test,y_p))

    errors = resulting[resulting[:, 0] != resulting[:, 1]]

    accuracy = numpy.mean(predictions == ys) * 100
    print(f"Accuracy of the model: {accuracy:.2f}%")

    # Confusion matrix

    ys_test_flat = ys_test.flatten()

    n_classes = len(numpy.unique(ys_test_flat))

    matrix = numpy.zeros((n_classes, n_classes), dtype=int)
        
    # Map each label to an index
    label_to_index = {label: i for i, label in enumerate(numpy.unique(ys_test_flat))}
        
    # Populate the confusion matrix
    for true, pred in zip(ys_test_flat, y_p):
        matrix[label_to_index[true], label_to_index[pred]] += 1
        
    print(matrix)
    return

def test_RDAModel_3D():
    # Generate 3D train and test data (works with n-dimensional data)
    c_1=numpy.array([-1,1,2])
    c_2=numpy.array([2,1,3])
    c_3=numpy.array([0,-10,4])
    c_4=numpy.array([-3,-3,-4])
    centroid_list = [c_1,c_2,c_3,c_4]
    sigma_1 = numpy.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])
    sigma_2 = numpy.array([[0.7, 0.5, -0.2], [0.5, 1.2, 0.3], [-0.2, 0.3, 0.8]])
    sigma_3 = numpy.array([[1.0, -0.4, 0.1], [-0.4, 1.0, 0.2], [0.1, 0.2, 1.2]])
    sigma_4 = numpy.array([[1.1, 0.3, 0.1], [0.3, 0.9, 0.2], [0.1, 0.2, 0.8]]) 
    sigma_list = [sigma_1,sigma_2,sigma_3,sigma_4]
    nsamples_1 = 500
    nsamples_2 = 500
    nsamples_3 = 500
    nsamples_4 =350
    nsamples_list = [nsamples_1,nsamples_2,nsamples_3,nsamples_4]
    xs, ys, n_classes = generate_classifier_data(centroid_list,sigma_list,nsamples_list, 73784) # This is the training data
    xs_test, ys_test, n_classes = generate_classifier_data(centroid_list,sigma_list,nsamples_list, 73784) # Test data

    # Plot 3D example

    model=RDAModel()
    alpha = 0.7
    result = model.fit(xs, ys, n_classes, alpha)
    #result.predict(numpy.array([-1,1]))

    x_p, predictions = result.viz_everything(xs, ys)

    plt.scatter(x_p[:,0], x_p[:, 1], c=predictions)
    plt.title("3D example")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    print("predictions 3D", result.predict_LL([-4,-3, 3]))

    # Metrics
    y_p=result.predict(xs_test)
    resulting=numpy.column_stack((ys_test,y_p))

    errors = resulting[resulting[:, 0] != resulting[:, 1]]

    accuracy = numpy.mean(predictions == ys) * 100
    print(f"Accuracy of the model: {accuracy:.2f}%")

    # Confusion matrix

    ys_test_flat = ys_test.flatten()

    n_classes = len(numpy.unique(ys_test_flat))

    matrix = numpy.zeros((n_classes, n_classes), dtype=int)
        
    # Map each label to an index
    label_to_index = {label: i for i, label in enumerate(numpy.unique(ys_test_flat))}
        
    # Populate the confusion matrix
    for true, pred in zip(ys_test_flat, y_p):
        matrix[label_to_index[true], label_to_index[pred]] += 1
        
    print(matrix)
    return

if __name__ == "__main__":
    test_RDAModel_2D()
    test_RDAModel_3D()

