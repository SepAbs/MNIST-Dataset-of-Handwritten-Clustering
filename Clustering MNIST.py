# from clusteval import clusteval
from hdbscan import HDBSCAN 
# from hnet import enrichment
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam, Nadam, SGD
from keras.utils import to_categorical
from matplotlib.pyplot import bar, colorbar, figure, gray, imshow, ion, legend, plot, savefig, scatter, show, style, subplot, subplots, title, xlabel, xticks, ylabel
from numpy import argsort, array, cumsum, matmul, prod, random, reshape, round, sort, sqrt, unique, vstack, where
from os import environ
from pandas import read_csv
from plotly.graph_objs import Figure, Layout, Scatter3d
from plotly.offline import iplot
from seaborn import histplot
from scipy.linalg import eigh
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
# from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, completeness_score, confusion_matrix, ConfusionMatrixDisplay, davies_bouldin_score, homogeneity_score, mutual_info_score, pairwise_distances_argmin, silhouette_score, rand_score, v_measure_score
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
from time import time
from warnings import filterwarnings
filterwarnings("ignore")
environ["KMP_DUPLICATE_LIB_OK"], environ["TF_CPP_MIN_LOG_LEVEL"], environ["TF_ENABLE_ONEDNN_OPTS"] = "TRUE", "3", "0"

# Loading & normalizing the dataset
trainSet, testSet = read_csv("mnist_train.csv"), read_csv("mnist_test.csv")
df = trainSet._append(testSet, ignore_index = True)
X_train, X_test, y_train, y_test, Optimizers, Activations, Architectures, Initializers, learningRates, Colors, lossFunction, Accuracy, Loss, validationAccuracy, validationLoss, Title, Alpha, localPopulation, Numbers, numberIterations, numberGenerations, Fitness, Parameters, bestLoss = trainSet.drop(["label"], axis = 1).astype("float32") / 255., testSet.drop(["label"], axis = 1).astype("float32") / 255., trainSet["label"].astype("int"), testSet["label"].astype("int"), {Adam: "Adam" , Nadam: "Nadam", SGD: "SGD"}, ["linear", "relu", "tanh"], [(128, 16), (500, 100, 30), (128, 64, 32)], ["glorot_normal", "glorot_uniform", "zero"], [0.01, 0.1, 0.25, 0.5, 0.85], ["black", "blue", "cyan", "green", "magenta", "maroon", "purple", "red", "teal", "yellow"], "mean_squared_error", "accuracy", "loss", "val_accuracy", "val_loss", "Model Evaluation", 0.6, 1, 10, 1, 1, [], [], {}
dfX_train, localBound, number_test_samples, Beta = X_train, X_train.shape[1], len(X_test), 1 - Alpha

# Defining the encoder input & decoder input dimentions.
Image, Length, Start = Input(shape = (localBound,)), int(sqrt(localBound)), int(number_test_samples * 0.35)

# 35% of test samples are chosen as validation samples
validationSet = testSet.iloc[Start : number_test_samples - Start]
y_val, X_val, arrX_train, arrX_test, arrtestSet = validationSet["label"], validationSet.drop(["label"], axis = 1), X_train.to_numpy().reshape(-1, 28, 28), X_test.to_numpy().reshape(-1, 28, 28), testSet.to_numpy()
print(f"There're {len(X_train)} samples of {Length} X {Length} images as train samples and {number_test_samples} samples as test samples.")

"""
# Plot some samples
for Number in range(Numbers):
    fig = figure
    imshow(arrX_train[Number], cmap = "gray")
    savefig(dpi = 1200)
    show()

# Plotting raw data records
Figure, ax = subplots()
ax.pie(df["label"].value_counts(), labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], autopct = '%1.1f%%', shadow = True, startangle = 90)
ax.axis("equal")
show()
histplot(df["label"])
show()
"""

"""
# Implementing PCA which causes weaker performance!
standardizedData, pca = StandardScaler().fit_transform(X_train), PCA(n_components = 784)

# Calculate eigenvalues and eigenvectors
Lambdas, Vectors = eigh(matmul(standardizedData.T, standardizedData), eigvals = (782, 783))

# Calculate unit vectors U1 = V1 and new coordinates
newCoordinates = vstack((matmul(Vectors.T, standardizedData.T), read_csv("mnist_train.csv")["label"])).T

pca.fit_transform(standardizedData)
cum_variance_retained = cumsum(pca.explained_variance_ / sum(pca.explained_variance_))

figure(1, figsize = (10, 6))
clf()
plot(cum_variance_retained, linewidth = 2)
axis("tight")
grid()
xlabel("number of compoments")
ylabel("cumulative variance retained")
savefig("pca_cumulative_variance.png")
show(block = False)

def encoderMaker(InputFunction, EncoderArchitecture):
    Encoded = Activation("relu")(Dense(EncoderArchitecture[0])(InputFunction))
    for Index in range(1, len(EncoderArchitecture)):
        Encoded = Activation("relu")(Dense(EncoderArchitecture[Index])(Encoded))
    return Model(InputFunction, Encoded), name = "Encoder")

def decoderMaker(InputFunction, EncoderArchitecture):
    reversedArchitecture = EncoderArchitecture[::-1]
    Decoded = Activation("relu")(Dense(reversedArchitecture[0])(InputFunction))
    for Index in range(1, len(reversedArchitecture)):
        Decoded = Activation("relu")(Dense(reversedArchitecture[Index])(Decoded))
    return Model(InputFunction, Activation("sigmoid")(Dense(localBound)(Decoded)), name = "Decoder")

def autoencoderTrainer(Architecture):
    inputEncoder, inputDecoder = Input(shape = (localBound,), name = "InputEncoder"), Input(shape = (16,), name = "InputDecoder")
    Encoder = encoderMaker(inputEncoder, Architecture)
    Autoencoder = Model(inputEncoder, decoderMaker(inputDecoder, Architecture)(Encoder(inputEncoder)), name = "Autoencoder")    
    Autoencoder.summary()
    Autoencoder.compile(loss = "mse", optimizer = Adam(learning_rate = 0.00025))
    Autoencoder.fit(Train, Train, batch_size = 6000, epochs = 5, shuffle = True)
    return Encoder, Autoencoder

def architectureMutation(Architectures):
    numberArchitectures = len(Architectures)
    for Architecture in Architectures:
        bufferArchitecture = Architecture.copy()
        bufferArchitecture[random.randint(0, len(Architecture))] = random.randint(2, localBound)
        mutatedArchitectures.append(bufferArchitecture)
    return mutatedArchitectures + [list(random.randint(2, localBound, size = random.randint(2, 10))) for Number in range(numberArchitectures)]

# Fine Tuning!
for Iteration in range(numberIterations):
    localArchitecture = list(sort(random.randint(2, localBound, size = random.randint(2, 10))))[::-1]
    _, localAutoencoder = autoencoderTrainer(localArchitecture)
    Architectures.append(localArchitecture)
    Fitness.append(localAutoencoder.evaluate(Validation, Validation))
    Parameters.append(localAutoencoder.count_params())

fsortedIndex = argsort(array(Fitness))[0]
selectedArchitectures, bestArchitecture, bestFitness, bestParameters = [Architectures[val] for val in fsortedIndex[0 : int(localPopulation / 2)]], [Architectures[fsortedIndex]], [Fitness[fsortedIndex]], [Parameters[fsortedIndex]]
for Generation in range(numberGenerations):
    Architectures, Fitness, Parameters = ArchitectureMutation(selectedArchitectures), [], []
    for localArchitecture in Architectures:
        _, localAutoencoder = autoencoderTrainer(localArchitecture)
        fitness.append(localAutoencoder.evaluate(Validation, Validation))
        Parameters.append(localAutoencoder.count_params())

    fsortedIndex = argsort(array(fitness))[0]
    bestArchitecture.append(Architectures[fsortedIndex])
    bestFitness.append(Fitness[fsortedIndex])
    bestParameters.append(Parameters[fsortedIndex])
    selectedArchitectures = [Architectures[val] for val in fsortedIndex[0 : int(localPopulation / 2)] ]

print(bestArchitecture, bestFitness, bestParameters)
"""

# Fine tuning hyperparameters
for Optimizer in Optimizers:
    for learningRate in learningRates:
        for Architecture in Architectures:
            decoderInput, number_hidden_layers = Input(shape = (Architecture[-1],)), len(Architecture) - 1
            for Initializer in Initializers:
                for Activation in Activations:
                    # Define & build the encoder & decoder
                    Encoded, Decoded = Dense(Architecture[0], activation = Activation, kernel_initializer = Initializer)(Image), Dense(Architecture[-2], activation = Activation, kernel_initializer = Initializer)(decoderInput)
                    for hiddenLayer in range(1, number_hidden_layers):
                        Encoded, Decoded = Dense(Architecture[hiddenLayer], activation = Activation, kernel_initializer = Initializer)(Encoded), Dense(Architecture[-(hiddenLayer + 2)], activation = Activation, kernel_initializer = Initializer)(Decoded)

                    Encoder, Decoder = Model(Image, Dense(Architecture[-1], activation = Activation)(Dense(Architecture[-2], activation = Activation)(Encoded)), name = "Encoder"), Model(decoderInput, Dense(localBound, activation = "sigmoid", kernel_initializer = Initializer)(Decoded), name = "Decoder")
                    
                    # Build main autoencoder
                    Autoencoder = Model(Image, Decoder(Encoder(Image)))

                    # Compile the autoencoder
                    Autoencoder.compile(loss = lossFunction, optimizer = Optimizer(learning_rate = learningRate))

                    # Fit the data on the Autoencoder
                    Autoencoder = Autoencoder.fit(X_train, X_train, batch_size = 500, epochs = 3, validation_split = 0.2, verbose = 0)
                    Results = Autoencoder.history
                    Tuner[(Optimizer, learningRate, Architecture, Initializer, Activation)] = Alpha * (Results[validationLoss][0] - Results[Loss][0]) + Results[Loss][0]

Optimizer, learningRate, Architecture, Initializer, Activation = min(Tuner, key = Tuner.get)
print(f"\nOptimal optimizer, architecture, primary weights initializer method, activation functions for hidden layers and learning rate are {Optimizers[Optimizer]}, {Architecture}, {Initializer}, {Activation} and {learningRate}, respectively.\n\nTrainig optimal autoencoder")
decoderInput, number_hidden_layers = Input(shape = (Architecture[-1],)), len(Architecture) - 1
# Define & build well-tuned encoder & decoder
Encoded, Decoded = Dense(Architecture[0], activation = Activation, kernel_initializer = Initializer)(Image), Dense(Architecture[-2], activation = Activation, kernel_initializer = Initializer)(decoderInput)

for hiddenLayer in range(1, number_hidden_layers):
    Encoded, Decoded = Dense(Architecture[hiddenLayer], activation = Activation, kernel_initializer = Initializer)(Encoded), Dense(Architecture[-(hiddenLayer + 2)], activation = Activation, kernel_initializer = Initializer)(Decoded)

Encoder, Decoder = Model(Image, Dense(Architecture[-1], activation = Activation)(Dense(Architecture[-2], activation = Activation)(Encoded)), name = "Encoder"), Model(decoderInput, Dense(localBound, activation = "sigmoid", kernel_initializer = Initializer)(Decoded), name = "Decoder")

# Build main autoencoder
Autoencoder = Model(Image, Decoder(Encoder(Image)))

# Compile the autoencoder
Autoencoder.compile(loss = lossFunction, optimizer = Optimizer(learning_rate = learningRate))
print(Autoencoder.summary())

# Saving model
Autoencoder.save("Autoencoder.keras")

# Fit the data on the Autoencoder
# Test set is now the validation set of model!
autoencoderHistory = Autoencoder.fit(X_train, X_train, batch_size = 500, epochs = 15, validation_split = 0.2, verbose = 0)

# Plot the training history (losses)
figure(figsize = (13, 5))
ion()
plot(autoencoderHistory.history[Loss])
plot(autoencoderHistory.history[validationLoss])
title(Title)
ylabel("Loss")
xlabel("Epochs")
legend(["Train Loss", "Test Loss"], loc = "upper right")
savefig(Title, dpi = 1200)
show()

# Plot the training history (accuracies)
figure(figsize = (13, 5))
ion()
plot(autoencoderHistory.history[Accuracy])
plot(autoencoderHistory.history[validationAccuracy])
title(Title)
ylabel("Accuracy")
xlabel("Epochs")
legend(["Train Accuracy", "Test Accuracy"], loc = "upper right")
savefig(Title, dpi = 1200)
show()

# Encoding images
X_train, X_test, X_val, Title = Encoder.predict(X_train, batch_size = 500, verbose = 0), Encoder.predict(X_test, batch_size = 50, verbose = 0), Encoder.predict(X_val, batch_size = 50, verbose = 0), "Elbow Method"

# Which number of clusters should be chosen as best one
figure(figsize = (13, 5))
ion()
plot(range(1, 11), [KMeans(algorithm = "elkan", init = "k-means++", max_iter = 1000, n_clusters = numberCluster, n_init = "auto", random_state = 42).fit(X_train).inertia_ for numberCluster in range(1, 11)], marker = "o")
title(Title)
xlabel("Number of Clusters")
ylabel("Inertia")
savefig(Title, dpi = 1200)
show()

# Define, fit (train) & evaluate four models
KAverages = {"K-Means Clustering Algorithm": (KMeans, GridSearchCV(KMeans(), param_grid = {"algorithm": ["elkan", "lloyd"], "init": ["k-means++", "random"], "max_iter": [100, 200, 500, 1000], "n_clusters": [10], "n_init": ["auto"]}, n_jobs = -1, cv = 5)), "Mini-Batch K-Means Clustering Algorithm": (MiniBatchKMeans, GridSearchCV(MiniBatchKMeans(), param_grid = {"init": ["k-means++", "random"], "max_iter": [100, 200, 500, 1000], "n_clusters": [10], "n_init": ["auto"]}, n_jobs = -1, cv = 5))}
for clustererName in KAverages:
    print(f"\n{clustererName} Modeling Started!")
    # Using Bayes Search approach for tuning model
    Clusterer, Title = KAverages[clustererName], f"{clustererName} Confusion Matrix"
    model = Clusterer[1]
    startTime = time()
    # Obtaining best paramteres by using splitted validation data
    model.fit(X_val)
    Duration = time() - startTime # Calculating how long fitting validation data takes
    print(f"\nModel's best score is:\n{model.best_score_}\n\nModel's best parameters are:\n{model.best_params_}")
    model = model.best_estimator_ # = Model(Optimal Parameters)
    print(f"\n{model}")
    # Evaluation Time!
    model.fit(X_train)
    Duration, Labels, Centroids, realLabels = time() - Start, model.labels_, model.cluster_centers_, len(unique(y_train))
    y_pred, nearestInstances, clusterIndexes, clustered_data_points, numberClusters = model.fit_predict(X_test), pairwise_distances_argmin(Centroids, X_train), [[] for realLabel in range(realLabels)], [[] for realLabel in range(realLabels)], len(unique(Labels))
    Trace = [[] for Cluster in range(numberClusters)]
    
    # Declaring number of datapoints in individual clusters
    for Index, Label in enumerate(Labels):
        for Number in range(numberClusters):
            if Label == Number:
                clustered_data_points[Number].append(Index)
    
    for Cluster in range(numberClusters):
        print(f"No. of items in {Cluster}th cluster: {len(clustered_data_points[Cluster])}")
    print(f"\nAll {sum([len(dataPoints) for dataPoints in clustered_data_points])} training data points are clustered.")

    """
    # Plotting clustered training samples in 3D
    for Cluster in range(numberClusters):
        Members = (clusterIndexes[Cluster])
        Trace[Cluster] = Scatter3d(x = arrX_train[Members, 0], y = arrX_train[Members, 1], z = arrX_train[Members, 2], mode = "markers", marker = dict(size = 2, color = Colors[Cluster]), name = f"Cluster {Cluster}", hoverinfo = "text")
        layout = Layout(title = "3D Scatter Plot")
    iplot(Figure(data = [Trace[0], Trace[1], Trace[2], Trace[3], Trace[4], Trace[5], Trace[6], Trace[7], Trace[8], Trace[9]], layout = layout))

    uniqueLabels = unique(y_pred)
    # Plotting the clusters
    figure(figsize = (8, 8)) 
    for i in unique_labels:
        scatter(arrtestSet["label" == i, 0], arrtestSet["label" == i, 1], label = i)
    scatter(Centroids[:, 0], Centroids[:, 1], marker = "x", s = 169, linewidths = 3, color = "k", zorder = 10) 
    legend() 
    show()
    # Use sklearn function to calculate the nearest data instance 
    # for each cluster center. The function returns the indices of 
    # the nearest instances.
    """
    numberClusters = len(unique(y_pred))
    # Plotting selected centroids by implied algorithm
    Fig, Axes = subplots(1, numberClusters)
    for Centroid in range(numberClusters):
        # Plot the images
        Axes[Centroid].imshow(arrX_train[nearestInstances[Centroid]], cmap = "grey")

        # Styling
        Axes[Centroid].set_yticks([])
        Axes[Centroid].set_xticks([])
    title(f"Centroids\nProvided by {clustererName}")
    savefig(f"Centroids by {clustererName}", dpi = 1200)
    show()

    # Evaluation metrics
    print(f"Model is trained in {Duration} UTC\n\nMeasures:\nAccuracy: {accuracy_score(y_test, y_pred)}\nAdjusted Mutual Information (AMI) Score: {round(adjusted_mutual_info_score(y_test, y_pred) * 100, 2)}\nAdjusted Rand Index (ARI) Score: {round(adjusted_rand_score(y_test, y_pred) * 100, 2)}\nCalinski Harabasz Score: {calinski_harabasz_score(X_test, y_pred)}\nCompleteness Score: {completeness_score(y_test, y_pred)}\nDavies Bouldin Score: {davies_bouldin_score(X_test, y_pred)}\nHomogeneity Score: {homogeneity_score(y_test, y_pred)}\nMutual Info Score: {mutual_info_score(y_test, y_pred)}\nRand Score: {rand_score(y_test, y_pred)}\nSSE of random data's cluster results: {model.inertia_}\nSilhouette Score: {silhouette_score(X_test, y_pred, metric = 'euclidean', sample_size = 300)}\nV Measure: {v_measure_score(y_test, y_pred)}")
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred)).plot()
    title(Title)
    savefig(Title, dpi = 1200)
    show()

# Compute the clustering using DBSCAN
print("\nDensity-Based Spatial Clustering of Applications with Noise (DBSCAN) Clustering Algorithm")
Title = "DBSCAN Clustering Algorithm Confusion Matrix"
startTime = time()
model = DBSCAN(eps = 0.025, metric = "euclidean")
model.fit(X_train)
y_pred = model.fit_predict(X_test)
Duration = time() - startTime
# Evaluation metrics
print(f"Model is trained in {Duration} UTC\n\nMeasures:\nAccuracy: {accuracy_score(y_test, y_pred)}\nAdjusted Mutual Information (AMI) Score: {round(adjusted_mutual_info_score(y_test, y_pred) * 100, 2)}\nAdjusted Rand Index (ARI) Score: {round(adjusted_rand_score(y_test, y_pred) * 100, 2)}\nCompleteness Score: {completeness_score(y_test, y_pred)}\nHomogeneity Score: {homogeneity_score(y_test, y_pred)}\nMutual Info Score: {mutual_info_score(y_test, y_pred)}\nRand Score: {rand_score(y_test, y_pred)}\nV Measure: {v_measure_score(y_test, y_pred)}")
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred)).plot()
title(Title)
savefig(Title, dpi = 1200)
show()

# Compute the clustering using HDBSCAN
print("\nHierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) Clustering Algorithm")
Title = "HDBSCAN Clustering Algorithm Confusion Matrix"
startTime = time()
model = HDBSCAN(algorithm = "best", alpha = 1.0, approx_min_span_tree = True, cluster_selection_method = "eom", gen_min_span_tree = False, leaf_size = 40, metric = "euclidean", min_cluster_size = 5, min_samples = 50)
model.fit(X_train)
y_pred = model.fit_predict(X_test)
Duration = time() - startTime
# Evaluation metrics
print(f"Model is trained in {Duration} UTC\n\nMeasures:\nAccuracy: {accuracy_score(y_test, y_pred)}\nAdjusted Mutual Information (AMI) Score: {round(adjusted_mutual_info_score(y_test, y_pred) * 100, 2)}\nAdjusted Rand Index (ARI) Score: {round(adjusted_rand_score(y_test, y_pred) * 100, 2)}\nCompleteness Score: {completeness_score(y_test, y_pred)}\nHomogeneity Score: {homogeneity_score(y_test, y_pred)}\nMutual Info Score: {mutual_info_score(y_test, y_pred)}\nRand Score: {rand_score(y_test, y_pred)}\nV Measure: {v_measure_score(y_test, y_pred)}")
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred)).plot()
title(Title)
savefig(Title, dpi = 1200)
show()

print("THE-END")
"""
# Silhouette cluster evaluation.
model = clusteval(evaluate = "silhouette")
# In case of using dbindex, it is best to clip the maximum number of clusters to avoid finding local minima.
model = clusteval(evaluate = "dbindex", max_clust = 10)
# Derivative method.
model = clusteval(evaluate = "derivative")
# DBscan method.
model = clusteval(cluster = "dbscan", params_dbscan = {"epsres": 100, "norm": True})

Results = model.fit(X_train)
# Clustering labels
print(Results["labx"])

model.plot()
model.plot_silhouette()
model.scatter()
model.dendrogram()

model = clusteval(cluster = "hdbscan")
# Evaluate
Results = model.fit(X_train)
print(Results)
Labels = Results["labx"]
print(Labels)

# Make plot of the evaluation
model.plot()
# Make scatter plot using the first two coordinates. 
model.scatter(X_train)

# Compute the enrichment of the cluster labels with the dataframe df
print(enrichment(dfX_train, Labels))
"""
