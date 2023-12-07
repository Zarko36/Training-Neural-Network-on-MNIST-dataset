import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
feature_combinations = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
color = ['green','yellow','red']
markers = ['o','o','o']
plt.figure(figsize=(15,10))
for i, (x_feature, y_feature) in enumerate(feature_combinations, 1):
    plt.subplot(2,3,i)
    for species in range(3):
        subset = iris_df[iris_df['species'] == species]
        plt.scatter(subset[iris_df.columns[x_feature]], subset[iris_df.columns[y_feature]], marker=markers[species], color=color[species], label=iris.target_names[species])
    plt.xlabel(iris_df.columns[x_feature])
    plt.ylabel(iris_df.columns[y_feature])
    plt.legend()
plt.tight_layout()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('species', axis=1), iris_df['species'], test_size = 0.3, random_state = 42, stratify=iris_df['species'])
X_train.shape, X_test.shape, y_train.shape, y_test.shape
def calculate_accuracy_for_specific_pairs(feature1, feature2):
    train_data_selected = X_train.iloc[:,[feature1, feature2]]
    test_data_selected = X_test.iloc[:,[feature1, feature2]]
    model_virginica_versicolor = LinearRegression()
    model_versicolor_setosa = LinearRegression()
    train_filter_vv = (y_train == 2) | (y_train == 1)
    train_data_vv = train_data_selected[train_filter_vv]
    train_labels_vv = y_train[train_filter_vv]
    train_labels_vv_binary = (train_labels_vv == 2).astype(int)
    model_virginica_versicolor.fit(train_data_vv, train_labels_vv_binary)
    train_filter_vs = (y_train == 1) | (y_train == 0)
    train_data_vs = train_data_selected[train_filter_vs]
    train_labels_vs = y_train[train_filter_vs]
    train_labels_vs_binary = (train_labels_vs == 0).astype(int)
    model_versicolor_setosa.fit(train_data_vs, train_labels_vs_binary)
    predictions_virginica_versicolor = model_virginica_versicolor.predict(test_data_selected)
    predictions_versicolor_setosa = model_versicolor_setosa.predict(test_data_selected)
    final_predictions = np.zeros_like(predictions_virginica_versicolor, dtype=int) + 1
    final_predictions[predictions_virginica_versicolor > 0.5] = 2 # Virginica
    final_predictions[predictions_versicolor_setosa > 0.5] = 0 # Setosa
    accuracy = accuracy_score(y_test, final_predictions)
    return (model_virginica_versicolor.coef_, model_virginica_versicolor.intercept_, model_versicolor_setosa.coef_, model_versicolor_setosa.intercept_, accuracy)
corrected_results = []
for x_feature, y_feature in feature_combinations:
    coeffs_vv, intercept_vv, coeffs_vs, intercept_vs, acc = calculate_accuracy_for_specific_pairs(x_feature, y_feature)
    feature1_name = iris.feature_names[x_feature]
    feature2_name = iris.feature_names[y_feature]
    corrected_results.append((f'Features {feature1_name} & {feature2_name}', coeffs_vv, intercept_vv, coeffs_vs, intercept_vs, acc))
for result in corrected_results:
    print(f'{result[0]}:')
    print(f'Virginica vs Versicolor: Coefficients: {result[1]}, Intercept: {result[2]}')
    print(f'Versicolor vs Setosa: Coefficients: {result[3]}, Intercept: {result[4]}')
    print(f'Test Accuracy for whole set: {result[5]:.2f}\n')