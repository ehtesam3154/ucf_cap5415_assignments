from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np

# #import digit dataset 
# data, target = load_digits(return_X_y=True)
# #split train and tes (500 test samples from 1800 samples s0 27.778%)
# X_train, X_test, y_train, y_test = train_test_split(data, target,
#                                                     test_size = 0.275, shuffle = True, 
#                                                     stratify= target) 



# import digit dataset
data, target = load_digits(return_X_y=True)

# get unique classes
classes = np.unique(target)

# initialize lists to store selected samples
selected_data = []
selected_target = []

# randomly pick 50 samples from each class
for class_label in classes:
    # get indices of samples for the current class
    class_indices = np.where(target == class_label)[0]
    
    # randomly shuffle indices
    np.random.shuffle(class_indices)
    
    # select the first 50 samples from the shuffled indices
    selected_data.append(data[class_indices[:50]])
    selected_target.append(target[class_indices[:50]])

# concatenate the selected samples
selected_data = np.concatenate(selected_data, axis=0)
selected_target = np.concatenate(selected_target, axis=0)

#split train and tes (500 test samples from 1800 samples s0 27.778%)
X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_target,
                                                    test_size=0.275, shuffle=True,
                                                    stratify=selected_target)


#instantiate knn classifier with n_neighbors=2
clf = KNeighborsClassifier(n_neighbors=2, p=2)
#train classifier
clf.fit(X_train, y_train)
#get prediction with the trained model
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

#output results
print('KNN result for n_neighbours = 2')
print('Train accuracy:', accuracy_score(y_train_pred, y_train))
print('Test accuracy:', accuracy_score(y_test_pred, y_test))
print('\n')

#work with different values of n_neighbours and save accuracy values in lists
train_acc_list = []
test_acc_list = []
n_list = [3,4,5,6,7,8,9,10,11,12,13,14,15, 30, 40] 

#experiment with different n_neighbours
for n in n_list:
    #initiate knn classifier
    clf = KNeighborsClassifier(n_neighbors=n, p=2)
    #train classifier
    clf.fit(X_train, y_train)
    #get prediction with the trained model
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    #claculate and store train and test accuracies
    train_acc = accuracy_score(y_train_pred, y_train) * 100
    train_acc_list.append(train_acc)
    test_acc = accuracy_score(y_test_pred, y_test) * 100
    test_acc_list.append(test_acc)

    #print results
    print(f'KNN result for n_neighbours = {n}')
    print("train accuracy:", train_acc)
    print("Test accuracy:", test_acc)

#creat subplots
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#set figure title
fig.suptitle('variance in result of knn classifier',
             fontsize = 16,
             y = 0.95)

#set figure facecolor to white
fig.set_facecolor('white')
#plot train and test accuracies
ax.plot(n_list, train_acc_list,
        marker = 'o',
        label = 'train accuracy')

ax.plot(n_list, test_acc_list,
        marker = 'x',
        label = 'test accuracy')


ax.set_xlabel(r'$n_{neighbors}$', size=10)
ax.set_ylabel(r'$Accuracy (\%)$', size=10)
ax.legend(fontsize=12)
ax.grid(linestyle='dotted')
plt.show()