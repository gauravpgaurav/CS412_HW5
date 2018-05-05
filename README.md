# CS412_HW5
This project aims to predict a person’s “empathy” on a scale from 1 to 5 using any of the other attributes in the dataset. From this rating, student volunteers can be recruited to help Alzheimer’s patients at a non-profit organization.

## 1. The Dataset & Preprocessing

The dataset used in this project is the [Young People Survey](https://www.kaggle.com/miroslavsabo/young-people-survey/) dataset, the data consists of 1010 responses of people, regarding 150 categories like - Music preferences, Personality traits, view on life & opinions etc. Since the data also contains 11 categorical columns, they were needed to preprocessed to numerical values. Similarly, specific data values were empty across all categories; these values are necessary to be filled to proceed with the process.

## 2. The Solution

### A.) The ML Solution
After cleaning the data, we could proceed to use the attribute values to predict the Empathy of any person. After training, the data on various classification models like - decision tree classifier, random forest classifier, SVM, voting classifier etc., the generated models underwent evaluation.
### B.) Steps to Run
* Clone/download the project
* Ensure following packages are already installed -
    * pandas
    * numpy
    * sklearn
* Create a ```data``` directory within the project directory
* Download and save the dataset in the ```data``` directory
* Run python file ```svm.py```

### C.) The Result
SVM performed the best amongst all the classifiers, with an average accuracy of 47.03%. On performing bagging on SVM accuracy dropped to 37.22%, so it wasn’t considered in the final model. However, the accuracy is still limited and can be improved via better feature selection.

## 3. The Experiment Process

### (a) Experiment Setup
Spyder IDE was solely used for development due to its simple interface and easy debugging tools.

### (b) The Evaluation Process
For the classification process, the data was split into three categories (training, development and testing) in the ratio 60:20:20. Accuracy was used as a parameter to evaluate classifiers. All the classification models were trained using the training data, tuned using the development data and a final accuracy was calculated on the test data.