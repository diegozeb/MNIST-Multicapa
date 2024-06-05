#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>

using namespace std;
using namespace Eigen;


vector<vector<float>> readCSV(const string& filename) 
{ 
    vector<vector<float>> data;
    ifstream file(filename);

    if (!file.is_open()) 
    {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        return data;
    }

    string line;
    while (getline(file, line)) 
    {
        vector<float> row;
        stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) 
        {
            try 
            {
                row.push_back(stof(cell));
            }

            catch (const invalid_argument& e) 
            {
                cerr << "Error: No se pudo convertir la celda a número: " << cell << endl;
                return vector<vector<float>>();
            }
        }
        data.push_back(row);
    }
    return data;
}


MatrixXd sigmoid(const MatrixXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}


MatrixXd sigmoidDerivative(const MatrixXd& x) {
    return x.array() * (1.0 - x.array());
}


class PerceptronMulticapa 
{
public:
    PerceptronMulticapa(int inputSize, int hiddenSize, int outputSize);
    void train(const MatrixXd& X, const MatrixXd& y, int epochs, double learningRate);
    VectorXd predict(const VectorXd& x);

private:
    MatrixXd W1, W2;
    VectorXd b1, b2;
    MatrixXd forward(const MatrixXd& X);
    void backward(const MatrixXd& X, const MatrixXd& y, const MatrixXd& output, double learningRate);
    double calculateAccuracy(const MatrixXd& X, const MatrixXd& y);
};

PerceptronMulticapa::PerceptronMulticapa(int inputSize, int hiddenSize, int outputSize) 
{
    W1 = MatrixXd::Random(hiddenSize, inputSize);
    W2 = MatrixXd::Random(outputSize, hiddenSize);
    b1 = VectorXd::Random(hiddenSize);
    b2 = VectorXd::Random(outputSize);
}

void PerceptronMulticapa::train(const MatrixXd& X, const MatrixXd& y, int epochs, double learningRate) 
{
    for (int i = 0; i < epochs; ++i) 
    {
        MatrixXd output = forward(X);
        backward(X, y, output, learningRate);

        double accuracy = calculateAccuracy(X, y);
        cout << "Epoch " << i + 1 << " completed. Accuracy: " << accuracy * 100 << "%" << endl;
    }
}

MatrixXd PerceptronMulticapa::forward(const MatrixXd& X) 
{
    MatrixXd Z1 = (W1 * X.transpose()).colwise() + b1;
    MatrixXd A1 = sigmoid(Z1);
    MatrixXd Z2 = (W2 * A1).colwise() + b2;
    return sigmoid(Z2);
}

void PerceptronMulticapa::backward(const MatrixXd& X, const MatrixXd& y, const MatrixXd& output, double learningRate) 
{
    MatrixXd error = output - y.transpose();
    MatrixXd delta2 = error.cwiseProduct(sigmoidDerivative(output));
    MatrixXd A1 = sigmoid((W1 * X.transpose()).colwise() + b1); // Computar A1 una vez para usarla después
    MatrixXd delta1 = (W2.transpose() * delta2).cwiseProduct(sigmoidDerivative(A1));

    W2 -= learningRate * delta2 * A1.transpose();
    b2 -= learningRate * delta2.rowwise().sum();

    W1 -= learningRate * delta1 * X;
    b1 -= learningRate * delta1.rowwise().sum();
}

VectorXd PerceptronMulticapa::predict(const VectorXd& x) 
{
    MatrixXd Z1 = (W1 * x).colwise() + b1;
    MatrixXd A1 = sigmoid(Z1);
    MatrixXd Z2 = (W2 * A1).colwise() + b2;
    return sigmoid(Z2).col(0);
}

double PerceptronMulticapa::calculateAccuracy(const MatrixXd& X, const MatrixXd& y) 
{
    int numSamples = X.rows();
    int numCorrect = 0;

    for (int i = 0; i < numSamples; ++i) {
        VectorXd prediction = predict(X.row(i).transpose());
        int predictedLabel = distance(prediction.data(), max_element(prediction.data(), prediction.data() + prediction.size()));
        int actualLabel = distance(y.row(i).data(), max_element(y.row(i).data(), y.row(i).data() + y.row(i).size()));

        if (predictedLabel == actualLabel) {
            numCorrect++;
        }
    }

    return static_cast<double>(numCorrect) / numSamples;
}

int main() 
{
    string trainFile = "C:/Users/diego/source/repos/Perceptron Multicapa/Perceptron Multicapa/x64/Debug/mnist_train.csv";
    string testFile = "C:/Users/diego/source/repos/Perceptron Multicapa/Perceptron Multicapa/x64/Debug/mnist_test.csv";

    vector<vector<float>> trainData = readCSV(trainFile);
    vector<vector<float>> testData = readCSV(testFile);

    if (trainData.empty() || testData.empty()) 
    {
        cerr << "Error: No se pudo cargar los datos de los archivos CSV." << endl;
        return -1;
    }

    int numTrain = trainData.size();
    int numFeatures = trainData[0].size() - 1;

    MatrixXd X_train(numTrain, numFeatures);
    MatrixXd y_train = MatrixXd::Zero(numTrain, 10);

    for (int i = 0; i < numTrain; ++i) 
    {
        if (trainData[i].size() != numFeatures + 1) 
        {
            cerr << "Error: La fila " << i << " en el conjunto de entrenamiento no tiene el número correcto de columnas." << endl;
            return -1;
        }

        for (int j = 1; j < trainData[i].size(); ++j) 
        {
            X_train(i, j - 1) = trainData[i][j] / 255.0;
        }

        int label = static_cast<int>(trainData[i][0]);

        if (label >= 0 && label < 10) 
        { 
            y_train(i, label) = 1.0;
        }

        else 
        {
            cerr << "Error: Etiqueta fuera de rango: " << label << " en la fila " << i << endl;
            return -1;
        }
    }

    PerceptronMulticapa mlp(numFeatures, 2, 10);
    mlp.train(X_train, y_train, 1000, 0.1);

   
    int numTest = testData.size();
    MatrixXd X_test(numTest, numFeatures);
    VectorXi y_test(numTest);

    for (int i = 0; i < numTest; ++i) 
    {
        if (testData[i].size() != numFeatures + 1) 
        {
            cerr << "Error: La fila " << i << " en el conjunto de prueba no tiene el número correcto de columnas." << endl;
            return -1;
        }

        for (int j = 1; j < testData[i].size(); ++j) 
        {
            X_test(i, j - 1) = testData[i][j] / 255.0;
        }

        y_test(i) = static_cast<int>(testData[i][0]);
    }

    int correct = 0;
    for (int i = 0; i < numTest; ++i) 
    {
        VectorXd prediction = mlp.predict(X_test.row(i).transpose());
        int predictedLabel = distance(prediction.data(), max_element(prediction.data(), prediction.data() + prediction.size()));

        if (predictedLabel == y_test(i)) 
        {
            correct++;
        }
    }

    cout << "Precisión: " << (double(correct) / numTest) * 100 << "%" << endl;

    return 0;
}
