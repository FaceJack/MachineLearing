#include <string>
#include "kNN.h"

int main(){
	vector<vector<float>> dataSet;
	dataSet.resize(4);
	for (int i = 0; i < dataSet.size(); i++)
		dataSet[i].resize(2);
	dataSet[0][0] = 1;
	dataSet[0][1] = 1.1;
	dataSet[1][0] = 1;
	dataSet[1][1] = 1;
	dataSet[2][0] = 0;
	dataSet[2][1] = 0;
	dataSet[3][0] = 0;
	dataSet[3][1] = 0.1;

	vector<string> labels;
	labels.resize(4);
	labels[0] = "A";
	labels[1] = "A";
	labels[2] = "B"; 
	labels[3] = "B";

	kNN<string> kNNexample = kNN<string>(dataSet, labels);
	vector<float> inX;
	inX.resize(2);
	inX[0] = 1;
	inX[1] = 1;
	string result = kNNexample.classify(inX, 3);
	cout << result << endl;

	return 0;
}