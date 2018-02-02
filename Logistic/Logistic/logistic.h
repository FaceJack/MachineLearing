//writen by WangJin  2018/1/8 
//目标：实现Logistic随机梯度上升回归算法，对目标进行分类

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

class Logistic{
private:
	MatrixXf dataMatrix; //数据集
	MatrixXf classLabels; //用于分类的标签
	MatrixXf weight; //最佳回归系数

public:
	//构造函数:向量转矩阵
	Logistic(vector<vector<float>> dataSet, vector<float> labels){
		int row = dataSet.size(), col = dataSet[0].size();
		int labelSize = labels.size();
		dataMatrix.resize(row, col);
		classLabels.resize(labelSize,1);
		if (row != labelSize)
			cout << "Please check weather the size of dataSet and the size of labels are equal" << endl;
		for (int i = 0; i < row; i++){
			classLabels(i,0) = labels[i];
			for (int j = 0; j < col; j++)
				dataMatrix(i, j) = dataSet[i][j];
		}
	}

	//随机梯度上升算法:返回最佳回归系数
	vector<float> stocGradAscentl(int numIter){
		int m = dataMatrix.rows(), n = dataMatrix.cols();
		weight.setOnes(1, n); //最佳回归系数的矩阵形式
		vector<float> weights; //最佳回归系数的向量表示形式
		weights.resize(n);
		srand((unsigned)time(NULL));
		for (int j = 0; j < numIter; j++){
			int dataIndex = m;
			for (int i = 0; i < m; i++){
				float alpha = 4.0 / ((1.0 + i + j) + 0.01);
				int randIndex = (int)rand()%dataIndex; //产生一个[0,dataIndex)内的随机整数
				float inX = (dataMatrix.row(randIndex).transpose()*weight).sum();
				float h = sigmoid(inX);
				float error = classLabels(randIndex, 0) - h;
				weight = weight + alpha*error*dataMatrix.row(randIndex);
				dataIndex--;
			}
		}

		for (int i = 0; i < n; i++)
			weights[i] = weight(0, i);

		return weights;
	}

	//输入测试数据，输出分类的正确率
	float colicTest(vector<vector<float>> testDataSet, vector<float> testLabels){
		int countSize = 0;
		for (int i = 0; i < testDataSet.size(); i++){
			int result = classifyVector(testDataSet[i]);
			if (result != (int)testLabels[i])
				countSize++;
		}
		return (float)countSize / (float)testDataSet.size();
	}

private:
	//sigmoid()函数
	float sigmoid(float inX){
		return (float)1.0 / ((float)1.0 + (float)exp(-inX));
	}

	//利用logistic分类算法进行分类
	int classifyVector(vector<float> inVec){
		MatrixXf inX(21, 1);
		for (int i = 0; i < inVec.size(); i++)
			inX(i, 0) = inVec[i];
		float prod = sigmoid((inX*weight).sum());
		cout << (inX*weight).sum() << " " << prod << endl;
		if (prod>0.5)
			return 1;
		else
			return 0;
	}
};