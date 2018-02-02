// writen by WangJin  2018/1/9
//利用Logistic算法实现回归梯度上升优化算法

#include <iostream>
#include <vector>
#include <cmath>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

class LogGrad{
	MatrixXf dataMatrix; //数据集
	MatrixXf classLabels; //用于分类的标签
	MatrixXf weight; //最佳回归系数

public:
	//构造函数:向量转矩阵
	LogGrad(vector<vector<float>> dataSet, vector<float> labels){
		int row = dataSet.size(), col = dataSet[0].size();
		int labelSize = labels.size();
		dataMatrix.resize(row, col+1);
		classLabels.resize(labelSize, 1);
		if (row != labelSize)
			cout << "Please check weather the size of dataSet and the size of labels are equal" << endl;
		for (int i = 0; i < row; i++){
			classLabels(i, 0) = labels[i];
			dataMatrix(i, 0) = 1.0;
			for (int j = 0; j < col; j++){
				dataMatrix(i, j+1) = dataSet[i][j];
			}
		}
	}

	//回归上升梯度算法计算最佳回归系数
	vector<float> gradAscent(int numIter){
		vector<float> weightVec; //回归系数的向量表示
		int m = dataMatrix.rows(), n = dataMatrix.cols();
		float alpha = 0.01;
		weight.setOnes(n, 1); //（3×1列的矩阵）
		for (int k = 0; k < numIter; k++){
			MatrixXf h = sigmoidf(dataMatrix*weight);
			cout << h << endl; //调试语句
			MatrixXf error = classLabels - h;
			weight = weight + alpha*dataMatrix.transpose()*error;
		}
		weightVec.resize(n);
		for (int i = 0; i < n; i++)
			weightVec[i] = weight(i, 0);
		return weightVec;
	}

	//输出最佳回归系数
	void showWeight(){
		for (int i = 0; i < weight.rows(); i++)
			cout << weight(i, 0) << " ";
		cout << endl;
	}

private:
	//sigmoid()函数:输入一个矩阵，输出对矩阵元素的sigmoid（）函数值矩阵
	MatrixXf sigmoidf(MatrixXf inX){
		MatrixXf resultInx;
		int m = inX.rows(), n = inX.cols();
		resultInx.resize(m, n);
		for (int i = 0; i < m;i++)
			for (int j = 0; j < n; j++)
				resultInx(i, j) = (float)1.0 / (float)(1.0 + exp(-inX(i, j)));
		return resultInx;
	}

};