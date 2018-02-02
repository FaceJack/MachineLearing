// writen by WangJin  2018/1/9
//����Logistic�㷨ʵ�ֻع��ݶ������Ż��㷨

#include <iostream>
#include <vector>
#include <cmath>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

class LogGrad{
	MatrixXf dataMatrix; //���ݼ�
	MatrixXf classLabels; //���ڷ���ı�ǩ
	MatrixXf weight; //��ѻع�ϵ��

public:
	//���캯��:����ת����
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

	//�ع������ݶ��㷨������ѻع�ϵ��
	vector<float> gradAscent(int numIter){
		vector<float> weightVec; //�ع�ϵ����������ʾ
		int m = dataMatrix.rows(), n = dataMatrix.cols();
		float alpha = 0.01;
		weight.setOnes(n, 1); //��3��1�еľ���
		for (int k = 0; k < numIter; k++){
			MatrixXf h = sigmoidf(dataMatrix*weight);
			cout << h << endl; //�������
			MatrixXf error = classLabels - h;
			weight = weight + alpha*dataMatrix.transpose()*error;
		}
		weightVec.resize(n);
		for (int i = 0; i < n; i++)
			weightVec[i] = weight(i, 0);
		return weightVec;
	}

	//�����ѻع�ϵ��
	void showWeight(){
		for (int i = 0; i < weight.rows(); i++)
			cout << weight(i, 0) << " ";
		cout << endl;
	}

private:
	//sigmoid()����:����һ����������Ծ���Ԫ�ص�sigmoid��������ֵ����
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