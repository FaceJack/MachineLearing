//writen by WangJin  2018/1/8 
//Ŀ�꣺ʵ��Logistic����ݶ������ع��㷨����Ŀ����з���

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

class Logistic{
private:
	MatrixXf dataMatrix; //���ݼ�
	MatrixXf classLabels; //���ڷ���ı�ǩ
	MatrixXf weight; //��ѻع�ϵ��

public:
	//���캯��:����ת����
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

	//����ݶ������㷨:������ѻع�ϵ��
	vector<float> stocGradAscentl(int numIter){
		int m = dataMatrix.rows(), n = dataMatrix.cols();
		weight.setOnes(1, n); //��ѻع�ϵ���ľ�����ʽ
		vector<float> weights; //��ѻع�ϵ����������ʾ��ʽ
		weights.resize(n);
		srand((unsigned)time(NULL));
		for (int j = 0; j < numIter; j++){
			int dataIndex = m;
			for (int i = 0; i < m; i++){
				float alpha = 4.0 / ((1.0 + i + j) + 0.01);
				int randIndex = (int)rand()%dataIndex; //����һ��[0,dataIndex)�ڵ��������
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

	//����������ݣ�����������ȷ��
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
	//sigmoid()����
	float sigmoid(float inX){
		return (float)1.0 / ((float)1.0 + (float)exp(-inX));
	}

	//����logistic�����㷨���з���
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