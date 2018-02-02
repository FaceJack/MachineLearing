#include <iostream>
#include <vector>

#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

template<typename T>
class Regression{
private:
	MatrixXf xMat;
	MatrixXf yMat;
	MatrixXf ws;
	int m, n; //ѵ����������������

public:
	Regression(vector<vector<T>> xVec, vector<T> yVec){
		m = xVec.size();
		n = xVec[0].size();
		xMat.resize(m, n);
		yMat.resize(m, 1);
		for (int i = 0; i < m; i++){
			yMat(i, 0) = yVec[i];
			for (int j = 0; j < n; j++)
				xMat(i, j) = xVec[i][j];
		}
		MatrixXf xTx = xMat.transpose()*xMat;
		if (xTx.determinant() == 0.0){
			cout << "This matrix is singular,can not do inverse" << endl;
			system("pause"); //�ڴ˴���ͣ
		}
		ws = xTx.inverse()*(xMat.transpose()*yMat);
	}

	//��ѵ�����ϵ����ݽ��в��ԣ�������
	void testTrainingSet(){
		MatrixXf errorYMat;
		MatrixXf preYMat = xMat*ws;
		errorYMat = preYMat - yMat;
		cout << errorYMat;
	}
};