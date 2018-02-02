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
	int m, n; //训练集的行数和列数

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
			system("pause"); //在此处暂停
		}
		ws = xTx.inverse()*(xMat.transpose()*yMat);
	}

	//对训练集上的数据进行测试，输出误差
	void testTrainingSet(){
		MatrixXf errorYMat;
		MatrixXf preYMat = xMat*ws;
		errorYMat = preYMat - yMat;
		cout << errorYMat;
	}
};