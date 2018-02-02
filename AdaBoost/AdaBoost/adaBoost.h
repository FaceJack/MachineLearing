//writen by WangJin 2018/1/16
//C++AdaBoost������

#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <cmath>
#include "Eigen/Dense"


using namespace std;
using namespace Eigen;

class AdaBoost{
	struct bestStump{
		int dim;
		float thresh;
		string inequal;
		float alpha;

		bestStump(){
			dim = -1;
			thresh = 0.0;
			inequal = "";
			alpha = 0.0;
		}
	};
private:
	MatrixXf dataMatIn; //���ݼ�
	MatrixXf classLabels; //��ǩ
	int m; //������Ŀ
	int n; //������������Ŀ
	vector<bestStump> weakClassArr; //�洢��������

public:
	//���캯��
	AdaBoost(vector<vector<float>> dataSet, vector<float> labels);

	//ͨ����ֵ�Ƚ϶����ݽ��з��ࣨ��ѵ������dataMatIn���з��ࣩ
	MatrixXf stumpClassify(int dimen, float threshVal, string threshIneq);

	//����stumpClassify�������Բ�������testData���з��ࣩ
	MatrixXf stumpClassify(MatrixXf testData ,int dimen, float threshVal, string threshIneq);

	//������������ɺ���
	float buildStump(MatrixXf D, MatrixXf &predictedVals, bestStump &retBestStump);

	//���ڵ����������AdaBoostѵ������
	void adaBoostTrainDS(int numIter);

	//AdaBoost���ຯ��
	MatrixXf adaClassify(vector<vector<float>> dataTest);

private:
	float sign(float aggClassEs);
};