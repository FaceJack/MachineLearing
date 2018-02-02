//writen by WangJin 2018/1/16
//C++AdaBoost分类器

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
	MatrixXf dataMatIn; //数据集
	MatrixXf classLabels; //标签
	int m; //样本数目
	int n; //样本特征的数目
	vector<bestStump> weakClassArr; //存储弱分类器

public:
	//构造函数
	AdaBoost(vector<vector<float>> dataSet, vector<float> labels);

	//通过阈值比较对数据进行分类（对训练数据dataMatIn进行分类）
	MatrixXf stumpClassify(int dimen, float threshVal, string threshIneq);

	//重载stumpClassify函数（对测试数据testData进行分类）
	MatrixXf stumpClassify(MatrixXf testData ,int dimen, float threshVal, string threshIneq);

	//单层决策树生成函数
	float buildStump(MatrixXf D, MatrixXf &predictedVals, bestStump &retBestStump);

	//基于单层决策树的AdaBoost训练过程
	void adaBoostTrainDS(int numIter);

	//AdaBoost分类函数
	MatrixXf adaClassify(vector<vector<float>> dataTest);

private:
	float sign(float aggClassEs);
};