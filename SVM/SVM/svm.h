// writen by WangJin 2018/1/10
// 目标：实现支持向量机
// 支持向量机的头文件

#include <vector>
#include <iostream>
#include <string>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

class SVM{
private:
	MatrixXf dataMatIn; //训练数据集
	MatrixXf classLabels; //分类的标签
	float C; //常数
	float toler; //容错率
	int m, n; //训练集的行数和列数
	MatrixXf alpha; //待求解的矩阵
	float b; //带求解的常数
	MatrixXf eCache; //缓存误差

public:
	//构造函数
	SVM(vector<vector<float>> dataSet, vector<float> labels, float C, float toler);

	//Platt SMO算法:返回1，表示有一对alpha更新；返回0，表示没有任何一对alpha发生变化
	int innerL(int i);

	//SVM算法：支持向量机的实现
	void smoP(int maxIter, pair<string, int> kTup);

	//输出计算结果
	void showResult();

	//计算w
	MatrixXf calcWs();

	//输入一组数据，验证其属于哪一类
	int testWhichClass(MatrixXf testData);

private:
	//计算误差Ek：预测值与真实值之间的差值
	float calcEk(int k);

	//选择第二个alpha
	pair<int, float> selectJ(int i, float Ei);

	//更新函数：将计算的误差存入eCache中
	void updateEk(int k);

	//从[0,m)的区间内选择一个不为i的数
	int selectJrand(int i, int m);

	//用于调整大于H或者小于L的alpha值 aj属于[L,H]区间内
	float clipAlpha(float aj, float H, float L);

};