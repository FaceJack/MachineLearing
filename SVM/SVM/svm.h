// writen by WangJin 2018/1/10
// Ŀ�꣺ʵ��֧��������
// ֧����������ͷ�ļ�

#include <vector>
#include <iostream>
#include <string>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

class SVM{
private:
	MatrixXf dataMatIn; //ѵ�����ݼ�
	MatrixXf classLabels; //����ı�ǩ
	float C; //����
	float toler; //�ݴ���
	int m, n; //ѵ����������������
	MatrixXf alpha; //�����ľ���
	float b; //�����ĳ���
	MatrixXf eCache; //�������

public:
	//���캯��
	SVM(vector<vector<float>> dataSet, vector<float> labels, float C, float toler);

	//Platt SMO�㷨:����1����ʾ��һ��alpha���£�����0����ʾû���κ�һ��alpha�����仯
	int innerL(int i);

	//SVM�㷨��֧����������ʵ��
	void smoP(int maxIter, pair<string, int> kTup);

	//���������
	void showResult();

	//����w
	MatrixXf calcWs();

	//����һ�����ݣ���֤��������һ��
	int testWhichClass(MatrixXf testData);

private:
	//�������Ek��Ԥ��ֵ����ʵֵ֮��Ĳ�ֵ
	float calcEk(int k);

	//ѡ��ڶ���alpha
	pair<int, float> selectJ(int i, float Ei);

	//���º������������������eCache��
	void updateEk(int k);

	//��[0,m)��������ѡ��һ����Ϊi����
	int selectJrand(int i, int m);

	//���ڵ�������H����С��L��alphaֵ aj����[L,H]������
	float clipAlpha(float aj, float H, float L);

};