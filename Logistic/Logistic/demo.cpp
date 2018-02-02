#include "logistic.h"
#include "readFile.h"
#include "logGrad.h"

int main(int argc, char* argv[]){
	
	//�����������֢Ԥ�����������
	ReadFile readfile = ReadFile(argv[1]);
	ReadFile testfile = ReadFile(argv[2]);
	Logistic sigLog = Logistic(readfile.getDataSet(), readfile.getLabels());
	vector<float> weights = sigLog.stocGradAscentl(150);
	for (int i = 0; i < weights.size(); i++)
		cout << weights[i] << " ";
	cout << endl;
	//���Է����׼ȷ��
	cout << sigLog.colicTest(testfile.getDataSet(), testfile.getLabels());


	//�����ݶ������㷨�������ѻع�ϵ��
	ReadFile testfile = ReadFile(argv[1]);
	LogGrad logGrad = LogGrad(testfile.getDataSet(), testfile.getLabels());
	logGrad.gradAscent(500);
	logGrad.showWeight();


	return 0;
}