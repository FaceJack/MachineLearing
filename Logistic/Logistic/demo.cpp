#include "logistic.h"
#include "readFile.h"
#include "logGrad.h"

int main(int argc, char* argv[]){
	
	//从马的疝气病症预测马的死亡率
	ReadFile readfile = ReadFile(argv[1]);
	ReadFile testfile = ReadFile(argv[2]);
	Logistic sigLog = Logistic(readfile.getDataSet(), readfile.getLabels());
	vector<float> weights = sigLog.stocGradAscentl(150);
	for (int i = 0; i < weights.size(); i++)
		cout << weights[i] << " ";
	cout << endl;
	//测试分类的准确率
	cout << sigLog.colicTest(testfile.getDataSet(), testfile.getLabels());


	//利用梯度上升算法的求解最佳回归系数
	ReadFile testfile = ReadFile(argv[1]);
	LogGrad logGrad = LogGrad(testfile.getDataSet(), testfile.getLabels());
	logGrad.gradAscent(500);
	logGrad.showWeight();


	return 0;
}