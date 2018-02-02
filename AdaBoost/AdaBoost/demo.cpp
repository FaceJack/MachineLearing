#include "adaBoost.h"
#include "readFile.h"
#include "readTestFile.h"

int main(int argc,char* argv[]){
	ReadFile file = ReadFile(argv[1]);
	ReadTestFile file1 = ReadTestFile(argv[2]);
	AdaBoost AdaB = AdaBoost(file.getDataSet(), file.getLabels());
	AdaB.adaBoostTrainDS(30);
	AdaB.adaClassify(file1.getDataSet());

	return 0;
}