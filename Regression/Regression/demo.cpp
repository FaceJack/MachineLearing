#include "readFile.h"
#include "regression.h"

int main(int argc, char* argv[]){
	ReadFile readFile = ReadFile(argv[1]);
	Regression<float> reg = Regression<float>(readFile.getDataSet(), readFile.getLabels());
	reg.testTrainingSet();
	
	return 0;
}