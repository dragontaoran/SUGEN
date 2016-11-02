#include "sugen_utils.h"
#include <ctime>

int main(int argc, char *argv[]) {

	SUGEN sugen_item;
	time_t now = time(NULL);
	
	sugen_item.CommandLineArgs_(argc, argv);
	
	sugen_item.FO_log_.open(sugen_item.FN_log_.c_str(), std::fstream::app);
	
	sugen_item.InputData_();
	sugen_item.Analysis_();
	
	now = time(NULL);
	sugen_item.FO_log_ << "Program successfully finishes at " << asctime(localtime(&now)) << endl;
	sugen_item.FO_log_.close();

	return 0;
}
