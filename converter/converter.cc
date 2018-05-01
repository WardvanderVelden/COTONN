#include <iostream>
#include <string>
#include <array>

#include "scots.hh"

const int state_dim = 2; // need to be read in as well

void print(std::string text) {
	std::cout << text << std::endl;
}

int main() {
	std::string controller_filename = "../controllers/dcdc_bdd/controller";
	std::string name = "plainController";

	Cudd manager;
	BDD bdd;
	scots::SymbolicSet controller;

	// read controller using SCOTS
	if(!read_from_file(manager, controller, bdd, controller_filename)) {
		print("Could not read controller from: " + controller_filename);
		return 0;
	}

	// find state space ids and corresponding input space ids
	auto ids = controller.bdd_to_id(manager, bdd); // get all state space ids that are encoded in the bdd
	auto length = ids.size();

	print("ID's encoded in the BDD: " + std::to_string(length));

	for(unsigned int i = 0; i < length; i++) {
		auto ss_id = ids[i];
		std::array<double, state_dim> ss_x;
		controller.itox(ss_id, ss_x);

		auto is_x = controller.restriction(manager, bdd, ss_x);

		if(i%100 == 0) {
			print("ss_id: " + std::to_string(ss_id) + " is_id: " + std::to_string(is_x[0]));
		}
	}

	return 1;
}