#include <iostream>
#include <string>
#include <array>

#include "scots.hh"

/*
	This parameters need to be changed in order to operate the converter
*/
const std::string controller_filename = "../../Scots2C/controllers/dcdc_bdd/controller";
const std::string new_controller_filename = "../../Scots2C/controllers/dcdc_bdd/newStaticController";
const int state_dim = 2;
const int input_dim = 1;


/*
	TODO:
	- Make sure behaviour is absolutely the same as bdd controller for v1.01
	- Put functions in separate class
	- Order write file function
	- (Make write file function independent of lib (so lib can be updated without breaking the converter))
*/
const bool debug_mode = false;

using state_type = std::array<double,state_dim>;
using input_type = std::array<double,input_dim>;

struct ControllerPoint {
	unsigned int state; // state id
	unsigned int input; // input id
	ControllerPoint(unsigned int s, unsigned int i) : state(s), input(i) { }
};

bool sort_winning_domain(ControllerPoint a, ControllerPoint b) {
	return (a.state < b.state);
}

void print(std::string text) {
	std::cout << text << std::endl;
}

// Get bounds and eta's
void get_bounds_and_etas(scots::SymbolicSet &controller, state_type &ss_lb, state_type &ss_ub, state_type &ss_eta, input_type &is_lb, input_type &is_ub, input_type &is_eta) {
	auto dim = controller.get_dim();
	auto eta = controller.get_eta();
	auto lower_bound = controller.get_lower_left();
	auto upper_bound = controller.get_upper_right();

	// state_space
	for(int i = 0; i < state_dim; i++) {
		ss_lb[i] = lower_bound[i];
		ss_ub[i] = upper_bound[i];
		ss_eta[i] = eta[i];
	}

	// input space
	for(int i = 0; i < (dim - state_dim); i++) {
		is_lb[i] = lower_bound[i + state_dim];
		is_ub[i] = upper_bound[i + state_dim];
		is_eta[i] = eta[i + state_dim];
	}
}

// Find and return the uniform grids that are associated with the controller
void get_uniform_grids(scots::SymbolicSet &controller, scots::UniformGrid &ss, scots::UniformGrid &is) {
	state_type ss_lb, ss_ub, ss_eta;
	input_type is_lb, is_ub, is_eta;
	get_bounds_and_etas(controller, ss_lb, ss_ub, ss_eta, is_lb, is_ub, is_eta);
	ss = scots::UniformGrid(state_dim, ss_lb, ss_ub, ss_eta);
	is = scots::UniformGrid(input_dim, is_lb, is_ub, is_eta);
	if(debug_mode) {
		print("\nState space:");
		ss.print_info();
		print("\nInput space:");
		is.print_info();
	}
}

// Find the winning domain from the controller, bdd, state space and input space
std::vector<ControllerPoint> find_winning_domain(scots::SymbolicSet &controller, Cudd &manager, BDD &bdd, scots::UniformGrid &ss, scots::UniformGrid &is) {
	std::vector<ControllerPoint> winning_domain;

	auto size = ss.size();
	for(unsigned int s = 0; s < size; s++) {
		// read x at state
		state_type x;
		ss.itox(s, x);

		// read input u at that state x
		auto u = controller.restriction(manager, bdd, x);

		// check if state is in winning domain
		if(!u.empty()) {
			// read input id from input x
			auto i = is.xtoi(u);

			// add controllerpoint to winning_domain
			winning_domain.push_back(ControllerPoint(s, i));
		}
	}
	return winning_domain;
}

// Write a new controller using the old symbolic set controller and the newly formatted controller
bool write_new_controller(scots::SymbolicSet &controller, std::vector<ControllerPoint> &winning_domain, scots::UniformGrid &ss, scots::UniformGrid &is, bool append_to_file = false) {
	scots::FileWriter writer(new_controller_filename);
	if(append_to_file) {
        if(!writer.open()) {
            return false;
        }
    } else {
        if(!writer.create()) {
            return false;
        }
    }

    auto eta = controller.get_eta();
    auto lb = controller.get_lower_left();
    auto ub = controller.get_upper_right();

    std::vector<double> ss_eta(state_dim);
    std::vector<double> ss_lb(state_dim);
    std::vector<double> ss_ub(state_dim);

    std::vector<double> is_eta(input_dim);
    std::vector<double> is_lb(input_dim);
    std::vector<double> is_ub(input_dim);

    for(int i = 0; i < state_dim; i++) {
    	ss_eta[i] = eta[i];
    	ss_ub[i] = ub[i];
    	ss_lb[i] = lb[i];
    }

    for(int i = 0; i < (state_dim + input_dim); i++) {
    	is_eta[i] = eta[i+state_dim];
    	is_ub[i] = ub[i+state_dim];
    	is_lb[i] = lb[i+state_dim];
    }

    writer.add_VERSION();
    writer.add_TYPE(SCOTS_SC_TYPE);

    writer.add_TEXT("STATE_SPACE");
    writer.add_TYPE(SCOTS_UG_TYPE);
    writer.add_MEMBER(SCOTS_UG_DIM,state_dim);
    writer.add_VECTOR(SCOTS_UG_ETA,ss_eta);
    writer.add_VECTOR(SCOTS_UG_LOWER_LEFT,ss_lb);
    writer.add_VECTOR(SCOTS_UG_UPPER_RIGHT,ss_ub);

    writer.add_TEXT("INPUT_SPACE");
    writer.add_TYPE(SCOTS_UG_TYPE);
    writer.add_MEMBER(SCOTS_UG_DIM,input_dim);
    writer.add_VECTOR(SCOTS_UG_ETA,is_eta);
    writer.add_VECTOR(SCOTS_UG_LOWER_LEFT,is_lb);
    writer.add_VECTOR(SCOTS_UG_UPPER_RIGHT,is_ub);

    writer.add_PLAIN("#TYPE:WINNINGDOMAIN\n#SCOTS:i (state) j_0 ... j_n (valid inputs)\n#MATRIX:DATA\n");
    writer.add_PLAIN("#BEGIN:" + std::to_string(ss.size()) + " " + std::to_string(is.size()) + "\n");

    for(unsigned int i = 0; i < winning_domain.size(); i++) {
    	writer.add_PLAIN(std::to_string(winning_domain[i].state) + " " + std::to_string(winning_domain[i].input) + "\n");
    }

    writer.add_PLAIN("#END");

    writer.close();

    return true;
}

int main() {
	print("Controller converter v1.01");
	Cudd manager;
	BDD bdd;
	scots::SymbolicSet controller;

	// read controller using SCOTS
	if(!read_from_file(manager, controller, bdd, controller_filename)) {
		print("Could not read controller from: " + controller_filename);
		return 0;
	}

	// init uniform grid state and input space
	scots::UniformGrid ss;
	scots::UniformGrid is;
	get_uniform_grids(controller, ss, is);

	// Find winning domain
	std::vector<ControllerPoint> winning_domain = find_winning_domain(controller, manager, bdd, ss, is);

	// sort winning domain
	std::sort(begin(winning_domain), end(winning_domain), sort_winning_domain);

	// print if in debug mode
	if(debug_mode) {
		for(unsigned int i = 0; i < winning_domain.size(); i += 100) {
			print("SS: " + std::to_string(winning_domain[i].state) + " IS: " + std::to_string(winning_domain[i].input));
		}
	}

	// write out new controller
	if(write_new_controller(controller, winning_domain, ss, is)) {
		print("Converter controller to static controller and written to: " + new_controller_filename + ".scs");
	} else {
		print("An error occured while writing the controller.");
	}

	return 1;
}