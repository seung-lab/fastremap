#ifndef __FASTREMAP_CONTOUR_HPP__
#define __FASTREMAP_CONTOUR_HPP__

#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>

namespace fastremap {
namespace contour {

const uint8_t VISITED_BIT = 0b10000;

// voxel connectivity matches cc3d_graphs.hpp 4 connected
// four bits: -y+y-x+x true is passable
enum VCGDirectionCode {
	NONE = 0b0000,
	LEFT = 0b0010,
	RIGHT = 0b0001,
	UP = 0b1000,
	DOWN = 0b0100,
	ANY = 0b1111
};

struct VCGGraph {
	std::vector<uint8_t>& vcg;
	
	int64_t sx;
	int64_t sy;

	VCGGraph(std::vector<uint8_t>& _vcg, int64_t _sx, int64_t _sy) 
		: vcg(_vcg), sx(_sx), sy(_sy) {}

	bool next_contour(int64_t& idx, int64_t& y) {
		int64_t x = idx - sx * y;

		for (; y < sy; y++) {
			for (; x < sx; x++, idx++) {
				// condensing this conditional seems to save 5% in one speed test
				// if (((vcg[idx] & 0b11) < 0b11) && (vcg[idx] & VISIT_COUNT) == 0) {
				// -----
				// check that the next voxel isn't visited and is a barrier

				if ((vcg[idx] & 0b110011) < 0b11 
					|| (x < sx - 1 && (vcg[idx+1] & 0b11110010) == 0b0)) {
					return true;
				}
			}
			x = 0;
		}

		return false;
	}
};

template <typename LABEL>
void create_boundary_map(
	const LABEL* labels,
	std::vector<uint8_t>& boundaries,
	const int64_t sx, const int64_t sy
) {
	for (int64_t y = 0; y < sy; y++) {
		for (int64_t x = 0; x < sx; x++) {
			// assign vertical edges
			if (x > 0 && labels[x + sx * y] == labels[(x-1) + sx * y]) {
				int64_t node_left = (x-1) + sx * y;
				int64_t node_right = x + sx * y;
				boundaries[node_left] |= VCGDirectionCode::RIGHT;
				boundaries[node_right] |= VCGDirectionCode::LEFT;
			}
			// assign horizontal edges
			if (y > 0 && labels[x + sx * y] == labels[x + sx * (y-1)]) {
				int64_t node_up = x + sx * y;
				int64_t node_down = x + sx * (y-1);
				boundaries[node_up] |= VCGDirectionCode::UP;
				boundaries[node_down] |= VCGDirectionCode::DOWN;
			}
		}
	}
}

#define TRY_LEFT if (allowed_dirs & VCGDirectionCode::LEFT) {return VCGDirectionCode::LEFT;}
#define TRY_RIGHT if (allowed_dirs & VCGDirectionCode::RIGHT) {return VCGDirectionCode::RIGHT;}
#define TRY_UP if (allowed_dirs & VCGDirectionCode::UP) {return VCGDirectionCode::UP;}
#define TRY_DOWN if (allowed_dirs & VCGDirectionCode::DOWN) {return VCGDirectionCode::DOWN;}

uint8_t compute_next_move(
	const bool clockwise,
	const uint8_t last_move,
	const uint8_t allowed_dirs
) {
	if (clockwise) {
		if (last_move == VCGDirectionCode::RIGHT) {
			TRY_DOWN
			else TRY_RIGHT
			else TRY_UP
			else TRY_LEFT
		}
		else if (last_move == VCGDirectionCode::LEFT) {
			TRY_UP
			else TRY_LEFT
			else TRY_DOWN
			else TRY_RIGHT
		}
		else if (last_move == VCGDirectionCode::UP) {
			TRY_RIGHT
			else TRY_UP
			else TRY_LEFT
			else TRY_DOWN
		}
		else { // last_move == 'd'
			TRY_LEFT
			else TRY_DOWN
			else TRY_RIGHT
			else TRY_UP
		}
	}
	else {
		if (last_move == VCGDirectionCode::RIGHT) {
			TRY_UP
			else TRY_RIGHT
			else TRY_DOWN
			else TRY_LEFT
		}
		else if (last_move == VCGDirectionCode::LEFT) {
			TRY_DOWN
			else TRY_LEFT
			else TRY_UP
			else TRY_RIGHT
		}
		else if (last_move == VCGDirectionCode::UP) {
			TRY_LEFT
			else TRY_UP
			else TRY_RIGHT
			else TRY_DOWN
		}
		else { // last_move == 'd'
			TRY_RIGHT
			else TRY_DOWN
			else TRY_LEFT
			else TRY_UP
		}
	}

	return VCGDirectionCode::NONE;
}

#undef TRY_UP
#undef TRY_DOWN
#undef TRY_LEFT
#undef TRY_RIGHT

std::vector<std::vector<std::pair<uint16_t, uint16_t>>> 
extract_contours_helper(
	std::vector<uint8_t>& vcg,
	const uint64_t sx, const uint64_t sy
) {
	std::vector<
		std::vector<std::pair<uint16_t, uint16_t>>
	> contours;

	VCGGraph G(vcg, sx, sy);
	for (uint64_t i = 0; i < sx; i++) {
		vcg[i] = vcg[i] & ~VCGDirectionCode::UP;
		int idx = i + sx * (sy-1);
		vcg[idx] = vcg[idx] & ~VCGDirectionCode::DOWN;
	}
	for (uint64_t i = 0; i < sy; i++) {
		int idx = sx * i;
		vcg[idx] = vcg[idx] & ~VCGDirectionCode::LEFT;
		idx = (sx-1) + sx * i;
		vcg[idx] = vcg[idx] & ~VCGDirectionCode::RIGHT;
	}

	// clockwise for outer boundaries
	// counterclockwise for inner boundaries
	bool clockwise = true;
	int64_t start_node = 0;

	// corresponds to VCGDirectionCodes
	int64_t move_amt[9];
	move_amt[VCGDirectionCode::NONE] = 0;
	move_amt[VCGDirectionCode::RIGHT] = 1;
	move_amt[VCGDirectionCode::LEFT] = -1;
	move_amt[VCGDirectionCode::DOWN] = static_cast<int64_t>(sx);
	move_amt[VCGDirectionCode::UP] = -static_cast<int64_t>(sx);

	int64_t move_amt_x[9];
	move_amt_x[VCGDirectionCode::NONE] = 0;
	move_amt_x[VCGDirectionCode::RIGHT] = 1;
	move_amt_x[VCGDirectionCode::LEFT] = -1;
	move_amt_x[VCGDirectionCode::DOWN] = 0;
	move_amt_x[VCGDirectionCode::UP] = 0;

	int64_t move_amt_y[9];
	move_amt_y[VCGDirectionCode::NONE] = 0;
	move_amt_y[VCGDirectionCode::RIGHT] = 0;
	move_amt_y[VCGDirectionCode::LEFT] = 0;
	move_amt_y[VCGDirectionCode::DOWN] = 1;
	move_amt_y[VCGDirectionCode::UP] = -1;

	// Moore Neighbor Tracing variation
	int64_t y = 0; // breaking abstraction to save a frequent division
	while (G.next_contour(start_node, y)) {

		std::vector<std::pair<uint16_t, uint16_t>> connected_component;

		int64_t node = start_node;
		uint8_t allowed_dirs = vcg[node] & 0b1111;
		uint8_t next_move, ending_orientation;

		uint64_t nodes_already_visited = (vcg[node] >> 4) > 0;

		int64_t loop_y = node / sx;
		int64_t loop_x = (node - sx * loop_y);

		if (allowed_dirs == VCGDirectionCode::NONE) {
			vcg[node] |= VISITED_BIT;
			connected_component.emplace_back(loop_x, loop_y);
		}
		else {
			connected_component.reserve(100);

			// ENABLE THIS LINE IF YOU WANT TO BE ABLE TO RECOVER
			// INDIVIDUAL CONTOURS IN POSTPROCESSING
			// This allows you to find the beginning and end of a contour
			// by duplicating a point at the beginning and end of a contour.

			// connected_component.emplace_back(loop_x, loop_y);

			// go counterclockwise for |x  vs clockwise for x|
			next_move = VCGDirectionCode::UP;
			clockwise = ((vcg[start_node] & 0b1) == 0) || (vcg[start_node] == 0b11100);

			ending_orientation = compute_next_move(
				clockwise, next_move, allowed_dirs
			);
			next_move = ending_orientation;

			loop_y = node / sx;
			loop_x = (node - sx * loop_y);

			do {
				node += move_amt[next_move];
				loop_x += move_amt_x[next_move];
				loop_y += move_amt_y[next_move];
				connected_component.emplace_back(loop_x, loop_y);
				uint8_t is_visited = (vcg[node] >> 4) > 0;
				nodes_already_visited += is_visited;
				vcg[node] |= VISITED_BIT;
				allowed_dirs = vcg[node] & 0b1111;

				next_move = compute_next_move(
					clockwise, next_move, allowed_dirs
				);
			} while (
				!(node == start_node && next_move == ending_orientation)
			);
		}

		start_node++;

		if (connected_component.size() == 0 || connected_component.size() == nodes_already_visited) {
			continue;
		}

		contours.push_back(connected_component);
	}

	return contours;
}

template <typename LABEL>
std::unordered_map<LABEL, std::vector<uint16_t>>
extract_contours(
	const LABEL* labels,
	const int64_t sx, const int64_t sy, const int64_t sz
) {
	std::unordered_map<
		LABEL, 
		std::vector<uint16_t>
	> contours;

	contours.reserve(sz * 10);

	const int64_t sxy = sx * sy;

	std::vector<uint8_t> boundaries(sxy);

	for (int64_t y = 0; y < sy; y++) {
		for (int64_t x = 0; x < sx; x++) {
			LABEL label = labels[x + sx * y];
			auto& ctr = contours[label];
			ctr.push_back(x);
			ctr.push_back(y);
			ctr.push_back(0);
		}
	}

	// Note: z = 0, z = sz - 1 need special handling
	// if you want the shell to be capped in the z direction.
	for (int64_t z = 1; z < sz - 1; z++) {
		std::fill(boundaries.begin(), boundaries.end(), 0);
		create_boundary_map(
			(labels + sxy * z), 
			boundaries,
			sx, sy
		);

		auto slice_contours = extract_contours_helper(
			boundaries, sx, sy
		);

		for (auto& cc_contour : slice_contours) {
			int64_t x = cc_contour[0].first;
			int64_t y = cc_contour[0].second;
			int64_t loc = x + sx * y + sxy * z;
			LABEL label = labels[loc];

			auto& ctr = contours[label];
			for (auto coord : cc_contour) {
				ctr.push_back(coord.first);
				ctr.push_back(coord.second);
				ctr.push_back(z);
			}
		}
	}

	for (int64_t y = 0; y < sy; y++) {
		for (int64_t x = 0; x < sx; x++) {
			LABEL label = labels[x + sx * y + sxy * (sz-1)];
			auto& ctr = contours[label];
			ctr.push_back(x);
			ctr.push_back(y);
			ctr.push_back(sz-1);
		}
	}

	return contours;
}

};
};

#endif