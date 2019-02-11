#include <cstdio>
#include <cstdint>
#include "ipt.hpp"

void print(int* arr, int sx, int sy) {
	for (int y = 0; y < sy; y++) {
		for (int x = 0; x < sx; x++) {
			printf("%i ", arr[x + sx * y]);
		}
		printf("\n");
	}
	printf("\n");
}

void print(int* arr, int sx, int sy, int sz) {
	if (sx == 1) {
		print(arr, sy, sz);
		return;
	}

	for (int z = 0; z < sz; z++) {
		for (int y = 0; y < sy; y++) {
			for (int x = 0; x < sx; x++) {
				printf("%i ", arr[x + sx * y + sx * sy * z]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

int main () {
	int sx = 10;
	int sy = 10;
	int sz = 3;
	int sw = 3;

	float *arr = new float[sx * sy * sz * sw]();
	
	for (int i = 0; i < sx * sy * sz * sw; i++) {
		arr[i] = (float)i;
	}
	
	// print(arr, sx, sy, sz);
	// ipt::rect_ipt_2d<int>(arr, sx, sy);
	// ipt::rect_ipt_3d<int>(arr, sx, sy, sz);
	ipt::rect_ipt_4d<float>(arr, sx, sy, sz, sw);
	// print(arr, sz, sy, sx);
	delete []arr;

	return 0;
}