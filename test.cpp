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


void print(float* arr, int sx, int sy, int sz, int sw) {
	for (int w = 0; w < sw; w++) {
		for (int z = 0; z < sz; z++) {
			for (int y = 0; y < sy; y++) {
				for (int x = 0; x < sx; x++) {
					printf("%.1f ", arr[x + sx * y + sx * sy * z]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

int main () {
	int sx = 7;
	int sy = 3;
	int sz = 5;
	int sw = 1;

	int *arr = new int[sx * sy * sz * sw]();
	
	for (int i = 0; i < sx * sy * sz * sw; i++) {
		arr[i] = (int)i;
	}
	
	print(arr, sx, sy, sz);
	// ipt::rect_ipt_2d<int>(arr, sx, sy);
	// ipt::rect_ipt_3d<int>(arr, sx, sy, sz);
	ipt::rect_ipt<int>(arr, sx, sy, sz);
	print(arr, sz, sy, sx);
	delete []arr;

	return 0;
}