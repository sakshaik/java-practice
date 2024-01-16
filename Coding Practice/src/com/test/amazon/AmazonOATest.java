package com.test.amazon;

import java.util.Arrays;
import java.util.List;

public class AmazonOATest {

	public static void main(String[] args) {
		int[] averageUtil = { 25, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 76, 80 };
		System.out.println(AmazonOATest.finalInstances(2, averageUtil));
		System.out.println(AmazonOATest.countPossibleSegments(3, Arrays.asList(1, 5, 4)));
		System.out.println(AmazonOATest.countPossibleSegments(3, Arrays.asList(5, 1, 1, 4)));
		System.out.println(AmazonOATest.countPossibleSegments(1, Arrays.asList(1, 2, 3, 4, 5)));
		System.out.println(AmazonOATest.countPossibleSegments(10, Arrays.asList(1, 2, 3, 4, 5, 20)));
		System.out.println(AmazonOATest.getNoOfInclusions(Arrays.asList(1, 3, 4, 6, 9), Arrays.asList(2, 8, 5, 7, 10)));
		System.out.println(AmazonOATest.getNoOfInclusions(Arrays.asList(1, 2, 3), Arrays.asList(5, 3, 4)));
		System.out.println(AmazonOATest.getNoOfInclusions(Arrays.asList(1, 3, 4), Arrays.asList(7, 7, 7)));
	}

	public static int finalInstances(int instances, int[] averageUtil) {
		if (instances > 216) {
			return instances;
		}
		for (int i = 0; i < averageUtil.length; i++) {
			boolean action = false;
			if (averageUtil[i] < 25 && instances > 1) {
				instances = instances / 2;
				action = true;
			} else if (averageUtil[i] > 60 && (instances * 2 <= 216)) {
				instances = instances * 2;
				action = true;
			}
			if (action) {
				i += 10;
			}
		}
		return instances;
	}

	public static long countPossibleSegments(int k, List<Integer> weights) {
		long[] count = new long[1];
		getSubArrays(weights, 0, 0, k, count);
		return count[0];
	}

	private static void getSubArrays(List<Integer> weights, int start, int end, int k, long[] count) {
		if (end == weights.size())
			return;
		else if (start > end)
			getSubArrays(weights, 0, end + 1, k, count);
		else {
			if (end - start == 0) {
				count[0]++;
			} else {
				int min = weights.get(start);
				int max = weights.get(start);
				for (int i = start; i <= end; i++) {
					if (min > weights.get(i)) {
						min = weights.get(i);
					}
					if (max < weights.get(i)) {
						max = weights.get(i);
					}
				}
				if ((max - min) <= k) {
					count[0]++;
				}
			}
			getSubArrays(weights, start + 1, end, k, count);
		}
		return;
	}

	public static int getNoOfInclusions(List<Integer> regionStart, List<Integer> regionEnd) {
		if (regionStart == null || regionStart.size() == 0 || regionEnd == null || regionEnd.size() == 0
				|| regionStart.size() != regionEnd.size()) {
			return 0;
		}
		int size = regionStart.size();
		RegionPair[] pairs = new RegionPair[size];
		for (int i = 0; i < size; i++) {
			pairs[i] = new RegionPair(regionStart.get(i), regionEnd.get(i));
		}
		return isIntersect(pairs, pairs.length);
	}

	static int isIntersect(RegionPair arr[], int n) {
		int max_ele = 0;
		for (int i = 0; i < n; i++) {
			if (max_ele < arr[i].end)
				max_ele = arr[i].end;
		}
		int[] aux = new int[max_ele + 1];
		for (int i = 0; i < n; i++) {
			int x = arr[i].start;
			int y = arr[i].end;
			aux[x]++;
			aux[y]--;
		}
		int count = 0;
		for (int i = 1; i < max_ele; i++) {
			aux[i] += aux[i - 1];
			if (aux[i] == 0) {
				count++;
			}
		}
		return count;
	}

	static class RegionPair {
		int start;
		int end;

		RegionPair(int a, int b) {
			start = a;
			end = b;
		}
	}
}
