package com.test.euler;

public class Problem21 {

	public static void main(String[] args) {
		System.out.println(Problem21.getSumOfAmicableNumbers(10000));
	}

	private static int getSumOfAmicableNumbers(int n) {
		int sum = 0;
		for (int i = 1; i <= n; i++) {
			if (isAmicable(i)) {
				sum += i;
			}
		}
		return sum;
	}

	private static boolean isAmicable(int n) {
		int m = divisorSum(n);
		if (m != n && divisorSum(m) == n) {
			return true;
		}
		return false;
	}

	private static int divisorSum(int n) {
		int sum = 0;
		for (int i = 1; i < n; i++) {
			if (n % i == 0) {
				sum += i;
			}
		}
		return sum;
	}
}
