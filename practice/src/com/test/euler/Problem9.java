package com.test.euler;

public class Problem9 {

	public static void main(String[] args) {
		int sum = 1000;
		for (int a = 1; a < sum; a++) {
			for (int b = a + 1; b < sum; b++) {
				int c = sum - a - b;
				if (a * a + b * b == c * c) {
					System.out.println(
							"a: " + a + ", b: " + b + ", c: " + c + ", product : " + Integer.toString(a * b * c));
				}
			}
		}
	}
	
}
