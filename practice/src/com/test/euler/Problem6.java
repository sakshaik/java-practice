package com.test.euler;

// https://projecteuler.net/problem=6
public class Problem6 {

	public static void main(String[] args) {
		int maxValue = 1000;
		int sumOfSquares = 0;
		int sumOfNumbers = 0;
		for (int i = 1; i <= maxValue; i++) {
			sumOfSquares += Math.pow(i, 2);
			sumOfNumbers += i;
		}
		System.out.println((sumOfNumbers * sumOfNumbers) - sumOfSquares);
	}

}
