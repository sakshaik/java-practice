package com.test.euler;

import java.math.BigInteger;

public class Problem16 {

	public static void main(String[] args) {
		int power = 1000;
		int powerOf = 2;
		BigInteger number = new BigInteger(String.valueOf(powerOf));
		String powerValue = number.pow(power).toString();
		char[] numbers = powerValue.toCharArray();
		int sum = 0;

		for (int i = 0; i < numbers.length; i++) {
			sum += Character.getNumericValue(numbers[i]);
		}

		System.out.println(powerValue + ":\n" + sum);
	}
}
