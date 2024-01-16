package com.test.euler;

import java.math.BigInteger;

public class Problem20 {

	public static void main(String[] args) {
		System.out.println(Problem20.factorialOfANumber(100));
	}

	public static int factorialOfANumber(int n) {
		BigInteger result = BigInteger.ONE;
		int sum = 0;
		for (int i = 2; i <= n; i++) {
			result = result.multiply(BigInteger.valueOf(i));
		}
		
		System.out.println(result.toString());
		
		char[] digits = result.toString().toCharArray();

		for (int i = 0; i < digits.length; i++) {
			sum += (digits[i] - 48);
		}

		return sum;
	}

}
