package com.test.euler;

public class Problem4 {

	public static void main(String[] args) {
		int noOfDigits = 3;
		int minValue = (int) Math.pow(10, noOfDigits - 1);
		int maxValue = (int) (Math.pow(10, noOfDigits) - 1);
		System.out.println(minValue + ":" + maxValue);		
		long largestPallindrome = 0L;
		for (int i = maxValue; i >= minValue; i--) {
			for (int j = maxValue; j >= minValue; j--) {
				long number = i * j;
				if (isPallindrome(String.valueOf(number))) {
					if (isLargerThanPrevious(number, largestPallindrome)) {
						largestPallindrome = number;
						break;
					}
				}
			}
		}		
		System.out.println(largestPallindrome);
	}

	private static boolean isLargerThanPrevious(long number, long largestPallindrome) {
		if (number > largestPallindrome) {
			return true;
		}
		return false;
	}

	private static boolean isPallindrome(String number) {
		for (int i = 0; i < number.length() / 2; i++) {
			if (number.charAt(i) != number.charAt(number.length() - i - 1)) {
				return false;
			}
		}
		return true;
	}
}
